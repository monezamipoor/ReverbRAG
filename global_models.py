from typing import Optional, Tuple

import torch
import torch.nn as nn
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding


def fs_to_stft_params(fs: int):
    if fs == 48000:
        return dict(N_freq=513, hop_len=256, win_len=512)
    elif fs == 16000:
        return dict(N_freq=257, hop_len=128, win_len=256)
    return dict(N_freq=513, hop_len=256, win_len=512)


def _act(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.LeakyReLU(0.1, inplace=True)


def _build_mlp(
    in_dim: Optional[int],
    hidden: list,
    out_dim: int,
    act: str = "lrelu",
    dropout: float = 0.0,
    out_act: bool = False,
) -> nn.Sequential:
    layers = []
    hidden = list(hidden)
    if len(hidden) == 0:
        if in_dim is None:
            layers.append(nn.LazyLinear(int(out_dim)))
        else:
            layers.append(nn.Linear(int(in_dim), int(out_dim)))
        if out_act:
            layers.append(_act(act))
        return nn.Sequential(*layers)

    if in_dim is None:
        layers.append(nn.LazyLinear(int(hidden[0])))
    else:
        layers.append(nn.Linear(int(in_dim), int(hidden[0])))
    layers.append(_act(act))
    if dropout > 0.0:
        layers.append(nn.Dropout(float(dropout)))

    last = int(hidden[0])
    for h in hidden[1:]:
        layers.append(nn.Linear(last, int(h)))
        layers.append(_act(act))
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        last = int(h)
    layers.append(nn.Linear(last, int(out_dim)))
    if out_act:
        layers.append(_act(act))
    return nn.Sequential(*layers)


class GlobalDualBranchModel(nn.Module):
    """
    Shared global encoder + two branches:
      - LF branch: global-only (no time encoding)
      - HF branch: global + time encoding
    Produces [B, C, F, T] log-magnitude prediction.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._logger = None
        self.debug_outputs = {}

        aabb = getattr(cfg, "scene_aabb", None)
        aabb = aabb.clone().detach().float() if aabb is not None else torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32
        )
        self.register_buffer("aabb", aabb, persistent=False)

        stft = fs_to_stft_params(cfg.sample_rate)
        self.N_freq = int(stft["N_freq"])
        self.n_channels = 1 if cfg.database == "raf" else 2
        self.T_default = 60

        # Keep trainer compatibility.
        self.use_temporal_attention = False
        self.temporal_stack = None
        self.rag_gen = None

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        d_pos = int(self.position_encoding.get_out_dim())
        d_rot = int(self.rot_encoding.get_out_dim())
        d_time = int(self.time_encoding.get_out_dim())

        gcfg = (cfg.opt.get("global_dual_branch", {}) or {})

        # g0 = [mic_e, src_e, rot_e, visual_feat] (raw visual features)
        # visual_feat dim may vary by dataset setup, so we keep the first layer lazy.
        g0_fixed_dim = (2 * d_pos) + d_rot

        # Shared global encoder g0 -> g1
        ge = gcfg.get("global_encoder", {}) or {}
        g_hidden = list(ge.get("hidden", [2048, 1024, 1024]))
        g_out = int(ge.get("out_dim", int(getattr(cfg, "W_field", 1024))))
        g_act = str(ge.get("act", "lrelu"))
        g_drop = float(ge.get("dropout", 0.0))
        self.global_encoder = _build_mlp(
            in_dim=None, hidden=g_hidden, out_dim=g_out, act=g_act, dropout=g_drop, out_act=True
        )
        self.g1_dim = g_out
        self.g0_fixed_dim = g0_fixed_dim

        # Frequency split
        lf_cfg = gcfg.get("lf_branch", {}) or {}
        cutoff_hz = float(lf_cfg.get("cutoff_hz", 600.0))
        n_fft = int((self.N_freq - 1) * 2)
        hz_per_bin = float(cfg.sample_rate) / float(n_fft)
        base_split = int(cutoff_hz / max(hz_per_bin, 1e-8))
        base_split = max(1, min(self.N_freq - 1, base_split + 1))  # n low bins (include DC)

        xcfg = gcfg.get("crossover", {}) or {}
        self.cross_mode = str(xcfg.get("mode", "hard")).lower()
        trans_bins = int(xcfg.get("transition_bins", 0))
        if self.cross_mode != "soft":
            trans_bins = 0
        trans_bins = max(0, trans_bins)

        self.hf_lo = max(0, base_split - trans_bins)
        self.lf_hi = min(self.N_freq, base_split + trans_bins)
        self.n_lf_pred = int(self.lf_hi)
        self.n_hf_pred = int(self.N_freq - self.hf_lo)

        if self.n_lf_pred <= 0 or self.n_hf_pred <= 0:
            raise ValueError(
                f"Invalid frequency partition: N_freq={self.N_freq}, base_split={base_split}, "
                f"transition_bins={trans_bins}, n_lf_pred={self.n_lf_pred}, n_hf_pred={self.n_hf_pred}"
            )

        # LF decoder: predicts low part for all frames at once.
        lf_hidden = list(lf_cfg.get("decoder_hidden", [1024, 1024]))
        lf_act = str(lf_cfg.get("act", "lrelu"))
        lf_drop = float(lf_cfg.get("dropout", 0.0))
        lf_out_dim = self.n_channels * self.n_lf_pred * self.T_default
        self.lf_decoder = _build_mlp(
            in_dim=self.g1_dim, hidden=lf_hidden, out_dim=lf_out_dim, act=lf_act, dropout=lf_drop, out_act=False
        )

        # HF path: time-conditioned
        hf_cfg = gcfg.get("hf_branch", {}) or {}
        hf_enc_hidden = list(hf_cfg.get("time_encoder_hidden", [2048, 1024]))
        hf_token_dim = int(hf_cfg.get("token_dim", int(getattr(cfg, "W_field", 1024))))
        hf_dec_hidden = list(hf_cfg.get("decoder_hidden", [1024]))
        hf_act = str(hf_cfg.get("act", "lrelu"))
        hf_drop = float(hf_cfg.get("dropout", 0.0))

        self.hf_time_encoder = _build_mlp(
            in_dim=self.g1_dim + d_time,
            hidden=hf_enc_hidden,
            out_dim=hf_token_dim,
            act=hf_act,
            dropout=hf_drop,
            out_act=True,
        )
        self.hf_token_norm = nn.LayerNorm(hf_token_dim) if bool((gcfg.get("norm", {}) or {}).get("token_layernorm", True)) else nn.Identity()
        self.hf_decoder = _build_mlp(
            in_dim=hf_token_dim,
            hidden=hf_dec_hidden,
            out_dim=(self.n_channels * self.n_hf_pred),
            act=hf_act,
            dropout=hf_drop,
            out_act=False,
        )

    def set_logger(self, logger):
        self._logger = logger

    def _build_global_input(
        self,
        mic_xyz: torch.Tensor,
        src_xyz: torch.Tensor,
        head_dir: torch.Tensor,
        visual_feat: torch.Tensor,
    ) -> torch.Tensor:
        mic_n = SceneBox.get_normalized_positions(mic_xyz, self.aabb)
        src_n = SceneBox.get_normalized_positions(src_xyz, self.aabb)

        mic_e = self.position_encoding(mic_n)
        src_e = self.position_encoding(src_n)
        rot_e = self.rot_encoding(head_dir)
        if visual_feat.ndim == 1:
            visual_feat = visual_feat.unsqueeze(0)
        return torch.cat([mic_e, src_e, rot_e, visual_feat], dim=-1)

    def _hf_tokens(self, g1: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        B, T = t_idx.shape[0], t_idx.shape[1]
        if T > 1:
            t_norm = (t_idx.float() / float(T - 1)).clamp(0.0, 1.0)
        else:
            t_norm = (t_idx.float() / float(self.T_default - 1)).clamp(0.0, 1.0)
        t_e = self.time_encoding(t_norm.reshape(B * T, 1)).view(B, T, -1)
        g1e = g1.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([g1e, t_e], dim=-1).reshape(B * T, -1)
        h = self.hf_time_encoder(x).view(B, T, -1)
        return self.hf_token_norm(h)

    def _merge_branches(self, lf_part: torch.Tensor, hf_part: torch.Tensor) -> torch.Tensor:
        """
        lf_part: [B,C,n_lf_pred,T], covering bins [0:lf_hi]
        hf_part: [B,C,n_hf_pred,T], covering bins [hf_lo:N_freq]
        """
        B, C, _, T = lf_part.shape
        out = torch.zeros(B, C, self.N_freq, T, device=lf_part.device, dtype=lf_part.dtype)

        hf_lo = int(self.hf_lo)
        lf_hi = int(self.lf_hi)
        ov = max(0, lf_hi - hf_lo)

        # low non-overlap
        if hf_lo > 0:
            out[:, :, :hf_lo, :] = lf_part[:, :, :hf_lo, :]
        # high non-overlap
        if lf_hi < self.N_freq:
            out[:, :, lf_hi:, :] = hf_part[:, :, ov:, :]

        # overlap (or boundary bin when ov=0 with hard split)
        if ov > 0:
            lf_ov = lf_part[:, :, hf_lo:lf_hi, :]
            hf_ov = hf_part[:, :, :ov, :]
            if self.cross_mode == "soft":
                alpha = torch.linspace(0.0, 1.0, steps=ov, device=out.device, dtype=out.dtype).view(1, 1, ov, 1)
                out[:, :, hf_lo:lf_hi, :] = (1.0 - alpha) * lf_ov + alpha * hf_ov
            else:
                out[:, :, hf_lo:lf_hi, :] = hf_ov

        return out

    def forward(
        self,
        mic_xyz: torch.Tensor,
        src_xyz: torch.Tensor,
        head_dir: torch.Tensor,
        t_idx: torch.Tensor,
        visual_feat: torch.Tensor,
        refs_logmag: torch.Tensor = None,
        refs_mask: torch.Tensor = None,
        refs_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        del refs_logmag, refs_mask, refs_feats  # intentionally unused in this model
        self.debug_outputs = {}

        B, T = t_idx.shape[0], t_idx.shape[1]
        if T != self.T_default:
            raise RuntimeError(
                f"GlobalDualBranchModel expects T={self.T_default} frames at forward time, got T={T}. "
                "Use full-sequence mode (or EDCFullBatchSampler packing) for this model."
            )

        g0 = self._build_global_input(mic_xyz, src_xyz, head_dir, visual_feat)
        g1 = self.global_encoder(g0)  # [B, g1_dim]

        # LF branch: one shot over time
        lf_flat = self.lf_decoder(g1)  # [B, C*n_lf_pred*T]
        lf = lf_flat.view(B, self.n_channels, self.n_lf_pred, self.T_default)
        lf = torch.tanh(lf) * 10.0

        # HF branch: time-conditioned
        h = self._hf_tokens(g1, t_idx)  # [B,T,D]
        hf_flat = self.hf_decoder(h.reshape(B * T, -1))  # [B*T, C*n_hf_pred]
        hf = hf_flat.view(B, T, self.n_channels, self.n_hf_pred).permute(0, 2, 3, 1).contiguous()
        hf = torch.tanh(hf) * 10.0

        pred = self._merge_branches(lf, hf)

        self.debug_outputs["global_latent"] = g1
        self.debug_outputs["lf_pred"] = lf
        self.debug_outputs["hf_pred"] = hf
        self.debug_outputs["split"] = {
            "hf_lo": int(self.hf_lo),
            "lf_hi": int(self.lf_hi),
            "cross_mode": self.cross_mode,
        }

        return pred


class GlobalSeq2SeqUpscaleModel(nn.Module):
    """
    Seq2Seq transformer model:
      1) Build global latent token g1 from scene/pose/dir/visual context.
      2) Encoder input = [g1 token, 60 time tokens] with bidirectional self-attention.
      3) Decoder uses causal self-attention + cross-attention to encoder memory.
      4) Predict low-frequency STFT bins for 60 slices.
      5) Per-slice/channel upscaler MLP maps low-band -> full N_freq bins.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._logger = None
        self.debug_outputs = {}

        aabb = getattr(cfg, "scene_aabb", None)
        aabb = aabb.clone().detach().float() if aabb is not None else torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32
        )
        self.register_buffer("aabb", aabb, persistent=False)

        stft = fs_to_stft_params(cfg.sample_rate)
        self.N_freq = int(stft["N_freq"])
        self.n_channels = 1 if cfg.database == "raf" else 2
        self.T_default = 60

        # Keep trainer compatibility and force packed full-sequence training path.
        self.use_temporal_attention = True
        self.temporal_stack = None
        self.rag_gen = None

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")
        d_pos = int(self.position_encoding.get_out_dim())
        d_rot = int(self.rot_encoding.get_out_dim())

        mcfg = (cfg.opt.get("global_seq2seq", {}) or {})
        d_model = int(mcfg.get("d_model", 384))
        n_heads = int(mcfg.get("n_heads", 6))
        n_enc_layers = int(mcfg.get("n_enc_layers", 4))
        n_dec_layers = int(mcfg.get("n_dec_layers", 4))
        ff_mult = int(mcfg.get("ff_mult", 4))
        dropout = float(mcfg.get("dropout", 0.1))
        ff_dim = int(d_model * ff_mult)
        act_name = str(mcfg.get("act", "gelu")).lower()
        tf_act = "gelu" if act_name in ("gelu", "silu") else "relu"

        # g0 = [mic_e, src_e, rot_e, visual_feat] -> g1 token of size d_model.
        ge_cfg = mcfg.get("global_encoder", {}) or {}
        ge_hidden = list(ge_cfg.get("hidden", [1536, 1024, 512]))
        ge_act = str(ge_cfg.get("act", "lrelu"))
        ge_drop = float(ge_cfg.get("dropout", 0.0))
        self.global_encoder = _build_mlp(
            in_dim=None, hidden=ge_hidden, out_dim=d_model, act=ge_act, dropout=ge_drop, out_act=True
        )
        self.g0_fixed_dim = (2 * d_pos) + d_rot
        self.d_model = d_model

        # Low-band width from cutoff.
        low_cfg = mcfg.get("low_freq", {}) or {}
        cutoff_hz = float(low_cfg.get("cutoff_hz", 900.0))
        n_fft = int((self.N_freq - 1) * 2)
        hz_per_bin = float(cfg.sample_rate) / float(max(n_fft, 1))
        n_low = int(cutoff_hz / max(hz_per_bin, 1e-8)) + 1  # include DC
        self.n_low = max(8, min(self.N_freq - 1, n_low))

        # Encoder tokens: 1 global + 60 learned time tokens.
        self.enc_time_tokens = nn.Parameter(torch.randn(self.T_default, d_model) * 0.02)
        self.enc_pos_embed = nn.Parameter(torch.randn(1 + self.T_default, d_model) * 0.02)

        # Decoder query tokens and positional embedding.
        self.dec_query_tokens = nn.Parameter(torch.randn(self.T_default, d_model) * 0.02)
        self.dec_pos_embed = nn.Parameter(torch.randn(self.T_default, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=tf_act,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=tf_act,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        # Decoder -> low-band per channel.
        low_head_cfg = low_cfg.get("head_hidden", [512]) or []
        self.low_head = _build_mlp(
            in_dim=d_model,
            hidden=list(low_head_cfg),
            out_dim=self.n_channels * self.n_low,
            act=str(low_cfg.get("act", "lrelu")),
            dropout=float(low_cfg.get("dropout", 0.0)),
            out_act=False,
        )

        # Upscaler: low-band -> full-band for each (B, C, T) slice.
        up_cfg = mcfg.get("upscaler", {}) or {}
        up_hidden = list(up_cfg.get("hidden", [512, 768]))
        up_act = str(up_cfg.get("act", "lrelu"))
        up_drop = float(up_cfg.get("dropout", 0.0))
        self.upscaler = _build_mlp(
            in_dim=self.n_low,
            hidden=up_hidden,
            out_dim=self.N_freq,
            act=up_act,
            dropout=up_drop,
            out_act=False,
        )

    def set_logger(self, logger):
        self._logger = logger

    def _build_global_input(
        self,
        mic_xyz: torch.Tensor,
        src_xyz: torch.Tensor,
        head_dir: torch.Tensor,
        visual_feat: torch.Tensor,
    ) -> torch.Tensor:
        mic_n = SceneBox.get_normalized_positions(mic_xyz, self.aabb)
        src_n = SceneBox.get_normalized_positions(src_xyz, self.aabb)
        mic_e = self.position_encoding(mic_n)
        src_e = self.position_encoding(src_n)
        rot_e = self.rot_encoding(head_dir)
        if visual_feat.ndim == 1:
            visual_feat = visual_feat.unsqueeze(0)
        return torch.cat([mic_e, src_e, rot_e, visual_feat], dim=-1)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        mic_xyz: torch.Tensor,
        src_xyz: torch.Tensor,
        head_dir: torch.Tensor,
        t_idx: torch.Tensor,
        visual_feat: torch.Tensor,
        refs_logmag: torch.Tensor = None,
        refs_mask: torch.Tensor = None,
        refs_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        del refs_logmag, refs_mask, refs_feats  # intentionally unused in this model
        self.debug_outputs = {}

        B, T = t_idx.shape[0], t_idx.shape[1]
        if T != self.T_default:
            raise RuntimeError(
                f"GlobalSeq2SeqUpscaleModel expects T={self.T_default} frames, got T={T}. "
                "Use grouped full-sequence training/eval."
            )

        # Global context token.
        g0 = self._build_global_input(mic_xyz, src_xyz, head_dir, visual_feat)
        g1 = self.global_encoder(g0)  # [B, D]
        gtok = g1.unsqueeze(1)  # [B,1,D]

        # Encoder sequence: [global token; 60 learned time tokens]
        time_tok = self.enc_time_tokens.unsqueeze(0).expand(B, -1, -1)  # [B,60,D]
        enc_in = torch.cat([gtok, time_tok], dim=1) + self.enc_pos_embed.unsqueeze(0)
        memory = self.encoder(enc_in)  # [B,61,D]

        # Decoder query sequence (60 steps), causal.
        dec_q = self.dec_query_tokens.unsqueeze(0).expand(B, -1, -1) + self.dec_pos_embed.unsqueeze(0)
        mask = self._causal_mask(self.T_default, dec_q.device)
        dec_out = self.decoder(tgt=dec_q, memory=memory, tgt_mask=mask)  # [B,60,D]

        # Low-band prediction.
        low_flat = self.low_head(dec_out.reshape(B * self.T_default, self.d_model))
        low_btcf = low_flat.view(B, self.T_default, self.n_channels, self.n_low)
        low = low_btcf.permute(0, 2, 3, 1).contiguous()  # [B,C,F_low,T]
        low = torch.tanh(low) * 10.0

        # Upscale low -> full for each (B,C,T).
        low_bctf = low.permute(0, 1, 3, 2).contiguous()  # [B,C,T,F_low]
        up_in = low_bctf.reshape(B * self.n_channels * self.T_default, self.n_low)
        full_flat = self.upscaler(up_in)
        full = full_flat.view(B, self.n_channels, self.T_default, self.N_freq).permute(0, 1, 3, 2).contiguous()
        full = torch.tanh(full) * 10.0

        self.debug_outputs["global_latent"] = g1
        self.debug_outputs["low_pred"] = low
        self.debug_outputs["memory"] = memory
        return full
