# model.py
# Unified NVAS wrapper with faithful NeRAF path and an AV-NeRF-style variant for RIR.
# - NeRAF mode: exactly as before (orientation SH in encoder; NeRAF-style heads).
# - AV-NeRF mode: orientation SH goes to the DECODER (FiLM on W), not the encoder.
# - Time index is normalized INSIDE the model (full: / (T-1); slice: / 59).
# - Visual features:
#     * NeRAF mode uses global 1024-D features (as before).
#     * AV-NeRF mode prefers per-pose features from dataset; still accepts 1024-D fallback.

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from generator import ReverbRAGGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.data.scene_box import SceneBox


# -----------------------
# Minimal config object
# -----------------------
@dataclass
class ModelConfig:
    baseline: Literal["neraf", "avnerf"] = "neraf"
    database: Literal["raf", "soundspaces"] = "raf"
    scene_root: str = "../NeRAF/data/RAF"
    scene_name: str = "FurnishedRoom"
    sample_rate: int = 48000
    W_field: int = 1024
    scene_aabb: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0],
                                             [1.0, 1.0, 1.0]], dtype=torch.float32)
    opt: dict = field(default_factory=dict)
# -----------------------


def fs_to_stft_params(fs: int):
    if fs == 48000:
        return dict(N_freq=513, hop_len=256, win_len=512)
    elif fs == 16000:
        return dict(N_freq=257, hop_len=128, win_len=256)
    # default to 48k spec
    return dict(N_freq=513, hop_len=256, win_len=512)


class NeRAFEncoder(nn.Module):
    """
    NeRAF encoder; takes global context c and time indices, and outputs tokens.
    """

    def __init__(
        self,
        W: int,
        visual_dim: int = 1024,
        use_aux: bool = False,
        hidden: Optional[list] = None,
    ):
        super().__init__()
        self.use_aux = use_aux
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.position_encoding = NeRFEncoding(  # still used externally via Unified for c
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")  # used for c in Unified

        d_time = self.time_encoding.get_out_dim()
        d_pos = self.position_encoding.get_out_dim()
        d_rot = self.rot_encoding.get_out_dim()
        # c = [mic_e, src_e, rot_e, vis_e]
        self.static_dim = (2 * d_pos) + d_rot + visual_dim

        in_size = d_time + self.static_dim + (W if self.use_aux else 0)

        # ---- configurable trunk MLP ----
        hidden = list(hidden) if hidden is not None else [5096, 2048, 1024]
        layers = []
        last = in_size
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            last = h
        # final projection to W + activation (kept same behaviour)
        layers.append(nn.Linear(last, W))
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.trunk = nn.Sequential(*layers)

    def forward(
        self,
        c: torch.Tensor,                 # [B, static_dim], built in Unified
        t_idx: torch.Tensor,             # [B, T]
        aux: Optional[torch.Tensor] = None,  # [B, W] when use_aux=True
    ) -> torch.Tensor:
        B, T = t_idx.shape[0], t_idx.shape[1]

        # normalize time inside model
        if T > 1:
            t_norm = (t_idx.float() / float(T - 1)).clamp(0.0, 1.0)
        else:
            t_norm = (t_idx.float() / 59.0).clamp(0.0, 1.0)

        static = c
        if self.use_aux:
            if aux is None:
                aux = torch.zeros(B, static.shape[-1], device=static.device, dtype=static.dtype)
            static = torch.cat([static, aux], dim=-1)

        t_e = self.time_encoding(t_norm.reshape(B * T, 1)).view(B, T, -1)
        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])
        enc = torch.cat([t_e, static_exp], dim=-1)                       # [B,T,in_size]

        w = self.trunk(enc.reshape(B * T, -1)).view(B, T, -1)            # [B,T,W]
        return w


# --------------------------------------------------
# AV-NeRF-style Encoder (+ optional aux)
# --------------------------------------------------
class AVNeRFEncoder(nn.Module):
    """
    AV-NeRF variant; takes global context c (incl. direction) + time and outputs tokens.
    """

    def __init__(
        self,
        W: int,
        visual_dim_in: int = 1024,
        visual_dim_out: int = 128,
        dropout_p: float = 0.0,
        use_aux: bool = False,
        hidden: Optional[list] = None,
    ):
        super().__init__()
        self.use_aux = use_aux
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        # visual MLP kept as-is (you can also make this configurable later if you want)
        self.av_mlp = nn.Sequential(
            nn.Linear(visual_dim_in, 512), nn.ReLU(inplace=True),
            nn.Linear(512, visual_dim_out), nn.ReLU(inplace=True),
            nn.Linear(visual_dim_out, visual_dim_out),
        )
        self.dropout_p = dropout_p

        d_time = self.time_encoding.get_out_dim()
        d_pos  = self.position_encoding.get_out_dim()
        # c = [mic_e, src_e, rot_e, v]
        self.static_dim = (2 * d_pos) + visual_dim_out + 16  # SH dim still assumed 16

        in_size = d_time + self.static_dim + (W if self.use_aux else 0)

        # ---- configurable trunk MLP ----
        hidden = list(hidden) if hidden is not None else [5096, 2048, 1024, 1024]
        layers = []
        last = in_size
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            last = h
        layers.append(nn.Linear(last, W))
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.trunk = nn.Sequential(*layers)

    @staticmethod
    def mydropout(tensor, p=0.5, training=True):
        if not training or p == 0:
            return tensor
        b = tensor.shape[0]
        mask = (torch.rand(b, device=tensor.device) > p).float().view(b, *([1] * (tensor.ndim - 1)))
        return tensor * mask

    def forward(
        self,
        c: torch.Tensor,                 # [B, static_dim], built in Unified
        t_idx: torch.Tensor,             # [B, T]
        aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = t_idx.shape[0], t_idx.shape[1]

        if T > 1:
            t_norm = (t_idx.float() / float(T - 1)).clamp(0.0, 1.0)
        else:
            t_norm = (t_idx.float() / 59.0).clamp(0.0, 1.0)

        static = c
        if self.use_aux:
            if aux is None:
                aux = torch.zeros(B, static.shape[-1], device=static.device, dtype=static.dtype)
            static = torch.cat([static, aux], dim=-1)

        t_e = self.time_encoding(t_norm.reshape(B * T, 1)).view(B, T, -1)
        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])
        enc = torch.cat([t_e, static_exp], dim=-1)

        w = self.trunk(enc.reshape(B * T, -1)).view(B, T, -1)
        return w

# ----------------------------------------
# NeRAF-style Decoder with optional FiLM
# ----------------------------------------
class NeRAFDecoder(nn.Module):
    """
    Per-channel head(s) mapping W (+ optional dir) -> N_freq, followed by tanh*10.
    If head_hidden is non-empty, each head becomes an MLP with those hidden sizes.
    """

    def __init__(
        self,
        W: int,
        n_freq: int,
        n_channels: int,
        dir_cond_dim: int = 0,
        dropout_p: float = 0.0,
        head_hidden: Optional[list] = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.W = W
        self.dir_dim = dir_cond_dim
        self.use_dir = dir_cond_dim > 0
        self.p = dropout_p

        in_dim = W + dir_cond_dim if self.use_dir else W
        self.head_hidden = list(head_hidden) if head_hidden is not None else []

        def _build_head(in_dim: int, out_dim: int) -> nn.Module:
            if not self.head_hidden:
                return nn.Linear(in_dim, out_dim)
            layers = []
            last = in_dim
            for h in self.head_hidden:
                layers.append(nn.Linear(last, h))
                layers.append(nn.LeakyReLU(0.1, inplace=True))
                last = h
            layers.append(nn.Linear(last, out_dim))
            return nn.Sequential(*layers)

        self.heads = nn.ModuleList([_build_head(in_dim, n_freq) for _ in range(n_channels)])

    @staticmethod
    def mydropout(tensor: torch.Tensor, p: float = 0.0, training: bool = True) -> torch.Tensor:
        """
        Randomly zero entire samples in the batch (like AV-NeRF).
        """
        if (not training) or p <= 0.0:
            return tensor
        batch_size = tensor.shape[0]
        keep_mask = (torch.rand(batch_size, device=tensor.device) > p).float().view(batch_size, *([1] * (tensor.ndim - 1)))
        return tensor * keep_mask

    def forward(self, w_tokens: torch.Tensor, dir_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            w_tokens: [B, T, W]
            dir_cond: [B, T, Ddir] or [B, Ddir] or None
        Returns:
            stft: [B, C, F, T]
        """
        B, T, W = w_tokens.shape
        assert W == self.W, "Token width mismatch"
        w = w_tokens.reshape(B * T, W)

        if self.use_dir:
            if dir_cond is None:
                dir_b = torch.zeros(B, T, self.dir_dim, device=w_tokens.device, dtype=w_tokens.dtype)
            else:
                if dir_cond.ndim == 2:  # [B, Ddir] -> broadcast across T
                    dir_b = dir_cond.unsqueeze(1).expand(B, T, -1)
                else:
                    dir_b = dir_cond
                dir_b = dir_b.to(w_tokens.dtype)

            # Apply AV-NeRF style mydropout to orientation embedding
            dir_b = self.mydropout(dir_b, p=self.p, training=self.training)
            dir_flat = dir_b.reshape(B * T, self.dir_dim)
            dec_in = torch.cat([w, dir_flat], dim=-1)  # [B*T, W + Ddir]
        else:
            dec_in = w  # NeRAF path unchanged

        outs = []
        for head in self.heads:
            f_flat = head(dec_in)                 # [B*T, F]
            f_flat = torch.tanh(f_flat) * 10
            f = f_flat.view(B, T, -1).transpose(1, 2)  # [B, F, T]
            outs.append(f.unsqueeze(1))           # [B,1,F,T]
        return torch.cat(outs, dim=1)             # [B, C, F, T]


class GlobalEnvelopeHead(nn.Module):
    """
    MLP(c) -> log-envelope over time [B, T_env].
    """
    def __init__(self, in_dim: int, n_frames: int = 60,
                 hidden: Optional[list] = None,
                 act: str = "lrelu",
                 tanh_scale: float = 5.0):
        super().__init__()
        hidden = hidden or [512, 512, 512]
        layers = []
        last = in_dim

        def _act(name: str):
            name = name.lower()
            if name == "relu":  return nn.ReLU(inplace=True)
            if name == "gelu":  return nn.GELU()
            if name == "silu":  return nn.SiLU(inplace=True)
            return nn.LeakyReLU(0.1, inplace=True)

        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(_act(act))
            last = h
        layers.append(nn.Linear(last, n_frames))

        self.mlp = nn.Sequential(*layers)
        self.tanh_scale = float(tanh_scale)
        self.n_frames = int(n_frames)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        log_env = self.mlp(c)                         # [B, T_env]
        if self.tanh_scale > 0:
            log_env = self.tanh_scale * torch.tanh(log_env)
        return log_env

# ----------------------------
# Unified ReverbRAG NVAS Model
# ----------------------------
class UnifiedReverbRAGModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self._logger = None

        aabb = getattr(cfg, "scene_aabb", None)
        aabb = aabb.clone().detach().float() if aabb is not None else torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32
        )
        self.register_buffer("aabb", aabb, persistent=False)

        stft = fs_to_stft_params(cfg.sample_rate)
        self.N_freq = stft["N_freq"]
        self.n_channels = 1 if cfg.database == "raf" else 2

        rag_cfg = cfg.opt.get("reverbrag", {})
        self.fusion = str(rag_cfg.get("fusion", "film")).lower()

        # Encoder / decoder MLP configs (from YAML: cfg.opt == model subtree)
        enc_neraf_cfg = (cfg.opt.get("neraf_encoder", {}) or {})
        enc_av_cfg    = (cfg.opt.get("avnerf_encoder", {}) or {})
        dec_cfg       = (cfg.opt.get("decoder", {}) or {})

        neraf_enc_hidden  = list(enc_neraf_cfg.get("hidden", [5096, 2048, 1024]))
        avnerf_enc_hidden = list(enc_av_cfg.get("hidden", [5096, 2048, 1024, 1024]))
        dec_head_hidden   = list(dec_cfg.get("hidden", []))

        # Shared SH for direction (used in c and in decoder for AV-NeRF)
        self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")
        d_rot = self.rot_encoding.get_out_dim()

        if cfg.baseline == "neraf":
            self.encoder_kind = "neraf"
            self.encoder = NeRAFEncoder(
                W=cfg.W_field,
                visual_dim=1024,
                use_aux=(self.fusion == "input"),
                hidden=neraf_enc_hidden,
            )
            dir_cond_dim = 0  # NeRAF decoder doesn't use dir cond
            static_dim = self.encoder.static_dim
        elif cfg.baseline == "avnerf":
            self.encoder_kind = "avnerf"
            self.encoder = AVNeRFEncoder(
                W=cfg.W_field,
                visual_dim_in=1024,
                visual_dim_out=128,
                dropout_p=0.0,
                use_aux=(self.fusion == "input"),
                hidden=avnerf_enc_hidden,
            )
            dir_cond_dim = d_rot
            static_dim = self.encoder.static_dim
        else:
            raise ValueError(f"Unknown baseline {cfg.baseline}")

        self.decoder = NeRAFDecoder(
            W=cfg.W_field,
            n_freq=self.N_freq,
            n_channels=self.n_channels,
            dir_cond_dim=dir_cond_dim,
            head_hidden=dec_head_hidden,
        )

        self.use_rag = rag_cfg.get("enabled", False)
        if self.use_rag:
            self.rag_gen = ReverbRAGGenerator(cfg=rag_cfg, W=cfg.W_field)
        else:
            self.rag_gen = None

        self.token_norm = nn.LayerNorm(cfg.W_field)

        # ---- Envelope head ----
        env_cfg = cfg.opt.get("envelope", {}) or {}
        self.use_envelope = bool(env_cfg.get("enabled", False))
        self.env_combine_mode = str(env_cfg.get("combine_mode", "mag")).lower()  # NEW

        if self.use_envelope:
            hidden = list(env_cfg.get("hidden", [512, 512, 512]))
            act = str(env_cfg.get("act", "lrelu"))
            tanh_scale = float(env_cfg.get("tanh_scale", 0.0))
            n_frames = int(env_cfg.get("n_frames", 60))

            self.env_head = GlobalEnvelopeHead(
                in_dim=static_dim,
                n_frames=n_frames,
                hidden=hidden,
                act=act,
                tanh_scale=tanh_scale,
            )
            self.env_n_frames = n_frames
        else:
            self.env_head = None
            self.env_n_frames = None
        self.normalize_residual = env_cfg.get("normalize", True)
        self.use_envelope = env_cfg.get("enabled", False)
        self.debug_outputs = {}

    def set_logger(self, logger):
        self._logger = logger
        if hasattr(self.rag_gen, "set_logger"):
            self.rag_gen.set_logger(logger)
    
    def _build_context(
        self,
        mic_xyz: torch.Tensor,
        src_xyz: torch.Tensor,
        head_dir: torch.Tensor,
        visual_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Build global context c = [mic_e, src_e, rot_e, vis_e]."""
        mic_n = SceneBox.get_normalized_positions(mic_xyz, self.aabb)
        src_n = SceneBox.get_normalized_positions(src_xyz, self.aabb)

        if self.encoder_kind == "neraf":
            mic_e = self.encoder.position_encoding(mic_n)
            src_e = self.encoder.position_encoding(src_n)
            rot_e = self.encoder.rot_encoding(head_dir)
            vis_e = visual_feat                                  # [B, 1024]
        else:
            mic_e = self.encoder.position_encoding(mic_n)
            src_e = self.encoder.position_encoding(src_n)
            rot_e = self.rot_encoding(head_dir)                  # shared SH
            if visual_feat.ndim == 1:
                visual_feat = visual_feat.unsqueeze(0)
            v = self.encoder.av_mlp(visual_feat)
            v = self.encoder.mydropout(v, p=self.encoder.dropout_p, training=self.training)
            vis_e = v

        c = torch.cat([mic_e, src_e, rot_e, vis_e], dim=-1)      # [B, static_dim]
        return c

    # --------------------------------------------------
    # Helper: envelope prediction + fusion
    # --------------------------------------------------
    def _apply_envelope(
        self,
        c: torch.Tensor,
        r_log: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply envelope head and combine with residual.
        Uses t_idx to pick the correct envelope value per frame/slice.

        Args:
            c:      [B, static_dim]
            r_log:  [B, C, F, T]
            t_idx:  [B, T, 1] in full mode, [B, 1, 1] in slice mode

        Returns:
            log_pred: [B,C,F,T]
            log_env:  [B,T] or None if envelope disabled
        """
        if (not self.use_envelope) or (self.env_head is None):
            # No envelope: just pass-through residual
            return r_log, None

        B, _, _, T_cur = r_log.shape

        # ---- envelope: c -> full log_env over all frames [B, T_env] ----
        log_env_full = self.env_head(c)                       # [B, T_env]

        # ---- map t_idx -> per-frame envelope indices ----
        # t_idx: [B, T, 1] (full) or [B, 1, 1] (slice)
        if t_idx.dim() == 3:
            t_long = t_idx[..., 0].long()                     # [B, T]
        elif t_idx.dim() == 2:
            t_long = t_idx.long()                             # [B, T]
        else:
            raise ValueError(f"Unexpected t_idx shape {t_idx.shape}")

        # clamp into valid [0, T_env-1]
        max_idx = self.env_n_frames - 1 if self.env_n_frames is not None else (log_env_full.shape[1] - 1)
        t_long = t_long.clamp(0, max_idx)                     # [B, T_cur]

        # gather e_{t} for each sample/time
        # log_env_full: [B, T_env], t_long: [B, T_cur]
        log_env = log_env_full.gather(1, t_long)              # [B, T_cur]
        log_env_bt = log_env.view(B, 1, 1, T_cur)             # [B,1,1,T_cur]
        
        # store envelope predictions for diagnostics / future losses
        self.debug_outputs["log_env_full"] = log_env_full     # [B, T_env]
        self.debug_outputs["log_env_bt"]   = log_env_bt       # [B,1,1,T_cur]

        # ==== combine envelope & residual ====
        if self.env_combine_mode == "log":
            # LOG-DOMAIN ADDITIVE
            log_pred = log_env_bt + r_log
        else:
            # MAG-DOMAIN MULTIPLICATIVE
            mag_res = (r_log.exp() - 1e-3).clamp_min(0.0)
            mag_env = (log_env_bt.exp() - 1e-3).clamp_min(0.0)
            mag_pred = mag_env * mag_res
            log_pred = torch.log(mag_pred + 1e-3)

        return log_pred, log_env


    # --------------------------------------------------
    # Refactored forward
    # --------------------------------------------------
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
        
        self.debug_outputs = {} # reset additional outputs
        # ---- optional RAG 'input' fusion: build_h / project_aux ----
        aux = None
        if self.use_rag and (self.fusion == "input"):
            h = self.rag_gen.build_h(refs_feats, refs_mask)
            aux = self.rag_gen.project_aux(h) if h is not None else None

        B = mic_xyz.shape[0]
        T = t_idx.shape[1]

        # ---- global context c ----
        c = self._build_context(mic_xyz, src_xyz, head_dir, visual_feat)

        # ---- encoder: tokens from (c, t) ----
        w = self.encoder(c, t_idx, aux=aux)                  # [B, T, W]
        w = self.token_norm(w)

        # ---- RAG pre-fusion (film / concat) ----
        if self.use_rag and self.fusion in ("film", "concat"):
            w = self.rag_gen.pre_fuse(w, refs_logmag, refs_mask, refs_feats)

        # ---- decoder -> log residual r_log ----
        if self.encoder_kind == "neraf":
            r_log = self.decoder(w)
        else:
            dir_cond = self.rot_encoding(head_dir).unsqueeze(1).expand(B, T, -1).contiguous()
            r_log = self.decoder(w, dir_cond=dir_cond)

        # zero-mean residual over freq per time
        if self.normalize_residual and self.use_envelope:
            r_log = r_log - r_log.mean(dim=2, keepdim=True)  # [B,C,F,T]
        # expose residual for extra losses / debugging
        if self.use_envelope:
            self.debug_outputs["residual_log"] = r_log

        # ---- envelope fusion ----
        log_pred, log_env = self._apply_envelope(c, r_log, t_idx)

        # wandb logging
        if self._logger is not None and self.use_envelope:
            with torch.no_grad():
                log_dict = {
                    "envelope/res_log_mean": float(r_log.mean().item()),
                    "envelope/res_log_std": float(r_log.std().item()),
                    "envelope/log_env_mean": float(log_env.mean().item()),
                    "envelope/log_env_std": float(log_env.std().item()),
                }
                self._logger.log(log_dict)

        return log_pred