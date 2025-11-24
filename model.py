# model.py
# Unified NVAS wrapper with faithful NeRAF path and an AV-NeRF-style variant for RIR.
# - NeRAF mode: exactly as before (orientation SH in encoder; NeRAF-style heads).
# - AV-NeRF mode: orientation SH goes to the DECODER (FiLM on W), not the encoder.
# - Time index is normalized INSIDE the model (full: / (T-1); slice: / 59).
# - Visual features:
#     * NeRAF mode uses global 1024-D features (as before).
#     * AV-NeRF mode prefers per-pose features from dataset; still accepts 1024-D fallback.

from dataclasses import dataclass, field
from typing import Literal, Optional

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
    reverbrag: dict = field(default_factory=dict)
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
    NeRAF encoder; optionally concatenates an aux vector (size=W) before the trunk
    when fusion == 'input'.
    """

    def __init__(self, W: int, visual_dim: int = 1024, use_aux: bool = False):
        super().__init__()
        self.use_aux = use_aux
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")

        d_time = self.time_encoding.get_out_dim()
        d_pos = self.position_encoding.get_out_dim()
        d_rot = self.rot_encoding.get_out_dim()
        in_size = d_time + (2 * d_pos) + d_rot + visual_dim + (W if self.use_aux else 0)

        self.trunk = nn.Sequential(
            nn.Linear(in_size, 5096), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(5096, 2048), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(2048, 1024), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, W),    nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(
        self,
        mic_xyz: torch.Tensor,       # [B,3]
        src_xyz: torch.Tensor,       # [B,3]
        head_dir: torch.Tensor,      # [B,3]
        t_idx: torch.Tensor,         # [B,T,1] or [B,1,1]
        visual_feat: torch.Tensor,   # [B,1024]
        aabb: torch.Tensor,          # [2,3]
        aux: Optional[torch.Tensor] = None,  # [B,W] when use_aux=True
    ) -> torch.Tensor:
        B, T = t_idx.shape[0], t_idx.shape[1]

        # normalize time inside model
        if T > 1:
            t_norm = (t_idx.float() / float(T - 1)).clamp(0.0, 1.0)
        else:
            t_norm = (t_idx.float() / 59.0).clamp(0.0, 1.0)

        mic_n = SceneBox.get_normalized_positions(mic_xyz, aabb)
        src_n = SceneBox.get_normalized_positions(src_xyz, aabb)

        mic_e = self.position_encoding(mic_n)
        src_e = self.position_encoding(src_n)
        rot_e = self.rot_encoding(head_dir)
        vis_e = visual_feat

        static = torch.cat([mic_e, src_e, rot_e, vis_e], dim=-1)  # [B,S]
        if self.use_aux:
            if aux is None:
                aux = torch.zeros(B, vis_e.shape[-1], device=static.device, dtype=static.dtype)  # safe default
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
    AV-NeRF variant; optionally concatenates an aux vector (size=W) before the trunk
    when fusion == 'input'. (Orientation goes to decoder.)
    """

    def __init__(self, W: int, visual_dim_in: int = 1024, visual_dim_out: int = 128,
                 dropout_p: float = 0.0, use_aux: bool = False):
        super().__init__()
        self.use_aux = use_aux
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )

        self.av_mlp = nn.Sequential(
            nn.Linear(visual_dim_in, 512), nn.ReLU(inplace=True),
            nn.Linear(512, visual_dim_out), nn.ReLU(inplace=True),
            nn.Linear(visual_dim_out, visual_dim_out),
        )
        self.dropout_p = dropout_p

        d_time = self.time_encoding.get_out_dim()
        d_pos  = self.position_encoding.get_out_dim()
        in_size = d_time + (2 * d_pos) + visual_dim_out + (W if self.use_aux else 0)

        self.trunk = nn.Sequential(
            nn.Linear(in_size, 5096), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(5096, 2048), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(2048, 1024), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 1024), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, W),    nn.LeakyReLU(0.1, inplace=True),
        )

    @staticmethod
    def mydropout(tensor, p=0.5, training=True):
        if not training or p == 0:
            return tensor
        b = tensor.shape[0]
        mask = (torch.rand(b, device=tensor.device) > p).float().view(b, *([1] * (tensor.ndim - 1)))
        return tensor * mask

    def forward(
        self,
        mic_xyz: torch.Tensor,       # [B,3]
        src_xyz: torch.Tensor,       # [B,3]
        t_idx: torch.Tensor,         # [B,T,1]
        visual_feat: torch.Tensor,   # [B,Dv]
        aabb: torch.Tensor,          # [2,3]
        aux: Optional[torch.Tensor] = None,  # [B,W] when use_aux=True
    ) -> torch.Tensor:
        B, T = t_idx.shape[0], t_idx.shape[1]

        if T > 1:
            t_norm = (t_idx.float() / float(T - 1)).clamp(0.0, 1.0)
        else:
            t_norm = (t_idx.float() / 59.0).clamp(0.0, 1.0)

        mic_n = SceneBox.get_normalized_positions(mic_xyz, aabb)
        src_n = SceneBox.get_normalized_positions(src_xyz, aabb)

        mic_e = self.position_encoding(mic_n)
        src_e = self.position_encoding(src_n)

        if visual_feat.ndim == 1:
            visual_feat = visual_feat.unsqueeze(0)
        v = self.av_mlp(visual_feat)
        v = self.mydropout(v, p=self.dropout_p, training=self.training)

        static = torch.cat([mic_e, src_e, v], dim=-1)
        if self.use_aux:
            if aux is None:
                aux = torch.zeros(B, v.shape[-1], device=static.device, dtype=static.dtype)
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
    Per-channel Linear(W, N_freq) heads with tanh*10 (NeRAF).
    AV-NeRF-style conditioning: when dir_cond_dim>0 and dir_cond is provided,
    apply 'mydropout' to the orientation embedding and simply CONCAT it to the token
    before the heads (like AV-NeRF's feats, ori path). No FiLM, no gating.
    """

    def __init__(self, W: int, n_freq: int, n_channels: int, dir_cond_dim: int = 0, dropout_p: float = 0.0):
        super().__init__()
        self.n_channels = n_channels
        self.W = W
        self.dir_dim = dir_cond_dim
        self.use_dir = dir_cond_dim > 0
        self.p = dropout_p

        in_dim = W + dir_cond_dim if self.use_dir else W
        self.heads = nn.ModuleList([nn.Linear(in_dim, n_freq) for _ in range(n_channels)])

    @staticmethod
    def mydropout(tensor: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
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

        # ---- RAG ----
        rag_cfg = getattr(cfg, "reverbrag", {})
        self.fusion = str(rag_cfg.get("fusion", "film")).lower()

        # Encoders
        if cfg.baseline == "neraf":
            self.encoder_kind = "neraf"
            self.encoder = NeRAFEncoder(W=cfg.W_field, visual_dim=1024, use_aux=(self.fusion == "input"))
            dir_cond_dim = 0
        elif cfg.baseline == "avnerf":
            self.encoder_kind = "avnerf"
            self.encoder = AVNeRFEncoder(W=cfg.W_field, visual_dim_in=1024, visual_dim_out=1024,
                                         dropout_p=0.1, use_aux=(self.fusion == "input"))
            self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")
            dir_cond_dim = self.rot_encoding.get_out_dim()
        else:
            raise ValueError(f"Unknown baseline {cfg.baseline}")

        self.decoder = NeRAFDecoder(W=cfg.W_field, n_freq=self.N_freq,
                                    n_channels=self.n_channels, dir_cond_dim=dir_cond_dim)

        self.use_rag = rag_cfg.get("enabled",False)
        if self.use_rag:
            self.rag_gen = ReverbRAGGenerator(cfg=rag_cfg, W=cfg.W_field)
        else:
            self.rag_gen = None
        self.token_norm = nn.LayerNorm(cfg.W_field)

    def set_logger(self, logger):
        self._logger = logger
        if hasattr(self.rag_gen, "set_logger"):
            self.rag_gen.set_logger(logger)

    # def forward(
    #     self,
    #     mic_xyz: torch.Tensor,
    #     src_xyz: torch.Tensor,
    #     head_dir: torch.Tensor,
    #     t_idx: torch.Tensor,
    #     visual_feat: torch.Tensor,
    #     refs_logmag: torch.Tensor = None,   # [B,K,1,F,60]
    #     refs_mask: torch.Tensor = None,     # [B,K]
    #     refs_feats: torch.Tensor = None,    # [B,K,32,4]
    # ) -> torch.Tensor:

    #     # --- input mode: build h BEFORE encoder and inject as aux ---
    #     aux = None
    #     if self.use_rag and (self.fusion == "input"):
    #         h = self.rag_gen.build_h(refs_feats, refs_mask)           # [B,h_dim] or None
    #         aux = self.rag_gen.project_aux(h)                         # [B,W] or None

    #     if self.encoder_kind == "neraf":
    #         w = self.encoder(mic_xyz, src_xyz, head_dir, t_idx, visual_feat, self.aabb, aux=aux)  # [B,T,W]
    #         if self.use_rag and self.fusion in ("film", "concat"):
    #             w = self.rag_gen.pre_fuse(w, refs_logmag, refs_mask, refs_feats)
    #         return self.decoder(w)

    #     # AV-NeRF path: orientation to decoder
    #     w = self.encoder(mic_xyz, src_xyz, t_idx, visual_feat, self.aabb, aux=aux)
    #     B, T = t_idx.shape[0], t_idx.shape[1]
    #     rot_e = self.rot_encoding(head_dir).unsqueeze(1).expand(B, T, -1).contiguous()

    #     if self.use_rag and self.fusion in ("film", "concat"):
    #         w = self.rag_gen.pre_fuse(w, refs_logmag, refs_mask, refs_feats)

    #     return self.decoder(w, dir_cond=rot_e)
    
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

        def _check(name, t):
            if t is None:
                return
            if not torch.isfinite(t).all():
                print(f"[NaN BUG] non-finite {name}")
                print(f"  {name} min/max:", t.min().item(), t.max().item())
                raise RuntimeError(f"NaN in {name}")

        # ---- inputs ----
        _check("mic_xyz", mic_xyz)
        _check("src_xyz", src_xyz)
        _check("head_dir", head_dir)
        _check("t_idx", t_idx)
        _check("visual_feat", visual_feat)
        _check("refs_logmag", refs_logmag)
        _check("refs_mask", refs_mask.float() if refs_mask is not None else None)
        _check("refs_feats", refs_feats)

        # ---- optional RAG 'input' fusion: build_h / project_aux ----
        aux = None
        if self.use_rag and (self.fusion == "input"):
            h = self.rag_gen.build_h(refs_feats, refs_mask)
            _check("rag_build_h", h)

            aux = self.rag_gen.project_aux(h) if h is not None else None
            _check("rag_project_aux", aux)

        # ---- encoder ----
        if self.encoder_kind == "neraf":
            w = self.encoder(mic_xyz, src_xyz, head_dir, t_idx, visual_feat, self.aabb, aux=aux)
        else:
            w = self.encoder(mic_xyz, src_xyz, t_idx, visual_feat, self.aabb, aux=aux)
            
        w = self.token_norm(w)
        _check("encoder_out", w)

        # ---- RAG fusion (film / concat) ----
        if self.use_rag and self.fusion in ("film", "concat"):
            w = self.rag_gen.pre_fuse(w, refs_logmag, refs_mask, refs_feats)
            _check("rag_pre_fuse", w)

        # ---- decoder ----
        if self.encoder_kind == "neraf":
            out = self.decoder(w)
        else:
            B, T = t_idx.shape[0], t_idx.shape[1]
            rot_e = self.rot_encoding(head_dir).unsqueeze(1).expand(B, T, -1).contiguous()
            _check("rot_e", rot_e)
            out = self.decoder(w, dir_cond=rot_e)

        _check("decoder_out", out)
        return out
