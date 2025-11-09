# model.py
# Unified NVAS wrapper with faithful NeRAF path and an AV-NeRF-style variant for RIR.
# - NeRAF mode: exactly as before (orientation SH in encoder; NeRAF-style heads).
# - AV-NeRF mode: orientation SH goes to the DECODER (FiLM on W), not the encoder.
# - Time index is normalized INSIDE the model (full: / (T-1); slice: / 59).
# - Visual features:
#     * NeRAF mode uses global 1024-D features (as before).
#     * AV-NeRF mode prefers per-pose features from dataset; still accepts 1024-D fallback.

from dataclasses import dataclass
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
    sample_rate: int = 48000  # 48k -> 513 bins; 16k -> 257 bins
    W_field: int = 1024
    scene_aabb: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0],
                                             [1.0, 1.0, 1.0]], dtype=torch.float32)


def fs_to_stft_params(fs: int):
    if fs == 48000:
        return dict(N_freq=513, hop_len=256, win_len=512)
    elif fs == 16000:
        return dict(N_freq=257, hop_len=128, win_len=256)
    # default to 48k spec
    return dict(N_freq=513, hop_len=256, win_len=512)


# --------------------------------
# NeRAF-style Encoder (unchanged)
# --------------------------------
class NeRAFEncoder(nn.Module):
    """
    Keep the NeRAF encoder stack intact:
      - NeRFEncoding for time (1D)
      - NeRFEncoding for 3D positions (mic, src)
      - SHEncoding(levels=4) for orientation
    The trunk maps concatenated encodings + visual(1024) → W token per time step.
    """

    def __init__(self, W: int, visual_dim: int = 1024):
        super().__init__()
        # Exact encoding hyperparams
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")

        # Trunk dimensions
        d_time = self.time_encoding.get_out_dim()
        d_pos = self.position_encoding.get_out_dim()
        d_rot = self.rot_encoding.get_out_dim()
        in_size = d_time + (2 * d_pos) + d_rot + visual_dim

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
        head_dir: torch.Tensor,      # [B,3]  (SH)
        t_idx: torch.Tensor,         # [B,T,1] raw indices 0..T-1 or [B,1,1] for slice
        visual_feat: torch.Tensor,   # [B,1024]
        aabb: torch.Tensor,          # [2,3]
    ) -> torch.Tensor:
        """
        Returns:
            w: [B, T, W]
        """
        B, T = t_idx.shape[0], t_idx.shape[1]

        # normalize time inside model
        if T > 1:
            t_norm = (t_idx.float() / float(T - 1)).clamp(0.0, 1.0)
        else:
            t_norm = (t_idx.float() / 59.0).clamp(0.0, 1.0)  # slice mode

        mic_n = SceneBox.get_normalized_positions(mic_xyz, aabb)
        src_n = SceneBox.get_normalized_positions(src_xyz, aabb)

        mic_e = self.position_encoding(mic_n)   # [B, d_pos]
        src_e = self.position_encoding(src_n)   # [B, d_pos]
        rot_e = self.rot_encoding(head_dir)     # [B, d_rot]
        vis_e = visual_feat                     # [B, 1024]

        static = torch.cat([mic_e, src_e, rot_e, vis_e], dim=-1)  # [B, S]

        # Time-dependent path (per-frame)
        t_e = self.time_encoding(t_norm.reshape(B * T, 1))  # [B*T, d_time]
        t_e = t_e.view(B, T, -1)

        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])  # [B,T,S]
        enc = torch.cat([t_e, static_exp], dim=-1)                       # [B,T,in_size]

        enc_flat = enc.reshape(B * T, -1)
        w_flat = self.trunk(enc_flat)                # [B*T, W]
        w = w_flat.view(B, T, -1)                    # [B,T,W]
        return w


# --------------------------------------------------
# AV-NeRF-style Encoder for RIR (no orientation here)
# --------------------------------------------------
class AVNeRFEncoder(nn.Module):
    """
    AV-NeRF variant: orientation is NOT consumed in the encoder (to match AV-NeRF),
    but we still keep the same time & position encodings, and we allow per-pose visual features.
    """

    def __init__(self, W: int, visual_dim_in: int = 1024, visual_dim_out: int = 128, dropout_p: float = 0.0):
        super().__init__()
        # same time/position encodings as NeRAF
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        # NOTE: no SH orientation here

        # small AV MLP for per-pose visual features (if present) + mydropout
        
        self.av_mlp = nn.Sequential(nn.Linear(visual_dim_in, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, visual_dim_out),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(visual_dim_out, visual_dim_out))
        
        self.dropout_p = dropout_p

        d_time = self.time_encoding.get_out_dim()
        d_pos = self.position_encoding.get_out_dim()
        in_size = d_time + (2 * d_pos) + visual_dim_out  # mic+src+time+visual

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
        else:
            batch_size = tensor.shape[0]
            random_tensor = torch.rand(batch_size, device=tensor.device)
            new_tensor = [torch.zeros_like(tensor[i]) if random_tensor[i] <= p else tensor[i] for i in range(batch_size)]
            new_tensor = torch.stack(new_tensor, dim=0) # [B, ...]
            return new_tensor

    def forward(
        self,
        mic_xyz: torch.Tensor,       # [B,3]
        src_xyz: torch.Tensor,       # [B,3]
        t_idx: torch.Tensor,         # [B,T,1]
        visual_feat: torch.Tensor,   # [B,Dv] (per-pose preferred; fallback 1024)
        aabb: torch.Tensor,          # [2,3]
    ) -> torch.Tensor:
        """
        Returns:
            w: [B, T, W]
        """
        B, T = t_idx.shape[0], t_idx.shape[1]

        # normalize time inside model
        if T > 1:
            t_norm = (t_idx.float() / float(T - 1)).clamp(0.0, 1.0)
        else:
            t_norm = (t_idx.float() / 59.0).clamp(0.0, 1.0)

        mic_n = SceneBox.get_normalized_positions(mic_xyz, aabb)
        src_n = SceneBox.get_normalized_positions(src_xyz, aabb)

        mic_e = self.position_encoding(mic_n)   # [B, d_pos]
        src_e = self.position_encoding(src_n)   # [B, d_pos]

        # per-pose visual pathway
        if visual_feat.ndim == 1:
            visual_feat = visual_feat.unsqueeze(0)
        v_feats = self.av_mlp(visual_feat)              # [B, visual_dim_out]
        v_feats = self.mydropout(v_feats, p=self.dropout_p, training=self.training)

        static = torch.cat([mic_e, src_e, v_feats], dim=-1)  # [B, S]

        t_e = self.time_encoding(t_norm.reshape(B * T, 1))  # [B*T, d_time]
        t_e = t_e.view(B, T, -1)

        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])  # [B,T,S]
        enc = torch.cat([t_e, static_exp], dim=-1)                       # [B,T,in_size]

        enc_flat = enc.reshape(B * T, -1)
        w_flat = self.trunk(enc_flat)                # [B*T, W]
        w = w_flat.view(B, T, -1)                    # [B,T,W]
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
    """
    Inputs:
      mic_xyz[B,3], src_xyz[B,3], head_dir[B,3], t_idx[B,T,1], visual_feat[B,Dv]
    Output: stft_pred[B, C, F, T]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        aabb = getattr(cfg, "scene_aabb", None)
        aabb = aabb.clone().detach().float() if aabb is not None else torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32
        )
        self.register_buffer("aabb", aabb, persistent=False)

        stft = fs_to_stft_params(cfg.sample_rate)
        self.N_freq = stft["N_freq"]
        self.n_channels = 1 if cfg.database == "raf" else 2

        # Encoders
        if cfg.baseline == "neraf":
            self.encoder_kind = "neraf"
            self.encoder = NeRAFEncoder(W=cfg.W_field, visual_dim=1024)
            dir_cond_dim = 0  # no decoder conditioning; NeRAF uses orientation in encoder
        elif cfg.baseline == "avnerf":
            self.encoder_kind = "avnerf"
            # allow arbitrary per-pose visual dims; we'll accept whatever comes and linearize
            self.encoder = AVNeRFEncoder(W=cfg.W_field, visual_dim_in=1024, visual_dim_out=1024, dropout_p=0.1)
            # decoder will accept SH(3D) as FiLM
            self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")
            dir_cond_dim = self.rot_encoding.get_out_dim()
        else:
            raise ValueError(f"Unknown baseline {cfg.baseline}")

        # Decoder (same heads in both modes; AV-NeRF may use FiLM on W)
        self.decoder = NeRAFDecoder(W=cfg.W_field, n_freq=self.N_freq,
                                    n_channels=self.n_channels, dir_cond_dim=dir_cond_dim)
        
        # ---- ReverbRAG (lightweight placeholder for now) ----
        self.use_rag = True
        rag_cfg = getattr(cfg, "reverbrag", {}) if hasattr(cfg, "reverbrag") else {}
        self.rag_gen = ReverbRAGGenerator(
            n_freq=self.N_freq, W=cfg.W_field, mode="film_fuse", rag_cfg=rag_cfg
        )

    def forward(
        self,
        mic_xyz: torch.Tensor,
        src_xyz: torch.Tensor,
        head_dir: torch.Tensor,
        t_idx: torch.Tensor,
        visual_feat: torch.Tensor,
        refs_logmag: torch.Tensor = None,     # [B,K,1,F,60] log-mag
        refs_mask: torch.Tensor = None,       # [B,K] (bool)
        refs_feats: torch.Tensor = None,          # [B,K,BANDS,4] decay features (not used yet)
    ) -> torch.Tensor:
        if self.encoder_kind == "neraf":
            w = self.encoder(mic_xyz, src_xyz, head_dir, t_idx, visual_feat, self.aabb)  # [B,T,W]
            # (for now) just let generator optionally pre-process w / refs and return a (maybe) modified w
            w = self.rag_gen.pre_fuse(w, refs_logmag, refs_mask, refs_feats) if self.use_rag else w
            return self.decoder(w)

        # AV-NeRF path
        w = self.encoder(mic_xyz, src_xyz, t_idx, visual_feat, self.aabb)
        B, T = t_idx.shape[0], t_idx.shape[1]
        rot_e = self.rot_encoding(head_dir).unsqueeze(1).expand(B, T, -1).contiguous()
        # same pre-fuse hook
        w = self.rag_gen.pre_fuse(w, refs_logmag, refs_mask, refs_feats) if self.use_rag else w
        return self.decoder(w, dir_cond=rot_e)