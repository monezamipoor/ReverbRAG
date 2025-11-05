# model.py
# Unified ReverbRAG NVAS model with NeRAF-identical encoder/decoder blocks
# - Encoders: NeRFEncoding (time, position), SHEncoding (heading)
# - Trunk: [in_size -> 5096 -> 2048 -> 1024 -> 1024 -> W] with LeakyReLU(0.1)
# - Heads: per-channel Linear(W, N_freq) with tanh*10 (NeRAF style)
# - Visual features (global.pt, 1024-D) are ALWAYS used (for both neraf and avnerf)

from dataclasses import dataclass
from typing import Literal, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding  # exact libs used in your NeRAF code


# -----------------------
# Configs (kept minimal)
# -----------------------
@dataclass
class ModelConfig:
    baseline: Literal["neraf", "avnerf"] = "neraf"
    database: Literal["raf", "soundspaces"] = "raf"
    scene_root: str = "../NeRAF/data/RAF"  # replaced by YAML at runtime
    scene_name: str = "FurnishedRoom"
    sample_rate: int = 48000  # 48000 -> 513, 16000 -> 257
    W_field: int = 1024       # width after trunk (same symbol W as in your NeRAF code)


def fs_to_stft_params(fs: int) -> Dict[str, int]:
    # Pulled from your NeRAF model:
    if fs == 48000:
        return dict(N_freq=513, hop_len=256, win_len=512)
    elif fs == 16000:
        return dict(N_freq=257, hop_len=128, win_len=256)
    else:
        # Sensible default mirroring 48k
        return dict(N_freq=513, hop_len=256, win_len=512)


# --------------------------------
# NeRAF-style Encoder (both modes)
# --------------------------------
class NeRAFEncoder(nn.Module):
    """
    Builds the exact encoding stack used by your NeRAF:
    - NeRFEncoding(time, 1-D)     : num_frequencies=10, [0..8], include_input=True
    - NeRFEncoding(position, 3-D) : same as above
    - SHEncoding(levels=4, tcnn)  : heading/rotation
    Concatenates: enc(time) + enc(mic) + enc(src) + sh(head_dir) + visual(1024)
    Then pushes through NeRAF trunk to produce W-dimensional token.
    """

    def __init__(self, W: int, visual_dim: int = 1024):
        super().__init__()
        # Exact encoding hyperparams from your NeRAF code
        self.time_encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.rot_encoding = SHEncoding(levels=4, implementation="tcnn")

        self._d_time = self.time_encoding.get_out_dim()
        self._d_pos = self.position_encoding.get_out_dim()
        self._d_rot = self.rot_encoding.get_out_dim()
        in_size = self._d_time + (2 * self._d_pos) + self._d_rot + visual_dim

        # NeRAF trunk (widths exactly as in your field):
        layers = []
        dims = [in_size, 5096, 2048, 1024, 1024, W]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.trunk = nn.Sequential(*layers)

    def forward(
        self,
        mic_xyz: torch.Tensor,    # [B,3]
        src_xyz: torch.Tensor,    # [B,3]
        head_dir: torch.Tensor,   # [B,3] direction vector or rotation representation
        t_norm: torch.Tensor,     # [B,T,1] in [0,1]
        visual_feat: torch.Tensor # [B,1024]
    ) -> torch.Tensor:
        """
        Returns:
            W-token per time step: [B, T, W]
        """
        B, T, _ = t_norm.shape

        # Encode static terms once per batch element
        mic_e = self.position_encoding(mic_xyz)      # [B, d_pos]
        src_e = self.position_encoding(src_xyz)      # [B, d_pos]
        rot_e = self.rot_encoding(head_dir)          # [B, d_rot]
        # Visual features are used as-is
        vis_e = visual_feat                           # [B, 1024]

        static = torch.cat([mic_e, src_e, rot_e, vis_e], dim=-1)  # [B, 2*d_pos + d_rot + 1024]

        # Time-dependent path
        t_e = self.time_encoding(t_norm.reshape(B * T, 1))           # [B*T, d_time]
        t_e = t_e.view(B, T, -1)                                  # [B, T, d_time]

        # Broadcast static to time dimension and concat
        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])   # [B, T, S]
        enc = torch.cat([t_e, static_exp], dim=-1)                        # [B, T, in_size]

        # Trunk is feed-forward per time step → flatten, run, then reshape
        enc_flat = enc.reshape(B * T, -1)           # [B*T, in_size]
        w_flat = self.trunk(enc_flat)               # [B*T, W]
        w = w_flat.view(B, T, -1)                   # [B, T, W]
        return w


# For now AVNeRF uses the exact same encoder form (visual features are mandatory anyway).
class AVNeRFEncoder(NeRAFEncoder):
    pass


# ----------------------------------------
# NeRAF-style Decoder (heads + tanh * 10)
# ----------------------------------------
class NeRAFDecoder(nn.Module):
    """
    Per-channel STFT heads:
      - Linear(W, N_freq) and apply tanh * 10 (matching your code)
    """

    def __init__(self, W: int, n_freq: int, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.heads = nn.ModuleList([nn.Linear(W, n_freq) for _ in range(n_channels)])

    def forward(self, w_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_tokens: [B, T, W]
        Returns:
            stft: [B, C, F, T] with tanh*10 applied
        """
        B, T, W = w_tokens.shape
        outs = []
        # Process each channel with its own linear head, per time step
        w_flat = w_tokens.reshape(B * T, W)  # [B*T, W]
        for head in self.heads:
            f_flat = head(w_flat)            # [B*T, F]
            f_flat = torch.tanh(f_flat) * 10
            f = f_flat.view(B, T, -1).transpose(1, 2)  # [B, F, T]
            outs.append(f.unsqueeze(1))      # [B,1,F,T]
        return torch.cat(outs, dim=1)        # [B, C, F, T]


# ----------------------------
# Unified ReverbRAG NVAS Model
# ----------------------------
class UnifiedReverbRAGModel(nn.Module):
    """
    Minimal, NeRAF-faithful forward:
       inputs: mic_xyz [B,3], src_xyz [B,3], head_dir [B,3], t_norm [B,T,1], visual_feat [B,1024]
       outputs: stft_pred [B, C, F, T]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Channels by database
        if cfg.database.lower() == "raf":
            n_channels = 1
        elif cfg.database.lower() == "soundspaces":
            n_channels = 2
        else:
            raise ValueError(f"Unknown database {cfg.database}")

        stft = fs_to_stft_params(cfg.sample_rate)
        self.N_freq = stft["N_freq"]

        # Encoder choice (both require visual features)
        if cfg.baseline == "neraf":
            self.encoder = NeRAFEncoder(W=cfg.W_field, visual_dim=1024)
        elif cfg.baseline == "avnerf":
            self.encoder = AVNeRFEncoder(W=cfg.W_field, visual_dim=1024)
        else:
            raise ValueError(f"Unknown baseline {cfg.baseline}")

        # NeRAF-style decoder heads
        self.decoder = NeRAFDecoder(W=cfg.W_field, n_freq=self.N_freq, n_channels=n_channels)

    def forward(
        self,
        mic_xyz: torch.Tensor,    # [B,3]
        src_xyz: torch.Tensor,    # [B,3]
        head_dir: torch.Tensor,   # [B,3]
        t_norm: torch.Tensor,     # [B,T,1]
        visual_feat: torch.Tensor # [B,1024]
    ) -> torch.Tensor:
        w = self.encoder(mic_xyz, src_xyz, head_dir, t_norm, visual_feat)  # [B, T, W]
        stft = self.decoder(w)                                             # [B, C, F, T]
        return stft