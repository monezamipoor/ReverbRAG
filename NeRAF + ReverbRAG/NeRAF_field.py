"""
NeRAF neural acoustic field (NAcF)

"""

# safest for 3.8:
# try:
#     from typing import Literal, Optional
# except ImportError:
#     from typing_extensions import Literal  # if your typing is older
#     from typing import Optional
from dataclasses import dataclass
from typing import Literal, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field
from .NeRAF_helper import build_band2freq_matrix, SpectralEnvelopePreservationLoss, BandQueryMaskHead

from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)

class NeRAFVisionFieldValue(nn.Module): # Should be deleted -> NOT TESTED

    def __init__(self,module):
        super().__init__()
        self.module = module

    def forward(self,ray_samples, compute_normals=False):
        return self.module(ray_samples, compute_normals=compute_normals)




HeadMode = Literal["default", "query", "stft", "refine"]
TemporalScope = Literal["global", "window", "both"]

@dataclass
class FieldHeadConfig:
    mode: HeadMode = "stft" # "stft", "query", "default", "refine"
    self_attention_time: bool = True
    gate_type: Literal["vector", "mlp"] = "vector"
    stft_token_dim: int = 256
    time_num_heads: int = 1
    num_heads: int = 1
    temporal_scope: TemporalScope = "window"   # "global" | "window" | "both"
    window_size: int = 9                       # must be odd
    fusion: Literal["attn", "bahdanau"] = "bahdanau"
    attn_dim: int = 128,
    gating: Literal["single", "film"] = "film"
    mlp_film: bool = True
    respect_no_ref_mask: bool = False
    masked_grad_mode: Literal["none", "ghost"] = "none"
    masked_grad_scale: float = 1.0
    use_ref_id_tokens: bool = True
    film_strength: float = 0.1
    learnable_film_strength: bool = False
    use_lora: bool = False
    # --- refine mode knobs ---
    mask_num_bands: int = 32
    mask_predictor: Literal["mlp", "film", "bandq"] = "mlp"       # mask FiLM is separate from W-space film
    mask_upsample: Literal["interp", "erb", "mel"] = "erb"
    use_conv: bool = True
    mask_gamma_init: float = 0.05                        # small gate on residual path
    put_mask_l1_in_outputs: bool = True                    # surface L1 (post-tanh, pre-γ)
    spectral_loss_mask: bool = True
    stft_global_stats_path: Optional[str] = "../data/RAF/FurnishedRoom/stft_band_norm_train.pt"

class NeRAFAudioSoundField(nn.Module):
    """
    Trunk produces query features of width W.
    STFT branch turns (B,K,F,T) into a single token g_t in R^W using:
      - F->W projection
      - + time PE
      - windowed self-attn over time returning the center query
      - additive (Bahdanau) mixing across refs conditioned on the trunk
    Fusion with the trunk is modular (FiLM or query-key attention) + gating.
    """
    def __init__(
        self,
        in_size: int,                 # trunk input dim
        W: int,                       # model width
        sound_rez: int = 2,
        N_frequencies: int = 257,
        head_cfg: Optional[FieldHeadConfig] = None,
        num_ref_embeddings: Optional[int] = None,   # <= NEW (size of ref bank for learned IDs)
    ):
        super().__init__()
        self.cfg = head_cfg or FieldHeadConfig()
        assert self.cfg.window_size % 2 == 1, "window_size must be odd"
        self.W = W
        self.N_f = N_frequencies
        self.mode = self.cfg.mode
        # ----- Trunk MLP -> W -----
        self.trunk = nn.Sequential(
            nn.Linear(in_size, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, W),
        )

        # ----- Output heads (example: STFT magnitude per mic) -----
        self.STFT_linear = nn.ModuleList([nn.Linear(W, N_frequencies) for _ in range(sound_rez)])

        # ----- STFT encoder: F -> W and time mixing -----
        self.stft_td = max(32, min(self.cfg.stft_token_dim, self.N_f))  # bound
        input_size = self.stft_td if self.mode != "query" else self.W
        self.stft_freq_proj = nn.Linear(self.N_f, self.stft_td)               # (F -> W)
        if self.cfg.temporal_scope == "window":
            self.time_self_attn = nn.MultiheadAttention(
                embed_dim=self.stft_td, num_heads=self.cfg.time_num_heads, batch_first=True
            )
        else:
            # Global time self-attention stack
            self.time_encoder_global = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=self.stft_td,               # == stft_token_dim
                    nhead=self.cfg.time_num_heads,
                    dim_feedforward=4 * self.stft_td,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=1,  # start with 1; you can increase if needed
            )

            # Learnable CLS token for time pooling
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.stft_td))
            torch.nn.init.normal_(self.cls_token, std=0.02)

        
        # ----- Query attention -----
        if self.cfg.fusion == "attn":
            self.trunk_proj = nn.Linear(self.W, self.cfg.attn_dim)
            self.ref_proj = nn.Linear(self.W if self.mode == "query" else self.stft_td, self.cfg.attn_dim)
            self.attn_to_stft = nn.Identity() if self.stft_td == self.cfg.attn_dim else nn.Linear(self.cfg.attn_dim, self.stft_td)
            self.attn_to_W= nn.Identity() if self.W == self.cfg.attn_dim else nn.Linear(self.cfg.attn_dim, self.W)
            self.ref_cross_attn = nn.MultiheadAttention(
                embed_dim=self.cfg.attn_dim, num_heads=self.cfg.num_heads, batch_first=True
            ) 
        else:
            self.trunk_proj = None
            self.ref_proj = None
            self.ref_cross_attn = None
            if self.mode != "query":
                self.stft_to_W= nn.Identity() if self.W == self.stft_td else nn.Linear(self.stft_td, self.W)
            else: self.stft_to_W = None
            
        # widths already known when you build the field
        self.ln_trunk_for_film = nn.LayerNorm(self.W)
        self.ln_token_for_film = nn.LayerNorm(self.W)

        # ----- Optional learned ref ID embeddings -----
        self.ref_id_embed = None
        if (self.cfg.use_ref_id_tokens
            and num_ref_embeddings is not None
            and num_ref_embeddings > 0):
            self.ref_id_embed = nn.Embedding(num_ref_embeddings, input_size)
            self.ref_id_dropout = nn.Dropout(p=0.1)


        # ----- Additive (Bahdanau) scorer for ref mixing -----
        hidden = max(64, self.stft_td)
        self.ref_score_Wv = nn.Linear(input_size, hidden, bias=False)
        self.ref_score_Wq = nn.Linear(W, hidden, bias=True)
        self.ref_score_a  = nn.Linear(hidden, 1, bias=False)

        # ----- Fusion (FiLM or Attn) + gate -----
        self.attn = nn.MultiheadAttention(embed_dim=W, num_heads=self.cfg.num_heads, batch_first=True)
        # token_size = self.W if self.mode == "query" else self.stft_td
        input_size = self.W *2
        film_norm_trunk: bool = getattr(self.cfg, "film_norm_trunk", False) # optional LN on trunk

        self.film_norm_trunk = film_norm_trunk
        if self.film_norm_trunk:
            self.trunk_ln = nn.LayerNorm(self.W)
   
            
        input_size = (2 * self.W + self.N_f) if self.mode == "refine" else input_size # [q_W || g_W || mask_bands]
        film_out = self.N_f * 2 if self.mode == "refine" else self.W * 2
        if self.cfg.gating == "film":
            if self.cfg.mlp_film:
                # 2W -> W/2 -> 2W (GELU), lightweight but expressive
                self.film_predictor = nn.Sequential(
                    nn.Linear(input_size, self.W // 2),
                    nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(self.W // 2, film_out)
                )
            else:
                # original single-layer FiLM
                self.film_predictor = nn.Linear(input_size, film_out)
        else:
            self.film_predictor = None

        # learnable FiLM strength (instead of hardcoded 0.1)
        if self.cfg.learnable_film_strength:
            self.film_gamma_scale = nn.Parameter(torch.tensor(float(self.cfg.film_strength)))
            self.film_beta_scale  = nn.Parameter(torch.tensor(float(self.cfg.film_strength)))
        else:
            self.film_gamma_scale = None
            self.film_beta_scale = None

        # GATE: vector or MLP, and which inputs it uses
        self.gate_mlp = None
        self.gate_w = None
        self.gate_b = None

        
        in_dim =  self.W * 2 if self.cfg.gating == "single" else (self.W * 2) + 1
        if self.cfg.gate_type == "mlp":
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim, max(64, in_dim // 2)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(max(64, in_dim // 2), 1),
            )
            nn.init.constant_(self.gate_mlp[2].bias, -2.0)
        else:  # "vector"
            self.gate_w = nn.Parameter(torch.zeros(in_dim, 1))
            nn.init.normal_(self.gate_w, mean=0.0, std=0.02)
            self.gate_b = nn.Parameter(torch.full((1,), -2.0))

        # LoRA config (tweak as you like)
        self.d_attn = self.cfg.attn_dim if self.cfg.fusion == "attn" else hidden       # input dim (p is 64)
        self.lora_rank = 16   # r
        self.lora_alpha = 1.0  # scale (alpha/r)
        self.lora_dropout_p = 0.1

        # Layers
        self.lora_up   = nn.Linear(self.d_attn, self.lora_rank, bias=False)   # 64 -> r
        self.lora_down = nn.Linear(self.lora_rank, self.W, bias=False)           # r  -> 512
        self.lora_dropout = nn.Dropout(self.lora_dropout_p)

        # Identity-safe init: start with zero delta
        nn.init.zeros_(self.lora_down.weight)
        # Small/standard init for the "up" leg
        nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))

        # LoRA scaling (like alpha/r)
        self.lora_scale = self.lora_alpha / max(1, self.lora_rank)    
        
        # ---------- refine modules ----------  
        if self.mode == "refine":   
            BANDS = self.cfg.mask_num_bands
            if self.cfg.mask_upsample != "interp":
                # after you compute W as a torch.Tensor of shape (Fbins, num_bands)
                self.W_psychoacoustic = build_band2freq_matrix(Fbins=self.N_f, n_fft=1024, sr=48000, n_bands=BANDS, scale=self.cfg.mask_upsample)  # (513, BANDS)
            # tiny global gate on residual magnitude
            self.refine_mask_gamma = nn.Parameter(
                torch.tensor(self.cfg.mask_gamma_init, dtype=torch.float32)
            )

            # 32-band mask predictor: either MLP(q||g) or FiLM(g ▷ q) -> bands
            if self.cfg.mask_predictor == "mlp":
                self.refine_mask_mlp = nn.Sequential(
                    nn.Linear(2 * self.W, min(256, W)),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(min(256, W), BANDS),
                )
                self.refine_mask_film_predictor = None
                self.refine_mask_out = None
            elif self.cfg.mask_predictor == "film":
                # keep in mind g is typically size stft_token_dim (e.g., 256)
                self.refine_mask_film_predictor = nn.Linear(self.W * 2, 2 * self.W)
                self.refine_mask_out = nn.Linear(self.W * 2, BANDS)
                self.refine_mask_mlp = None
            else:
                self.refine_mask_head = BandQueryMaskHead(
                    W=self.W,
                    bands=self.cfg.mask_num_bands,   # 32 by your config
                    use_g=True,                      # uses both q_W and g_W
                    conv_kernel=None               # tiny Conv1D across bands
                )



            # learned upsampling variant: we do interp -> Conv1d smoothing (stride-free, stable)
            if self.cfg.use_conv:
                self.refine_upconv = nn.Conv1d(
                    1, 1, kernel_size=3, padding=1, padding_mode="replicate", bias=True
                )
                with torch.no_grad():
                    self.refine_upconv.weight.zero_()
                    self.refine_upconv.weight[0, 0, 1] = 1.0  # center tap = 1  -> identity
                    self.refine_upconv.bias.zero_()
                self.refine_scale = nn.Parameter(torch.tensor(0.0))  # starts as exact no-op
            else: self.refine_upconv = None

            # ---- α gate over STFT combine, driven by gate_type ----
            # alpha_in = 2 * self.W  # [q_W || g_W]
            # if self.cfg.gate_type == "vector":
            #     self.refine_alpha_vec = nn.Parameter(torch.zeros(alpha_in))
            #     self.refine_alpha_bias = nn.Parameter(torch.tensor(-2.0))
            #     self.refine_gate_mlp = None
            # else:
            #     self.refine_gate_mlp = nn.Sequential(
            #         nn.Linear(alpha_in, max(64, alpha_in // 2)),
            #         nn.LeakyReLU(0.1, inplace=True),
            #         nn.Linear(max(64, alpha_in // 2), 1),
            #     )
            #     nn.init.constant_(self.refine_gate_mlp[-1].bias, -2.0)
            #     self.refine_alpha_vec = None
            #     self.refine_alpha_bias = None
            
            if self.cfg.spectral_loss_mask:
                self.env_loss_fn = SpectralEnvelopePreservationLoss(Fbins=self.N_f)

            # inside NeRAFAudioSoundField.__init__(...)
            self.register_buffer("stft_global_mu", None, persistent=False)
            self.register_buffer("stft_global_sigma", None, persistent=False)

            stats_path = getattr(self.cfg, "stft_global_stats_path", None)
            if stats_path:
                try:
                    stats = torch.load(stats_path, map_location="cpu")
                    # allow either a raw tensor or a dict wrapper
                    if isinstance(stats, dict) and "stats" in stats:
                        stats = stats["stats"]
                    stats = torch.as_tensor(stats, dtype=torch.float32)  # (F0, 2)
                    if stats.ndim == 2 and stats.size(-1) == 2:
                        mu = stats[:, 0]                                  # (F0,)
                        sigma = stats[:, 1].clamp_min(1e-6)               # (F0,)
                        # register as buffers so they .to(device) with the module
                        self.register_buffer("stft_global_mu", mu, persistent=True)
                        self.register_buffer("stft_global_sigma", sigma, persistent=True)
                    else:
                        print(f"[refine] WARN: stats at {stats_path} not shaped (F,2); got {tuple(stats.shape)}")
                except Exception as e:
                    print(f"[refine] WARN: failed to load global STFT stats from {stats_path}: {e}")


    # --------------------------- small helpers ---------------------------

    def _make_time_pe(self, T: int, device) -> torch.Tensor:
        """Sinusoidal PE in R^W over T positions."""
        d = self.stft_td
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
        i = torch.arange(d // 2, device=device, dtype=torch.float32)            # [d/2]
        div = torch.exp(-math.log(10000.0) * (2 * i) / d)
        pe = torch.zeros(T, d, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # [T,W]
    
    # ---------- α gate from [q_W || g_W] ----------
    # def _alpha_from_ctx(self, q_W: torch.Tensor, g_stft: torch.Tensor) -> torch.Tensor:
    #     ctx = torch.cat([q_W, g_stft], dim=-1)               # (B,2W)
    #     if self.refine_gate_mlp is not None:
    #         a = torch.sigmoid(self.refine_gate_mlp(ctx))  # (B,1)
    #     else:
    #         a = torch.sigmoid((ctx * self.refine_alpha_vec).sum(dim=-1, keepdim=True) + self.refine_alpha_bias)
    #     return a.view(-1, 1, 1)                           # (B,1,1) for broadcast
    
    def _alpha_from_ctx(self, q_W: torch.Tensor, g_stft: torch.Tensor) -> torch.Tensor:
        # q_W, g_stft: (B, W)
        x = torch.cat([q_W, g_stft], dim=-1)          # (B, 2W)
        if self.gate_mlp is not None:
            a = torch.sigmoid(self.gate_mlp(x))       # (B,1)
        else:
            a = torch.sigmoid(x @ self.gate_w + self.gate_b)  # (B,1)
        return a.view(-1, 1, 1) 

    
    def _apply_film_generic(self, predictor: nn.Linear, token: torch.Tensor, feat_q: torch.Tensor) -> torch.Tensor:
        gamma_strength = self.film_gamma_scale if self.cfg.learnable_film_strength else self.cfg.film_strength
        beta_strength = self.film_beta_scale if self.cfg.learnable_film_strength else self.cfg.film_strength
        gamma_beta = predictor(token)                 # (B, 2W)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = 1.0 + gamma_strength * torch.tanh(gamma)
        beta  = beta_strength * torch.tanh(beta)
        return gamma * feat_q + beta                  # (B, W)
    
    # ---------- predict 32-band residual mask (pre-upsampling)  ### NEW
    def _predict_mask_bands(self, q_W: torch.Tensor, g_stft: torch.Tensor) -> torch.Tensor:
        """
        q_W:     (B,W)
        g_stft:  (B,stft_td)  (token from stft path; usually stft_token_dim=256)
        returns: (B,BANDS) raw bands (pre-upsampling, pre-tanh, pre-γ)
        """
        if self.cfg.mask_predictor == "mlp":
            x = torch.cat([q_W, g_stft], dim=-1)  # ensure same scale
            return self.refine_mask_mlp(x)
        elif self.cfg.mask_predictor == "film":
            # FiLM: g (stft_td) conditions q (W), then linear to bands
            h = self._apply_film_generic(self.refine_mask_film_predictor, g_stft, q_W)  # (B,W)
            return self.refine_mask_out(h)             # (B,BANDS)
        else:
            return self.refine_mask_head(q_W, g_stft)

    # ---------- upsample bands to F bins (interp or interp->conv)  ### NEW
    def _upsample_bands_to_F(self, bands: torch.Tensor, Fbins: int) -> torch.Tensor:
        """
        bands: (B,BANDS) -> (B,F)
        """
        if self.cfg.mask_upsample == "interp":
            x = bands.unsqueeze(1)  # (B,1,BANDS)
            x = F.interpolate(x, size=Fbins, mode="linear", align_corners=False)
        else:
            x = bands @ self.W_psychoacoustic.T
            if self.refine_upconv is not None:
                x = x.unsqueeze(1)  # (B, 1, F)
        if self.refine_upconv is not None:
            x = x + self.refine_scale * self.refine_upconv(x)
        return x.squeeze(1)     # (B,F)

    # ---------- normalize per band (μ,σ over frequency bins inside each band)  ### NEW
    def _normalize_per_band(self, Y: torch.Tensor, num_bands: int) -> torch.Tensor:
        """
        Preferred: global per-frequency normalization using precomputed (mu, sigma).
        Fallback:  per-band Z-scoring on the fly (when global stats are unavailable).
        Y: (B, C, F)
        """
        B, C, Fbins = Y.shape

        # --- fast path: global stats available ---
        if (getattr(self, "stft_global_mu", None) is not None) and (getattr(self, "stft_global_sigma", None) is not None):
            mu = self.stft_global_mu
            sd = self.stft_global_sigma

            # If stats length != Fbins, interpolate along freq
            if mu.numel() != Fbins:
                mu_i = F.interpolate(mu.view(1, 1, -1), size=Fbins, mode="linear", align_corners=True).view(-1)
                sd_i = F.interpolate(sd.view(1, 1, -1), size=Fbins, mode="linear", align_corners=True).view(-1)
            else:
                mu_i, sd_i = mu, sd

            sd_i = sd_i.clamp_min(1e-6)
            # broadcast to (B,C,F)
            Y_norm = (Y - mu_i.view(1, 1, -1)) / sd_i.view(1, 1, -1)
            return Y_norm

        # --- fallback: per-band normalization (previous behavior) ---
        device = Y.device
        out = torch.empty_like(Y)
        band_id = torch.div(torch.arange(Fbins, device=device) * num_bands, Fbins, rounding_mode='floor')  # (F,)
        for b in range(num_bands):
            idx = (band_id == b).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            yb = Y[..., idx]                            # (B,C,Fb)
            mu = yb.mean(dim=-1, keepdim=True)
            sd = yb.std(dim=-1, keepdim=True).clamp_min(1e-6)
            out[..., idx] = (yb - mu) / sd
        return out

    # ---------- full refine pipeline (steps 7–10)  ### NEW
    def _refine_from_base(
        self,
        q_W: torch.Tensor,                 # (B,W)
        g_stft: torch.Tensor,              # (B,stft_td)
        Y_hat_base: torch.Tensor,          # (B,C,F)
        no_ref_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Implements:
        single: Y_final = Y_base + α * [ γ * tanh(Up(m)) ⊙ Φ(Y_base) ]
        film:   Y_res = Y_base + γ * tanh(Up(m)) ⊙ Φ(Y_base);
                Y_final = (1-α) * Y_base + α * FiLM(Y_res)
        """
        B, C, Fbins = Y_hat_base.shape
        if self.cfg.fusion == "attn":
            g_W = self.lora_expand_p(g_stft) if self.cfg.use_lora else self.attn_to_W(g_stft) 
        else: 
            g_W = self.stft_to_W(g_stft)
        # --- step 7: predict bands (pre-upsampling) ---
        bands = self._predict_mask_bands(q_W=q_W, g_stft=g_W)                # (B,BANDS)

        # --- step 8: upsample -> tanh (post-tanh L1) -> *γ ---
        mask_full = self._upsample_bands_to_F(bands, Fbins)                     # (B,F)
        mask_full = torch.tanh(mask_full)                                       # [-1,1]
        if self.cfg.put_mask_l1_in_outputs:
            self._last_refine_mask_l1 = mask_full.abs().mean()                      # post-tanh, pre-γ
        residual = (self.refine_mask_gamma * mask_full).unsqueeze(1)            # (B,1,F)

        # --- step 9: normalize base using global stats (already patched) ---
        Y_norm = self._normalize_per_band(Y_hat_base, self.cfg.mask_num_bands)  # (B,C,F)
        # --- α gate from [q_W || g_W] ---
        if self.cfg.gating == "single":
            alpha = self._alpha_from_ctx(q_W=q_W, g_stft=g_W)    # (B,1,1)

        if self.cfg.gating == "film":
            # 2) FiLM over STFT, conditioned on [q_W || g_W || bands]
            film_ctx = torch.cat([q_W, g_W, residual.squeeze(1)], dim=-1)                      # (B,2W+BANDS)
            Y_res_film = self._apply_film_generic(self.film_predictor,film_ctx,residual.squeeze(1)).unsqueeze(1)
            alpha = self._alpha_from_ctx(q_W=Y_norm.squeeze(1), g_stft=Y_res_film.squeeze(1))
            # 3) final mix with α (only here)
            Y_final = Y_hat_base + alpha * Y_res_film

            # no-ref bypass + ghost
            if no_ref_mask is not None and no_ref_mask.any():
                out = torch.where(no_ref_mask.view(-1,1,1), Y_hat_base, Y_final)
                if self.cfg.masked_grad_mode == "ghost":
                    m = no_ref_mask.view(-1,1,1).to(dtype=out.dtype)
                    ghost = m * Y_res_film * self.cfg.masked_grad_scale
                    out = out + (ghost - ghost.detach())
                return out
            return Y_final

        else:
            # ---- single-gate variant: add gated residual ONCE (no second combine) ----
            # stft references, Y_hat_base and generated output are already in log space, they are passed through a log function in dataset before being fed into this model
            delta = alpha * residual # You can multiply residual by Y_hat_base or Y_norm instead of Y_hat_base        # (B,C,F)
            Y_final = Y_hat_base + delta
            if self.cfg.spectral_loss_mask:
                self.loss_env = self.env_loss_fn(Y_final, Y_hat_base)
            self._last_alpha = alpha.detach()
            if no_ref_mask is not None and no_ref_mask.any():
                out = torch.where(no_ref_mask.view(-1,1,1), Y_hat_base, Y_final)
                if self.cfg.masked_grad_mode == "ghost":
                    m = no_ref_mask.view(-1,1,1).to(dtype=out.dtype)
                    ghost = m * delta * self.cfg.masked_grad_scale
                    out = out + (ghost - ghost.detach())
                return out
            return Y_final
        
        
    # **************************************************************************************************
    
    def lora_expand_p(self, p: torch.Tensor) -> torch.Tensor:
        """
        LoRA-style low-rank expansion from 64-d to 512-d.
        Args:
            p: (B, 64) attention output (or whatever your d_attn is)
        Returns:
            delta: (B, 512) low-rank expanded vector (scaled)
        """
        # Ensure we don't accidentally change dtype/device
        # (Not strictly necessary if modules are on same device)
        z = self.lora_dropout(p)                          # (B, 64)
        z = self.lora_up(z)                               # (B, r)
        z = F.gelu(z)                                     # nonlinearity for a bit more expressivity
        delta = self.lora_down(z)                         # (B, 512)
        return self.lora_scale * delta                    # (B, 512)

    
    def fuse_with_trunk(
        self,
        trunk: torch.Tensor,                  # (B,W)
        token: torch.Tensor,                  # (B,W) fused ref token (from query or stft path)
        no_ref_mask: Optional[torch.Tensor]   # (B,)
    ) -> torch.Tensor:
        """
        Merge fused reference token with the query/trunk per cfg.gating.
        - 'single': residual α-mix: out = α*token + (1-α)*trunk
        - 'film':   residual FiLM:  out = (1+γ(token))*trunk + β(token)
        Ghost grads: when masked, value is trunk but grads flow through 'token'.
        """

        if self.cfg.gating == "single":
            token_upscale = self.lora_expand_p(token) # or maybe self.attn_to_W
            trunk_n = self.ln_trunk_for_film(trunk)
            token_n = self.ln_token_for_film(token_upscale)
            x = torch.cat([trunk_n, token_n], dim=-1)  # (B,D) 
            if self.cfg.gate_type == "mlp":
                alpha = torch.sigmoid(self.gate_mlp(x))                 # (B,1)
            else:
                alpha = torch.sigmoid(x @ self.gate_w + self.gate_b)    # (B,1)
            fused = alpha * token_upscale + (1.0 - alpha) * trunk
        elif self.cfg.gating == "film":
            assert self.film_predictor is not None, "film gating requested but film_predictor not initialized"
            if self.cfg.fusion == "attn":
                token_upscale = self.lora_expand_p(token) if self.cfg.use_lora else self.attn_to_W(token) 
            elif self.mode != "query" and self.cfg.fusion == "bahdanau": 
                token_upscale = self.stft_to_W(token)
            else:
                token_upscale = token
            x = torch.cat([trunk, token_upscale], dim=-1)  # (B,D) 
            fused = self._apply_film_generic(self.film_predictor, x, trunk)
            delta_mag = (fused - trunk).pow(2).mean(dim=-1, keepdim=True).sqrt().detach()
            x = torch.cat([fused, trunk, delta_mag], dim=-1)
            if self.cfg.gate_type == "mlp":
                alpha = torch.sigmoid(self.gate_mlp(x))                 # (B,1)
            else:
                alpha = torch.sigmoid(x @ self.gate_w + self.gate_b)    # (B,1)
            fused = alpha * fused + (1.0 - alpha) * trunk

        else:
            fused = trunk  # fallback (shouldn't happen)
        self._last_alpha = alpha.detach()
        # Masking + optional ghost gradients
        if no_ref_mask is not None and no_ref_mask.any():
            out = torch.where(no_ref_mask.view(-1, 1), trunk, fused)
            if self.cfg.masked_grad_mode == "ghost":
                m = no_ref_mask.view(-1, 1).to(dtype=out.dtype)
                ghost = m * token * self.cfg.masked_grad_scale
                out = out + (ghost - ghost.detach())
            return out
            
        return fused
    
    def _fuse_refs_over_K(self, rW: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
        """
        rW:   (B,K,W)  per-ref tokens
        trunk:(B,W)    query token
        returns (B,W)
        """
        if self.cfg.fusion == "bahdanau":
            return self._mix_refs_additive(rW, trunk)
        else:
            q = trunk.unsqueeze(1)                # (B,1,W)
            rW = self.ref_proj(rW)
            q = self.trunk_proj(q)
            out, _ = self.ref_cross_attn(q, rW, rW)
            out = out.squeeze(1)
            return out

    # --------------------------- core new pieces ---------------------------

    def _project_stfts(self, stft_refs: torch.Tensor) -> torch.Tensor:
        """
        stft_refs: (B,K,F,T) -> x: (B,K,T,W)
        """
        B, K, nF, T = stft_refs.shape
        x = stft_refs.permute(0, 1, 3, 2).contiguous()  # (B,K,T,F)
        x = self.stft_freq_proj(x)                      # (B,K,T,W)
        # + time positional encoding
        pe = self._make_time_pe(T, x.device)            # (T,W)
        x = F.layer_norm(x, (x.shape[-1],))
        x = x + pe.unsqueeze(0).unsqueeze(0)
        return x  # (B,K,T,W)

    def _windowed_center_over_time(self, x: torch.Tensor, time_index: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        x: (B,K,T,W), time_index: (B,), returns r: (B,K,W) per-ref center token at t.
        Uses self.time_self_attn with Q = center token, K/V = window tokens.
        """
        B, K, T, W = x.shape
        pad = window_size // 2
        t0 = time_index.clamp(0, T - 1)                               # (B,)
        # Gather windows per (B,K)
        # Build indices for each offset
        offs = torch.arange(-pad, pad + 1, device=x.device)           # (win,)
        t_win = (t0.unsqueeze(-1) + offs).clamp(0, T - 1)             # (B,win)
        # Expand to K and gather: result (B,K,win,W)
        t_win = t_win.unsqueeze(1).expand(B, K, -1)
        gather = []
        for j in range(window_size):
            idx = t_win[..., j]                                       # (B,K)
            gather.append(x[torch.arange(B)[:,None], torch.arange(K)[None,:], idx])  # (B,K,W)
        Wseq = torch.stack(gather, dim=2)                             # (B,K,win,W)

        # Q is the center element at offset=pad
        q = Wseq[:,:,pad:pad+1,:]                                     # (B,K,1,W)

        Q = q.reshape(B*K, 1, W)
        KV = Wseq.reshape(B*K, window_size, W)
        out, _ = self.time_self_attn(Q, KV, KV)                       # (B*K,1,W)
        r = out.reshape(B, K, W)
        return r

    def _mix_refs_additive(self, r: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        r: (B,K,W) per-ref tokens at the chosen time
        q: (B,W)   trunk/query features
        returns g_t: (B,W) = sum_r softmax(a^T tanh(Wv r + Wq q + b)) * r
        """
        h = torch.tanh(self.ref_score_Wv(r) + self.ref_score_Wq(q).unsqueeze(1))  # (B,K,H)
        scores = self.ref_score_a(h).squeeze(-1)                                  # (B,K)
        w = torch.softmax(scores, dim=1)                                          # (B,K)
        g_t = torch.einsum('bk,bkw->bw', w, r)                                    # (B,W)
        return g_t

    # --------------------------- public reusable APIs ---------------------------

    @torch.no_grad()
    def _shape_check(self, x, name, dims):
        assert x is None or x.dim() == dims, f"{name} must have {dims} dims, got {None if x is None else x.shape}"

    def query_output_token(
        self,
        trunk: torch.Tensor,                 # (B,W) encoded query token from trunk
        h_refs: torch.Tensor,                # (B,K,D_in) raw ref features BEFORE trunk
        ref_ids: Optional[torch.Tensor]=None # (B,K) optional ref IDs for learned embeddings
    ) -> torch.Tensor:
        """
        Build a single reference-derived token for the query branch using h_refs (no STFT).
        Steps:
        1) encode refs with the same trunk → (B,K,W)
        2) (optional) add ref-ID embeddings → (B,K,W)
        3) fuse across refs using cfg.fusion ('attn' or 'bahdanau') → (B,W)
        """
            
        B, K, Din = h_refs.shape
        # 1) encode refs through the same trunk as the query
        r = h_refs.view(B * K, Din)
        rW = self.trunk(r)                    # (B*K, W)
        rW = rW.view(B, K, self.W)            # (B, K, W)

        # 2) optional ref-ID embeddings
        if (self.cfg.use_ref_id_tokens
            and self.ref_id_embed is not None
            and ref_ids is not None):
            id_tok = self.ref_id_embed(ref_ids)           # (B,K,W)
            id_tok = self.ref_id_dropout(id_tok)
            rW = F.layer_norm(rW + id_tok, (self.W,))

        # 3) fuse across refs
        g = self._fuse_refs_over_K(rW, trunk)
        
        return g


    def stft_output_token(
        self,
        trunk: torch.Tensor,                  # (B,W)  query/trunk features at time t
        stft_refs: torch.Tensor,              # (B,K,F,T)
        time_index: Optional[torch.Tensor],   # (B,)   center time index for each sample
        ref_ids: Optional[torch.Tensor] = None  # (B,K) optional reference indices
    ) -> torch.Tensor:
        """
        Build the single STFT token g_t used by multiple heads.
        """
        B, K, nF, T = stft_refs.shape
        x = self._project_stfts(stft_refs)                         # (B,K,T,W)

        if self.cfg.self_attention_time:
            if self.cfg.temporal_scope == "global":
                # flatten (B,K) for time encoder
                B, K, T, W = x.shape
                x_flat = x.view(B * K, T, W)  # (BK, T, W)

                # prepend CLS token
                cls = self.cls_token.to(x_flat.dtype).to(x_flat.device).expand(x_flat.size(0), -1, -1)  # (BK, 1, W)
                seq = torch.cat([cls, x_flat], dim=1)  # (BK, T+1, W)

                # full self-attention over time
                y = self.time_encoder_global(seq)       # (BK, T+1, W)

                # take CLS as pooled representation
                r = y[:, 0, :].view(B, K, W)           # (B, K, W)
            # Only apply local attention when time_index provided; otherwise global mean.
            else:
                r = self._windowed_center_over_time(x, time_index, self.cfg.window_size)  # (B,K,W)

        # Optional ref-ID embeddings (learned identity signal)
        if (self.cfg.use_ref_id_tokens
            and self.ref_id_embed is not None
            and ref_ids is not None):
            id_tok = self.ref_id_embed(ref_ids)  # (B,K,W)
            id_tok = self.ref_id_dropout(id_tok)
            r = F.layer_norm(r + id_tok, (self.stft_td,))


        # Query-conditioned additive mixing across refs
        g_t = self._fuse_refs_over_K(r, trunk)
        return g_t

    # --------------------------- main forward ---------------------------

    def forward(
        self,
        h_q: torch.Tensor,                            # (B, Din_trunk)
        h_refs: Optional[torch.Tensor] = None,        # (B, K, Din_ref)  (unused in this head)
        stft_refs: Optional[torch.Tensor] = None,     # (B, K, F, T)
        time_index: Optional[torch.Tensor] = None,    # (B,)
        no_ref_mask: Optional[torch.Tensor] = None,   # (B,)
        reference_idx: Optional[torch.Tensor] = None  # (B, K)  <= NEW
    ) -> torch.Tensor:
        """
        Produces per-mic STFT magnitude predictions: (B, sound_rez, N_frequencies).
        """
        # 1) trunk features
        target_dtype = self.trunk[0].weight.dtype
        h_q = h_q.to(dtype=target_dtype)
        if h_refs is not None:
            h_refs = h_refs.to(dtype=target_dtype)
        if stft_refs is not None:
            stft_refs = stft_refs.to(dtype=target_dtype)
            
        feat_q = self.trunk(h_q)   # (B,W)
        if self.mode == "refine" and self.cfg.mask_upsample != "interp":
            self.W_psychoacoustic = self.W_psychoacoustic.to(device=stft_refs.device, dtype=stft_refs.dtype)
        # --- respect_no_ref_mask bypass ---
        # assume feat_q = self.trunk(h_q) already computed (B,W)
        feat = None
        mask_keep = None

        if self.cfg.respect_no_ref_mask and no_ref_mask is not None and no_ref_mask.any():
            feat = feat_q.clone()
            if self.cfg.masked_grad_mode == "ghost":
                # push ghost from a dummy fused token (branch doesn’t matter; only gradients matter)
                dummy = feat_q  # or any small function of inputs
                ghost = no_ref_mask.view(-1,1).float() * dummy * self.cfg.masked_grad_scale
                feat = feat + (ghost - ghost.detach())
            mask_keep = ~no_ref_mask

        # ... existing preamble that builds feat_q, sets feat=None/mask_keep, etc. ...

        if self.mode == "query" and h_refs is not None:
            g = self.query_output_token(trunk=feat_q, h_refs=h_refs, ref_ids=reference_idx)  # (B, W or stft_td depending on path)
            fused = self.fuse_with_trunk(trunk=feat_q, token=g, no_ref_mask=no_ref_mask)

        elif self.mode in ("stft", "refine") and stft_refs is not None:   # ### CHANGED
            # guidance path normalization (abs->log->Z over time)
            mag = stft_refs.abs()
            logmag = (mag + 1e-6).log()
            mean = logmag.mean(dim=3, keepdim=True)
            std  = logmag.std(dim=3, keepdim=True) + 1e-6
            stft_refs_norm = (logmag - mean) / std

            # build guidance token g_t
            g = self.stft_output_token(
                trunk=feat_q, stft_refs=stft_refs_norm, time_index=time_index, ref_ids=reference_idx
            )  # (B, stft_td) for bahdanau; (B,W) for attn via stft_to_W inside _fuse_refs_over_K
            if self.mode == "stft":
                # feature-space fusion stays the same
                fused = self.fuse_with_trunk(trunk=feat_q, token=g, no_ref_mask=no_ref_mask)
            else:
                fused = feat_q

        else:
            fused = feat_q

        # respect_no_ref_mask logic (unchanged)
        if mask_keep is not None:
            feat = feat if feat is not None else feat_q.clone()
            feat[mask_keep] = fused[mask_keep]
        else:
            feat = fused

        # ---------- HEADS ----------
        # base prediction from feature-space fused token (same as before)
        outs = []
        for layer in self.STFT_linear:
            h = torch.tanh(layer(feat)) * 10.0   # (B, F)
            outs.append(h.unsqueeze(1))          # (B,1,F)
        Y_hat_base = torch.cat(outs, dim=1)      # (B, C, F)

        # refine branch replaces the return with STFT-space refined mix
        if self.mode == "refine":
            # note: g for bahdanau is stft_token_dim (256 by default) – perfect for our predictors
            Y_final = self._refine_from_base(
                q_W=feat_q, g_stft=g, Y_hat_base=Y_hat_base, no_ref_mask=no_ref_mask
            )  # (B, C, F)
            return Y_final

        # original return path
        return Y_hat_base