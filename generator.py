import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

ACTS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "lrelu": lambda: nn.LeakyReLU(0.1, inplace=True),
    "silu": nn.SiLU,
}

class ReverbRAGGenerator(nn.Module):
    """
    Minimum viable RAG generator that actually 'listens'.

    Public API (do not change):
        pre_fuse(w_tokens, refs_logmag, refs_mask, refs_feats) -> w_tokens'
    """

    def __init__(self, cfg: dict, W: int):
        super().__init__()
        # ---- read config (minimal) ----
        rag = cfg.get("reverbrag", {})
        self._logger = None

        self.k: int = int(rag.get("k", 1))                        # still pick top-1
        self.num_bands: int = int(rag.get("num_bands", 32))       # refs_feats: [B,K,BANDS,4]
        self.mlp_hidden: List[int] = list(rag.get("mlp_hidden", [128, 256, 256]))
        self.mlp_act: str = rag.get("mlp_act", "lrelu")
        self.feature_norm: str = str(rag.get("feature_norm", "layernorm"))  # {"none","layernorm"}
        self.film_strength: float = float(rag.get("film_strength", 0.3))
        self.learnable_film: bool = bool(rag.get("learnable_film", True))
        self.gating: bool = bool(rag.get("gating", False))

        # --- Feature MLP over [BANDS x 4] ---
        in_dim = self.num_bands * 4
        layers = []
        last = in_dim
        for h in self.mlp_hidden:
            layers += [nn.Linear(last, h), ACTS[self.mlp_act]()]
            last = h
        layers += [nn.Dropout(p=0.05)]
        self.feat_mlp = nn.Sequential(*layers)

        self.h_dim = last  # last hidden size from feat_mlp

        # Optional feature normalization (simple & robust)
        self.norm = nn.LayerNorm(in_dim) if self.feature_norm.lower() == "layernorm" else nn.Identity()

        # --- FiLM heads (BUILT AT INIT; NO LAZY CREATION) ---
        self.film_gamma = nn.Linear(self.h_dim, W, bias=True)
        self.film_beta  = nn.Linear(self.h_dim, W, bias=True)
        # Zero-init so baseline starts near identity; global scale controls magnitude
        nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight);  nn.init.zeros_(self.film_beta.bias)

        # --- Optional scalar gate s in [0,1] from features h ---
        if self.gating:
            self.gate_head = nn.Linear(self.h_dim, 1)
            nn.init.zeros_(self.gate_head.weight); nn.init.zeros_(self.gate_head.bias)

        # --- Learnable global scales for FiLM (start >0 so it actually acts) ---
        if self.learnable_film:
            self.gamma_scale = nn.Parameter(torch.tensor(self.film_strength, dtype=torch.float32))
            self.beta_scale  = nn.Parameter(torch.tensor(self.film_strength, dtype=torch.float32))
        else:
            self.register_buffer("gamma_scale", torch.tensor(self.film_strength, dtype=torch.float32))
            self.register_buffer("beta_scale",  torch.tensor(self.film_strength, dtype=torch.float32))

    def set_logger(self, logger):
        self._logger = logger

    def _safe_log(self, payload: dict):
        if self._logger is not None:
            try:
                self._logger.log(payload)
            except Exception:
                pass

    @staticmethod
    def _select_top1_valid(refs_feats: torch.Tensor,
                           refs_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Pick the first valid reference per batch: return [B, BANDS, 4].
        """
        B, K, BANDS, C = refs_feats.shape
        if refs_mask is None:
            return refs_feats[:, 0, :, :]
        with torch.no_grad():
            valid = refs_mask.bool()                   # [B,K]
            idx = torch.zeros(B, dtype=torch.long, device=refs_feats.device)
            any_valid = valid.any(dim=1)
            if any_valid.any():
                first_true = torch.argmax(valid.to(torch.int64), dim=1)
                idx = torch.where(any_valid, first_true, idx)
        ar = torch.arange(B, device=refs_feats.device)
        return refs_feats[ar, idx, :, :]               # [B,BANDS,4]

    def pre_fuse(
        self,
        w_tokens: torch.Tensor,               # [B,T,W]
        refs_logmag: Optional[torch.Tensor],  # [B,K,1,F,60] (unused here)
        refs_mask: Optional[torch.Tensor],    # [B,K] (bool)
        refs_feats: Optional[torch.Tensor],   # [B,K,BANDS,4]
    ) -> torch.Tensor:
        """
        Deterministic fusion:
          - select top-1 valid refs_feats per batch
          - flatten -> (LayerNorm) -> MLP -> h
          - FiLM: w' = (1 + s_g * tanh(G h)) * w + s_b * tanh(B h)
          - Optional gate s in [0,1]: return (1-s)*w + s*w'
        """
        if refs_feats is None:
            return w_tokens

        B, T, W = w_tokens.shape

        # Select one reference per batch (first valid)
        top1 = self._select_top1_valid(refs_feats, refs_mask)   # [B,BANDS,4]
        if top1 is None:
            return w_tokens

        # Feature pathway
        z = top1.reshape(B, -1)           # [B, BANDS*4]
        z = self.norm(z)                  # LayerNorm or Identity
        h = self.feat_mlp(z)              # [B, h_dim]

        with torch.no_grad():
            self._safe_log({
                "rag/feat_mean": float(h.mean().item()),
                "rag/feat_std":  float(h.std(unbiased=False).item()),
            })

        # FiLM parameters
        gamma = self.film_gamma(h)        # [B, W]
        beta  = self.film_beta(h)         # [B, W]
        gamma = 1.0 + torch.tanh(gamma) * self.gamma_scale
        beta  =        torch.tanh(beta)  * self.beta_scale

        # Apply to all time steps
        fused = gamma.unsqueeze(1) * w_tokens + beta.unsqueeze(1)   # [B,T,W]

        if self.gating:
            s = torch.sigmoid(self.gate_head(h)).view(B, 1, 1)      # broadcast over T,W
            out = (1.0 - s) * w_tokens + s * fused
            with torch.no_grad():
                self._safe_log({
                    "rag/gate_mean": float(s.mean().item()),
                })
        else:
            out = fused

        with torch.no_grad():
            denom = max(1.0, (B * W) ** 0.5)
            self._safe_log({
                "rag/gamma_l2": float(gamma.norm(p=2).item() / denom),
                "rag/beta_l2":  float(beta.norm(p=2).item() / denom),
                "rag/gamma_scale": float(self.gamma_scale.detach().item()),
                "rag/beta_scale":  float(self.beta_scale.detach().item()),
            })

        return out