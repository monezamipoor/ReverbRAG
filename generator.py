# generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

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

    def __init__(self, cfg: dict):
        super().__init__()
        # ---- read config (minimal) ----
        rag = cfg.get("reverbrag", {})
        self._logger = None
        self.k: int = int(rag.get("k", 1))                        # we’ll still pick top-1
        self.num_bands: int = int(rag.get("num_bands", 32))       # refs_feats: [B,K,BANDS,4]
        self.mlp_hidden: List[int] = list(rag.get("mlp_hidden", [128, 256]))
        self.mlp_act: str = rag.get("mlp_act", "gelu")
        self.film_strength: float = float(rag.get("film_strength", 0.3))
        self.learnable_film: bool = bool(rag.get("learnable_film", True))
        self.feature_norm: str = str(rag.get("feature_norm", "layernorm"))  # {"none","layernorm"}

        in_dim = self.num_bands * 4  # features are [BANDS x 4], flattened
        layers = []
        last = in_dim
        for h in self.mlp_hidden:
            layers += [nn.Linear(last, h), ACTS[self.mlp_act]()]
            last = h
        # small dropout helps generalization but is not “an escape hatch”
        layers += [nn.Dropout(p=0.05)]
        self.feat_mlp = nn.Sequential(*layers)

        # Optional feature normalization (simple & robust)
        self.norm = nn.LayerNorm(in_dim) if self.feature_norm.lower() == "layernorm" else nn.Identity()

        # FiLM heads are lazily created on first call (we need W)
        self.film_gamma = None    # nn.Linear(last, W) created at runtime
        self.film_beta  = None    # nn.Linear(last, W)

        # Learnable global scales for FiLM (start >0 so it actually acts)
        if self.learnable_film:
            self.gamma_scale = nn.Parameter(torch.tensor(self.film_strength, dtype=torch.float32))
            self.beta_scale  = nn.Parameter(torch.tensor(self.film_strength, dtype=torch.float32))
        else:
            self.register_buffer("gamma_scale", torch.tensor(self.film_strength, dtype=torch.float32))
            self.register_buffer("beta_scale",  torch.tensor(self.film_strength, dtype=torch.float32))

        # Remember last seen W for sanity checks
        self._W: Optional[int] = None

    def set_logger(self, logger):
        self._logger = logger

    def _safe_log(self, payload: dict):
        # wandb-like: object with .log(dict). Accept None silently.
        if self._logger is not None:
            try:
                self._logger.log(payload)
            except Exception:
                pass

    # ----------------- helpers -----------------
    @staticmethod
    def _select_top1_valid(refs_feats: torch.Tensor,
                           refs_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Pick the first valid reference per batch: return [B, BANDS, 4].
        If none valid in a row, returns None for that row (handled upstream).
        """
        # refs_feats: [B,K,BANDS,4], refs_mask: [B,K] bool (True=valid)
        B, K, BANDS, C = refs_feats.shape
        if refs_mask is None:
            # Assume at least one reference is present; pick k=0
            return refs_feats[:, 0, :, :]

        # For each batch, pick the first True; fallback to first index if none
        # Build indices: [B]
        with torch.no_grad():
            valid = refs_mask.bool()                                  # [B,K]
            # first valid index or 0
            idx = torch.zeros(B, dtype=torch.long, device=refs_feats.device)
            any_valid = valid.any(dim=1)
            if any_valid.any():
                # get first True per row
                first_true = torch.argmax(valid.to(torch.int64), dim=1)  # returns 0 if all False too
                idx = torch.where(any_valid, first_true, idx)
        # gather
        ar = torch.arange(B, device=refs_feats.device)
        top1 = refs_feats[ar, idx, :, :]                                # [B,BANDS,4]
        return top1

    def _maybe_build_film_heads(self, W: int, h_dim: int, device: torch.device):
        """
        Lazily create FiLM heads sized to token width W.
        """
        if (self.film_gamma is None) or (self._W != W):
            self.film_gamma = nn.Linear(h_dim, W, bias=True).to(device)
            self.film_beta  = nn.Linear(h_dim, W, bias=True).to(device)
            # Zero-init so initial direction is stable; global scale controls magnitude
            nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight);  nn.init.zeros_(self.film_beta.bias)
            self._W = W

    # ----------------- public hook -----------------
    def pre_fuse(
        self,
        w_tokens: torch.Tensor,               # [B,T,W]
        refs_logmag: Optional[torch.Tensor],  # [B,K,1,F,60] (unused in this minimal path)
        refs_mask: Optional[torch.Tensor],    # [B,K] (bool)
        refs_feats: Optional[torch.Tensor],   # [B,K,BANDS,4]
    ) -> torch.Tensor:
        """
        Minimal, deterministic fusion:
          - select top-1 valid refs_feats per batch
          - flatten -> LayerNorm -> MLP
          - FiLM: w' = (1 + s_g * tanh(G z)) * w + s_b * tanh(B z)
        If no valid refs available for any row, return w unchanged for that row.
        """
        if refs_feats is None:
            return w_tokens

        B, T, W = w_tokens.shape
        device = w_tokens.device

        # Select one reference per batch (first valid)
        top1 = self._select_top1_valid(refs_feats, refs_mask)   # [B,BANDS,4] or None
        if top1 is None:
            return w_tokens

        # Flatten to [B, BANDS*4]
        z = top1.reshape(B, -1)  # [B, in_dim]
        z = self.norm(z)         # LayerNorm or Identity
        h = self.feat_mlp(z)     # [B, H]
        H = h.shape[-1]
        
        with torch.no_grad():
            self._safe_log({
                "rag/feat_mean": float(h.mean().item()),
                "rag/feat_std":  float(h.std(unbiased=False).item()),
            })

        # Build FiLM heads (lazy) if needed
        self._maybe_build_film_heads(W=W, h_dim=H, device=device)

        # Compute FiLM parameters
        gamma = self.film_gamma(h)   # [B, W]
        beta  = self.film_beta(h)    # [B, W]

        # Bound via tanh; scale with learnable/const strengths
        gamma = 1.0 + torch.tanh(gamma) * self.gamma_scale
        beta  = torch.tanh(beta) * self.beta_scale

        # Apply FiLM to all time steps
        # w_tokens: [B,T,W] -> broadcast gamma/beta over T
        
        with torch.no_grad():
            self._safe_log({
                "rag/gamma_l2": float(gamma.norm(p=2).item()/max(1.0, (B*W)**0.5)),
                "rag/beta_l2":  float(beta.norm(p=2).item()/max(1.0, (B*W)**0.5)),
                "rag/gamma_scale": float(self.gamma_scale.detach().item()),
                "rag/beta_scale":  float(self.beta_scale.detach().item()),
            })
        
        w_out = gamma.unsqueeze(1) * w_tokens + beta.unsqueeze(1)

        return w_out