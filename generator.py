# generator.py
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
    Minimal RAG fusion module with three modes:
      - film   : FiLM(+optional gate) over w_tokens using retrieved features
      - concat : [w || h2w] then linear back to W (no gate)
      - input  : don't touch w_tokens; expose h (post-MLP) to encoder as an extra input

    Public API used by the model:
        - pre_fuse(w_tokens, refs_logmag, refs_mask, refs_feats) -> w_tokens'   (film / concat)
        - build_h(refs_feats, refs_mask) -> h                                 (input / reuse)
        - project_aux(h) -> aux_for_encoder (size W)                          (input)
    """

    def __init__(self, cfg: dict, W: int):
        super().__init__()
        rag = cfg.get("reverbrag", {}) if "reverbrag" in cfg else cfg
        print(f"[ReverbRAGGenerator] config: {rag}")

        # core knobs
        self.fusion: str = str(rag.get("fusion", "film")).lower()  # {"film","concat","input"}
        self.k: int = int(rag.get("k", 1))
        self.num_bands: int = int(rag.get("num_bands", 32))        # refs_feats: [B,K,32,4]
        self.gating: bool = bool(rag.get("gating", False))

        # feature MLP
        self.mlp_hidden: List[int] = list(rag.get("mlp_hidden", [128, 256, 256]))
        self.mlp_act: str = rag.get("mlp_act", "lrelu")
        self.feature_norm: str = str(rag.get("feature_norm", "layernorm"))  # {"none","layernorm"}

        in_dim = self.num_bands * 4
        layers = []
        last = in_dim
        for h in self.mlp_hidden:
            layers += [nn.Linear(last, h), ACTS[self.mlp_act]()]
            last = h
        layers += [nn.Dropout(p=0.05)]
        self.feat_mlp = nn.Sequential(*layers)
        self.h_dim = last

        self.norm = nn.LayerNorm(in_dim) if self.feature_norm.lower() == "layernorm" else nn.Identity()

        # FiLM heads (film mode)
        self.film_strength: float = float(rag.get("film_strength", 0.3))
        self.learnable_film: bool = bool(rag.get("learnable_film", True))

        self.film_gamma = nn.Linear(self.h_dim, W, bias=True)
        self.film_beta  = nn.Linear(self.h_dim, W, bias=True)
        nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight);  nn.init.zeros_(self.film_beta.bias)

        if self.learnable_film:
            self.gamma_scale = nn.Parameter(torch.tensor(self.film_strength, dtype=torch.float32))
            self.beta_scale  = nn.Parameter(torch.tensor(self.film_strength, dtype=torch.float32))
        else:
            self.register_buffer("gamma_scale", torch.tensor(self.film_strength, dtype=torch.float32))
            self.register_buffer("beta_scale",  torch.tensor(self.film_strength, dtype=torch.float32))

        # optional scalar gate s(h) for film
        if self.gating:
            self.gate_head = nn.Linear(self.h_dim, 1)
            nn.init.zeros_(self.gate_head.weight); nn.init.zeros_(self.gate_head.bias)
        else:
            self.gate_head = None

        # concat path: map h -> W, then [w||h2w] -> W (keeps decoder unchanged)
        self.h2w = nn.Linear(self.h_dim, W, bias=True)
        self.post_concat = nn.Linear(W + W, W, bias=True)

        # input path: projector to a fixed aux width = W (so encoders can add +W cleanly)
        self.aux_proj = nn.Linear(self.h_dim, W, bias=True)

        self._logger = None
        self._last_h: Optional[torch.Tensor] = None  # stash if caller wants to reuse

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

    # -------- feature builder (shared across all modes) --------
    def build_h(self, refs_feats: Optional[torch.Tensor],
                refs_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Returns h in R^{h_dim} per batch item (post-MLP), or None if no refs.
        """
        if refs_feats is None:
            self._last_h = None
            return None
        top1 = self._select_top1_valid(refs_feats, refs_mask)  # [B, BANDS, 4]
        if top1 is None:
            self._last_h = None
            return None
        B = top1.shape[0]
        z = top1.reshape(B, -1)        # [B, BANDS*4]
        z = self.norm(z)
        h = self.feat_mlp(z)           # [B, h_dim]
        self._last_h = h
        with torch.no_grad():
            self._safe_log({
                "rag/feat_mean": float(h.mean().item()),
                "rag/feat_std":  float(h.std(unbiased=False).item()),
            })
        return h

    # -------- input mode helper --------
    def project_aux(self, h: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Map h -> aux vector for encoder (size W). Returns None if h is None.
        """
        if h is None:
            return None
        return self.aux_proj(h)

    # -------- main hook (film / concat) --------
    def pre_fuse(
        self,
        w_tokens: torch.Tensor,               # [B,T,W]
        refs_logmag: Optional[torch.Tensor],  # [B,K,1,F,60] (unused here)
        refs_mask: Optional[torch.Tensor],    # [B,K] (bool)
        refs_feats: Optional[torch.Tensor],   # [B,K,BANDS,4]
    ) -> torch.Tensor:
        """
        film   : FiLM(+optional gate) on w_tokens
        concat : [w || h2w] -> W (no gate)
        input  : no-op (encoder consumed aux); returns w_tokens as-is
        """
        mode = self.fusion
        if mode == "input":
            # Encoder already consumed aux; leave tokens untouched.
            return w_tokens

        # build h once
        h = self.build_h(refs_feats, refs_mask)  # [B,h_dim] or None
        if h is None:
            return w_tokens

        B, T, W = w_tokens.shape

        if mode == "film":
            gamma = self.film_gamma(h)        # [B, W]
            beta  = self.film_beta(h)         # [B, W]
            gamma = 1.0 + torch.tanh(gamma) * self.gamma_scale
            beta  =        torch.tanh(beta)  * self.beta_scale

            fused = gamma.unsqueeze(1) * w_tokens + beta.unsqueeze(1)   # [B,T,W]

            if self.gating and (self.gate_head is not None):
                s = torch.sigmoid(self.gate_head(h)).view(B, 1, 1)      # broadcast over T,W
                out = (1.0 - s) * w_tokens + s * fused
                with torch.no_grad():
                    self._safe_log({"rag/gate_mean": float(s.mean().item())})
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

        elif mode == "concat":
            # h -> W, then [w || h2w] (broadcast across time) -> W
            h_w = self.h2w(h)                                  # [B,W]
            h_wt = h_w.unsqueeze(1).expand(B, T, W)            # [B,T,W]
            wx = torch.cat([w_tokens, h_wt], dim=-1)           # [B,T,2W]
            return self.post_concat(wx)                        # [B,T,W]

        # default fallthrough
        return w_tokens