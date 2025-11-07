# generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverbRAGGenerator(nn.Module):
    """
    Minimal scaffold for ReverbRAG.
    For now:
      - validates/reshapes refs
      - optional lightweight projection to a context token (to be used later)
      - returns the *same* trunk token w (no fusion yet)
    """
    def __init__(self, n_freq: int, W: int, mode: str = "passthrough"):
        super().__init__()
        self.n_freq = n_freq
        self.W = W
        self.mode = mode

        # Tiny encoders we’ll actually use in the next step
        self.ref_freq_proj = nn.Linear(n_freq, min(256, n_freq))  # [F]→[td]
        self.token_proj = nn.Linear(min(256, n_freq), W)          # to W-space

    @staticmethod
    def _ensure_ref_shape(refs):
        # Expect [B,K,1,F,60] (log-mag). Accept [B,K,F,60] and add channel.
        if refs is None: return None
        if refs.ndim == 4:
            refs = refs.unsqueeze(2)
        return refs

    def pre_fuse(self, w_tokens: torch.Tensor,
                 refs_logmag: torch.Tensor = None,
                 refs_mask: torch.Tensor = None) -> torch.Tensor:
        """
        w_tokens: [B,T,W]
        refs_logmag: [B,K,1,F,60] log-mags
        refs_mask: [B,K] bool
        returns (possibly modified) [B,T,W]
        """
        if self.mode == "passthrough":
            return w_tokens

        # (Reserved: encode refs into a single context token 'g' and FiLM/gate w_tokens)
        # This will be implemented in the next step.
        return w_tokens