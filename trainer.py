# trainer.py
import math
from copy import deepcopy
import os
from typing import Dict, Any, Optional, Tuple

from decay_features import build_ref_decay_features_bank
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from torchaudio.transforms import GriffinLim

# local
from evaluator import UnifiedEvaluator


# ------------ small NeRAF-style losses (as you provided) ------------
class SpectralConvergenceLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-12)

class LogSTFTMagnitudeLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type
    def forward(self, x_log, y_log):
        if self.loss_type == 'mse':
            return F.mse_loss(x_log, y_log)
        return F.l1_loss(x_log, y_log)

class STFTLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.sc = SpectralConvergenceLoss()
        self.lm = LogSTFTMagnitudeLoss(loss_type)
    def forward(self, x_log, y_log):
        x_mag = torch.exp(x_log) - 1e-3
        y_mag = torch.exp(y_log) - 1e-3
        return {'audio_sc_loss': self.sc(x_mag, y_mag),
                'audio_mag_loss': self.lm(x_log, y_log)}

# -------------------------------------------------------------------

def fs_to_stft_params(fs: int):
    if fs == 48000:   # RAF default
        return dict(N_freq=513, hop_len=256, win_len=512)
    if fs == 16000:   # SoundSpaces common
        return dict(N_freq=257, hop_len=128, win_len=256)
    # fallback consistent with evaluator
    n_freq = 513
    return dict(N_freq=n_freq, hop_len=256, win_len=512)


class Trainer:
    """
    Minimal, unified trainer for {NeRAF | AV-NeRF} × {RAF | SoundSpaces}.
    - train(): uses RandomSliceBatchSampler or EDCFullBatchSampler depending on YAML.
    - eval(): forces dataset_mode='full' (we need all T to compute metrics).
    - save_model(): checkpoint helper.
    """
    
    def __init__(
        self,
        model: nn.Module,
        opt_cfg: Dict[str, Any],
        run_cfg: Dict[str, Any],
        device: torch.device,
        baseline: str,
        visual_feat_builder,   # callable(batch, B)-> Tensor [B, Dv]
        stft_params: Optional[Dict[str, int]] = None,
        ref_bank: torch.Tensor = None, ref_bank_ids=None,
        ref_feats: torch.Tensor = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.run_cfg = run_cfg
        # self.scaler = GradScaler(enabled=True)
        self.log_every = int(run_cfg.get("log_every", 50))
        # STFT/ISTFT & Evaluator (parameters taken “from main.py”)
        fs = int(run_cfg.get("sample_rate", 48000))
        stftp = stft_params or fs_to_stft_params(fs)
        self.istft = GriffinLim(
            n_fft=(stftp["N_freq"] - 1) * 2,
            win_length=stftp["win_len"],
            hop_length=stftp["hop_len"],
            power=1,
        ).to(device=device, dtype=torch.float32)
        self.evaluator = UnifiedEvaluator(fs=fs, edc_bins=60, edc_dist="l1")

        # ---- ReverbRAG reference bank & decay features (CPU, once) ----
        self.ref_bank = ref_bank
        self.ref_bank_ids = ref_bank_ids

        def _is_valid_bank(t):
            if t is None: return False
            if not hasattr(t, "numel"): return False
            if t.numel() == 0: return False
            # must be [R,1,F,60]
            return (t.dim() == 4) and (t.shape[1] in (1,2)) and (t.shape[-1] == 60)

        hop_ms = (float(stftp["hop_len"]) / float(fs)) * 1000.0

        rv_cfg = run_cfg.get("reverbrag", None)
        if isinstance(rv_cfg, dict):
            num_bands = int(rv_cfg.get("num_bands", 32))
        else:
            num_bands = 32

        if ref_feats is not None and _is_valid_bank(self.ref_bank):
            self.ref_feats = ref_feats.to(torch.float32)
        elif _is_valid_bank(self.ref_bank):
            with torch.no_grad():
                bank_np = self.ref_bank.detach().cpu().numpy()   # safe for empty/invalid
                feats_np, stats = build_ref_decay_features_bank(
                    bank_np, num_bands=num_bands, hop_ms=hop_ms
                )
                self.ref_feats = torch.from_numpy(feats_np)      # [R,B,4] float32
        else:
            # disable all ref usage cleanly
            self.ref_bank = None
            self.ref_feats = None

        # ---- Loss configuration (fully config-driven; no baseline assumptions) ----
        loss_cfg = run_cfg.get("losses", {}) or {}
        self.baseline = loss_cfg.get("optimizer", "avnerf")
        # ---- Print final resolved losses (debug) ----
        print("\n================ LOSS CONFIG ================")
        for k, v in loss_cfg.items():
            print(f"{k:12s} : {v}")
        print("=============================================\n")

        # NeRAF-style STFT loss (SC + log-mag)
        mag_loss_type = loss_cfg.get("mag_loss_type", "mse")  # { "l1", "mse" }
        self.loss_fn = STFTLoss(loss_type=mag_loss_type)

        # Weights for each term
        self.w_sc  = float(loss_cfg.get("sc", 0.0))   # spectral convergence
        self.w_mag = float(loss_cfg.get("mag", 0.0))  # log-mag
        self.w_mse = float(loss_cfg.get("mse", 0.0))  # extra log-mse
        # Envelope + residual regularization
        self.w_env_rms = float(loss_cfg.get("env_rms", 0.0))  # RMS envelope (log-domain, full STFT)
        self.env_rms_loss_type = str(loss_cfg.get("env_rms_type", "l1"))  # {"l1","mse"}
        self.w_res_l2  = float(loss_cfg.get("res_l2", 0.0))   # L2 penalty on residual log-STFT

        # EDC weighting (support both new `losses.edc` and old `edc_loss`)
        self.lambda_edc = float(
            loss_cfg.get("edc", run_cfg.get("edc_loss", 0.0))
        )

        # This flag is now basically informational; EDC behaviour is tied to lambda_edc
        self.use_edc_full = bool(
            run_cfg.get("edc_full", self.lambda_edc > 0.0)
        )
        self._warned_slice_edc = False

        # Global scale for the final loss
        self.loss_factor = float(
            loss_cfg.get("global_scale", run_cfg.get("loss_factor", 1e-3))
        )

        # Gradient clipping
        grad_clip_cfg = run_cfg.get("grad_clip", {}) or {}
        self.grad_clip_enabled = bool(grad_clip_cfg.get("enabled", False))
        self.grad_clip_max_norm = float(grad_clip_cfg.get("max_norm", 1.0))
        self.grad_clip_norm_type = float(grad_clip_cfg.get("norm_type", 2.0))

        # Optional LR multiplier for temporal attention params
        self.temporal_lr_mult = float(run_cfg.get("temporal_attention_lr_mult", 1.0))

        # Build optimizer param groups (temporal stack can use a lower LR)
        temporal_module = getattr(self.model, "temporal_stack", None)
        temporal_on = bool(getattr(self.model, "use_temporal_attention", False)) and (temporal_module is not None)
        if temporal_on:
            temporal_param_ids = {id(p) for p in temporal_module.parameters()}
            temporal_params = [p for p in temporal_module.parameters() if p.requires_grad]
            base_params = [
                p for p in self.model.parameters()
                if p.requires_grad and (id(p) not in temporal_param_ids)
            ]
        else:
            temporal_params = []
            base_params = [p for p in self.model.parameters() if p.requires_grad]

        # Optimizer (abstract; two presets)
        if self.baseline == "avnerf":
            # 1:1 spirit of avnerf-trainer: Adam + warmup/exp decay inside train step
            lr = float(run_cfg.get("lr", 1e-3))
            param_groups = [{"params": base_params, "lr": lr}]
            if temporal_params:
                param_groups.append({
                    "params": temporal_params,
                    "lr": lr * self.temporal_lr_mult,
                    "name": "temporal_stack",
                })
            self.optimizer = torch.optim.Adam(param_groups)
            self._main_lr0 = lr
        else:
            # NeRAF (ICLR ’25 impl details): Adam(β1=0.9, β2=0.999, eps=1e-15), lr 1e-4→1e-8 exp
            lr_hi = 1e-4
            param_groups = [{"params": base_params, "lr": lr_hi}]
            if temporal_params:
                param_groups.append({
                    "params": temporal_params,
                    "lr": lr_hi * self.temporal_lr_mult,
                    "name": "temporal_stack",
                })
            self.optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-15)
            # compute per-step decay factor to reach 1e-8 by last step
            self._neraf_lr_hi, self._neraf_lr_lo = 1e-4, 1e-8
            self._neraf_total_steps = None  # set on first train()

        # Keep per-group base LR so schedules preserve temporal LR multiplier.
        self._group_base_lrs = [float(g["lr"]) for g in self.optimizer.param_groups]

        self.visual_feat_builder = visual_feat_builder
        if ref_bank is None or ref_bank.numel() == 0:
            self.ref_bank = None         # disables RAG path
        else:
            self.ref_bank = ref_bank     # [R, 1, F, 60] (log-mag)
        self.ref_bank_ids = ref_bank_ids or []
        self.rag_module = getattr(self.model, "rag_gen", None)

    # ------------- util: build time indices for both modes -------------
    @staticmethod
    def _make_t_idx(batch: Dict[str, torch.Tensor], device: torch.device, mode_full: bool):
        if mode_full:
            T = int(batch["stft"].shape[-1])
            B = batch["stft"].shape[0]
            t_idx = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1).expand(B, T, 1)
            return t_idx
        # slice mode
        slice_t = batch["slice_t"].to(device).long()
        return slice_t.float().view(slice_t.shape[0], 1, 1)  # [B,1,1]
    
    # -------------------------------------------------------------------

    def _gather_refs(self, batch_indices: torch.Tensor):
        """
        batch_indices: [B, K] with bank indices (or -1).
        returns: refs_logmag [B, K, 1, F, 60], mask [B, K] (True=valid)
        """
        if self.ref_bank is None or batch_indices is None:
            return None, None
        idx = batch_indices.clone()
        mask = idx >= 0
        # safe indexing: replace -1 with 0 to avoid crash, then mask later
        idx[~mask] = 0
        # pull from CPU bank then move to device
        refs = self.ref_bank.index_select(0, idx.view(-1)).view(
            idx.shape[0], idx.shape[1], *self.ref_bank.shape[1:]
        )  # [B,K,1,F,60]
        return refs.to(self.device, non_blocking=True), mask.to(self.device)

    def _gather_ref_feats(self, ref_idx: torch.Tensor):
        """
        ref_idx: [B,K] long -> returns [B,K,BANDS,4] float32 on device
        """
        if (ref_idx is None) or (self.ref_feats is None):
            return None
        idx = ref_idx.clone()
        idx[idx < 0] = 0
        flat = idx.view(-1)
        gathered = self.ref_feats.index_select(0, flat.cpu())  # [B*K, BANDS, 4] on CPU
        return gathered.view(*idx.shape, self.ref_feats.shape[1], 4).to(self.device, non_blocking=True)

    def _forward_batch(self, batch: Dict[str, torch.Tensor], mode_full: bool):
        mic  = batch["receiver_pos"].to(self.device).float()
        src  = batch["source_pos"].to(self.device).float()
        head = batch["orientation"].to(self.device).float()
        t_idx = self._make_t_idx(batch, self.device, mode_full)
        
        B = mic.shape[0]
        vfeat = self.visual_feat_builder(batch, B).to(self.device).float()

        # ---- ReverbRAG refs for this batch (optional) ----
        ref_idx = batch.get("ref_indices", None)
        refs_logmag = refs_mask = refs_feats = None
        if ref_idx is not None:
            refs_logmag, refs_mask = self._gather_refs(ref_idx)
            refs_feats = self._gather_ref_feats(ref_idx)

        # pure FP32 forward, like AV-NeRF
        pred = self.model(
            mic_xyz=mic, src_xyz=src, head_dir=head, t_idx=t_idx,
            visual_feat=vfeat,
            refs_logmag=refs_logmag, refs_mask=refs_mask,
            refs_feats=refs_feats,
        )

        return pred

    def _env_rms_loss(
            self,
            pred_env_log: torch.Tensor,   # [B, T_env] from envelope head (log_env_full)
            gt_log:       torch.Tensor,   # [B, C, F, T] or [B, 1, F, T] log-mag STFT
            loss_type:    str = "l1",
        ) -> torch.Tensor:
            """
            Global RMS envelope supervision on FULL STFTs.
            """

            assert gt_log.shape[-1] == 60, "env_rms expects full GT STFT with T=60 frames"

            # compute GT log-RMS envelope from STFT (all in log domain at the end)
            with torch.cuda.amp.autocast(enabled=False):
                gt_log32 = gt_log.float()

                mag_gt = (gt_log32.exp() - 1e-3).clamp_min(0.0)   # [B,C,F,T]

                E_gt = (mag_gt ** 2).sum(dim=-2)                  # sum over F → [B,C,T] or [B,T]

                if E_gt.dim() == 3:
                    E_gt = E_gt.mean(dim=1)                      # average channels → [B,T]

                rms_gt     = torch.sqrt(E_gt + 1e-8)              # [B,T]
                log_rms_gt = torch.log(rms_gt + 1e-6)             # [B,T]

            # enforce exact time AND batch alignment
            B_pred, T_env = pred_env_log.shape
            B_gt,   T_gt  = log_rms_gt.shape

            if T_gt != T_env:
                raise RuntimeError(
                    f"env_rms loss: time length mismatch between pred_env_log (T_env={T_env}) "
                    f"and GT log-RMS envelope (T={T_gt}). "
                    "Fix the envelope head output length or the GT envelope computation; "
                    "no implicit interpolation is performed."
                )

            if B_pred != B_gt:
                raise RuntimeError(
                    f"env_rms loss: batch mismatch between pred_env_log (B={B_pred}) "
                    f"and GT log-RMS envelope (B={B_gt}). "
                    "You probably passed per-slice envelopes instead of per-RIR envelopes, "
                    "or your STFT batch size doesn't match geometry batch size."
                )

            if loss_type == "mse":
                return F.mse_loss(pred_env_log, log_rms_gt)
            elif loss_type == "l1":
                return F.l1_loss(pred_env_log, log_rms_gt)
            else:
                raise ValueError(f"Unknown env_rms loss_type='{loss_type}'")



    # ---- compact EDC loss (computed on full STFT batches only) ----
    def _edc_loss(
        self,
        pred_log: torch.Tensor,   # [B, C, F, T] log-mag
        gt_log:   torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        NeRAF-style EDC:
          - magnitude = exp(log) - 1e-3
          - per-frame energy: sum_f mag^2
          - Schroeder integral over time
          - L1 on log10 energy curves (no z-score, no 0 dB anchoring)
        """
        assert pred_log.shape[-1] == 60

        with torch.cuda.amp.autocast(enabled=False):
            pred_log32 = pred_log.float()
            gt_log32   = gt_log.float()

            # mag reconstruction identical to NeRAF
            pred_mag = (pred_log32.exp() - 1e-3).clamp_min(0.0)
            gt_mag   = (gt_log32.exp()   - 1e-3).clamp_min(0.0)

            # energy per frame: sum over F, keep channels separate
            # E: [B, C, T] or [B, T] if no channel dim
            if pred_mag.dim() == 4:
                E_pred = (pred_mag**2).sum(dim=-2)        # [B, C, T]
                E_gt   = (gt_mag**2).sum(dim=-2)
            else:
                E_pred = (pred_mag**2).sum(dim=-2)        # [B, T]
                E_gt   = (gt_mag**2).sum(dim=-2)

            # Schroeder integral over time (last dim)
            S_pred = torch.flip(torch.cumsum(torch.flip(E_pred, dims=[-1]), dim=-1), dims=[-1])
            S_gt   = torch.flip(torch.cumsum(torch.flip(E_gt,   dims=[-1]), dim=-1), dims=[-1])

            # log10 in energy domain (no 0 dB anchoring, no z-score)
            s_pred = torch.log10(S_pred.clamp_min(eps))
            s_gt   = torch.log10(S_gt  .clamp_min(eps))

            # prepare mask
            if valid_mask is not None:
                # valid_mask: [B, T] -> broadcast to channels if needed
                if s_pred.dim() == 3:  # [B, C, T]
                    m = valid_mask.unsqueeze(1).to(s_pred.dtype)  # [B,1,T]
                else:                   # [B, T]
                    m = valid_mask.to(s_pred.dtype)
            else:
                m = torch.ones_like(s_pred, dtype=s_pred.dtype)

            diff = (s_pred - s_gt) * m

            # L1 over time (and channels if present), normalize by #valid
            denom = m.sum(dim=tuple(range(1, s_pred.dim()))).clamp_min(1.0)  # [B]
            per_b = diff.abs().sum(dim=tuple(range(1, s_pred.dim()))) / denom

            return per_b.mean()
        

    def _loss(self, pred_log, gt_log):
        pred_log = pred_log.float()
        gt_log   = gt_log.float()

        parts = {}
        total = 0.0

        # ----- STFT-based losses (SC + log-mag) -----
        if self.w_sc != 0.0 or self.w_mag != 0.0:
            stft_parts = self.loss_fn(pred_log, gt_log)  # returns {"audio_sc_loss", "audio_mag_loss"}
            parts.update(stft_parts)

            if self.w_sc != 0.0:
                total = total + self.w_sc * stft_parts["audio_sc_loss"]
            if self.w_mag != 0.0:
                total = total + self.w_mag * stft_parts["audio_mag_loss"]

        # ----- Extra log-MSE term (AV-NeRF-style) -----
        if self.w_mse != 0.0:
            mse_val = F.mse_loss(pred_log, gt_log)
            parts["mse"] = mse_val
            total = total + self.w_mse * mse_val
            
        # ----- Residual L2 regularization (envelope branch) -----
        if self.w_res_l2 != 0.0:
            dbg = getattr(self.model, "debug_outputs", None)
            if isinstance(dbg, dict):
                r_log = dbg.get("residual_log", None)
                if r_log is not None:
                    res_l2 = (r_log ** 2).mean()
                    parts["res_l2"] = res_l2
                    total = total + self.w_res_l2 * res_l2

        # If you set all weights to 0.0, total == 0 and you’ve soft-bricked training.
        return total, parts, None


    @staticmethod
    def _format_parts(parts_dict, edc_val):
        flat = {k: float(v.item()) if hasattr(v, "item") else float(v) for k, v in parts_dict.items()}
        if edc_val is not None:
            flat["edc_loss"] = float(edc_val.item())
        return flat

    def _update_lr(self, step_idx: int, steps_per_epoch: int, max_epoch: int, baseline: str):
        if baseline == "avnerf":
            # avnerf-trainer warmup 10% then exponential 0.1^(2 * progress)
            warmup = int(0.1 * max_epoch) * steps_per_epoch
            if step_idx < warmup:
                main_lr = self._main_lr0 * (step_idx / max(1, warmup))
            else:
                total = max_epoch * steps_per_epoch
                main_lr = self._main_lr0 * (0.1 ** (2 * (step_idx - warmup) / max(1, (total - warmup))))

            ratio = main_lr / max(self._main_lr0, 1e-12)
            for i, g in enumerate(self.optimizer.param_groups):
                g["lr"] = self._group_base_lrs[i] * ratio
            return self.optimizer.param_groups[0]["lr"]
        # NeRAF: per-step exponential from 1e-4 to 1e-8
        if self._neraf_total_steps is None:
            self._neraf_total_steps = max_epoch * steps_per_epoch
        t = step_idx / max(1, self._neraf_total_steps)
        main_lr = self._neraf_lr_hi * ((self._neraf_lr_lo / self._neraf_lr_hi) ** t)
        ratio = main_lr / max(self._neraf_lr_hi, 1e-12)
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = self._group_base_lrs[i] * ratio
        return self.optimizer.param_groups[0]["lr"]

    # inside Trainer class
    @staticmethod
    def _pack_slices_to_full(x: torch.Tensor, T: int) -> torch.Tensor:
        B = x.shape[0] // T
        return (
            x.view(B, T, x.shape[1], x.shape[2], x.shape[3])  # [B,T,C,F,1]
            .permute(0, 2, 3, 1, 4)                          # [B,C,F,T,1]
            .squeeze(-1)                                     # [B,C,F,T]
            .contiguous()
        )
        
    @staticmethod
    def _pack_grouped_slice_batch(batch: Dict[str, torch.Tensor], T: int) -> Dict[str, torch.Tensor]:
        """
        Convert grouped slice batch (B_slices = B_rir*T) into full-sequence batch for model forward.
        Assumes EDCFullBatchSampler ordering: contiguous T slices per RIR, in time order.
        """
        B_s = batch["receiver_pos"].shape[0]
        if (B_s % T) != 0:
            raise RuntimeError(f"Cannot pack grouped slices: B_s={B_s} not divisible by T={T}")
        B = B_s // T

        if "slice_t" in batch:
            slice_t = batch["slice_t"].view(B, T)
            expected = torch.arange(T, device=slice_t.device).view(1, T).expand(B, T)
            if not torch.equal(slice_t.long(), expected.long()):
                raise RuntimeError(
                    "Grouped slice batch is not time-ordered per RIR. "
                    "temporal_attention requires contiguous slice_t = [0..T-1] within each group."
                )

        def _first_per_rir(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, T, *x.shape[1:])[:, 0, ...]

        st = batch["stft_slice"]  # [B_s, 1, F]
        st_full = st.view(B, T, st.shape[1], st.shape[2]).permute(0, 2, 3, 1).contiguous()  # [B,1,F,T]

        out = {
            "receiver_pos": _first_per_rir(batch["receiver_pos"]),
            "source_pos": _first_per_rir(batch["source_pos"]),
            "orientation": _first_per_rir(batch["orientation"]),
            "stft": st_full,
        }

        if "feat_rgb" in batch:
            out["feat_rgb"] = _first_per_rir(batch["feat_rgb"])
        if "feat_depth" in batch:
            out["feat_depth"] = _first_per_rir(batch["feat_depth"])
        if "ref_indices" in batch:
            out["ref_indices"] = _first_per_rir(batch["ref_indices"])

        return out

    def save_model(self, path: str, extra: Optional[Dict[str, Any]] = None):
        state = {
            "model": self.model.state_dict(),
            "baseline": self.baseline,
        }
        # include optimizer state for easier resume
        state["optimizer"] = self.optimizer.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)
        
    def _grad_norm_of(self, submodule_name: str) -> float:
        mod = getattr(self.model, submodule_name, None)
        if mod is None:
            return 0.0

        total_sq, count = 0.0, 0
        for p in mod.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq += float(g.pow(2).sum().item())
                count += g.numel()
        return (total_sq / max(count, 1)) ** 0.5  # RMS grad

    def _grad_norm_total(self) -> float:
        total_sq, count = 0.0, 0
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq += float(g.pow(2).sum().item())
                count += g.numel()
        return (total_sq / max(count, 1)) ** 0.5

    def load_checkpoint(self, path: str, load_optimizer: bool = False, lr_override=None):
        chk = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chk["model"], strict=True)
        if load_optimizer and "optimizer" in chk:
            self.optimizer.load_state_dict(chk["optimizer"])
        # optional LR override (keeps things simple)
        if lr_override is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = float(lr_override)
        print(f"[resume] loaded {path} (load_optimizer={load_optimizer}, lr_override={lr_override})")


    # ---- helpers ----
    def _grad_stats_of(self, module) -> Tuple[float, float, int]:
        """Return (l2_norm, rms, n_params_with_grads). Call AFTER unscale_()."""
        total_sq = torch.zeros((), device="cuda", dtype=torch.float32)
        count = 0
        for p in module.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            total_sq += (g * g).sum()
            count += g.numel()
        l2  = float(torch.sqrt(total_sq + 1e-12).item())
        rms = float(torch.sqrt(total_sq / max(count, 1) + 1e-12).item())
        return l2, rms, count

    def _grad_stats_total(self) -> Tuple[float, float, int]:
        total_sq = torch.zeros((), device="cuda", dtype=torch.float32)
        count = 0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            total_sq += (g * g).sum()
            count += g.numel()
        l2  = float(torch.sqrt(total_sq + 1e-12).item())
        rms = float(torch.sqrt(total_sq / max(count, 1) + 1e-12).item())
        return l2, rms, count


    # ------------------------- API -------------------------
    def train(
        self, train_loader, val_loader, epochs: int, wandb_run=None,
        save_dir: str = None, save_every: int = 10, cfg_copy_path: str = None,
        resume_ckpt: str = None, resume_load_optimizer: bool = False, resume_lr_override=None,
    ):
        # one-time: say which sampler we actually have
        bs_obj = getattr(train_loader, "batch_sampler", None)
        sampler_name = type(bs_obj).__name__ if bs_obj is not None else type(train_loader.sampler).__name__
        print(f"[train] using sampler={sampler_name}")

        # ---- optional resume ----
        if resume_ckpt:
            self.load_checkpoint(
                resume_ckpt,
                load_optimizer=resume_load_optimizer,
                lr_override=resume_lr_override
            )

        self.model.train()
        step = 0
        for epoch in range(1, epochs + 1):
            if hasattr(self.model, "set_logger"):
                self.model.set_logger(wandb_run)  # NEW: let model/generator know about wandb
            pbar = tqdm(total=len(train_loader), desc=f"[train] epoch {epoch}", leave=False)
            for batch in train_loader:
                lr = self._update_lr(step, len(train_loader), epochs, self.baseline)
                # infer mode: slice batches carry 'stft_slice'
                is_slice = ("stft_slice" in batch)
                bs_obj = getattr(train_loader, "batch_sampler", None)
                sampler_name = type(bs_obj).__name__ if bs_obj is not None else type(train_loader.sampler).__name__
                temporal_on = bool(getattr(self.model, "use_temporal_attention", False))

                if is_slice and temporal_on:
                    if sampler_name != "EDCFullBatchSampler":
                        raise RuntimeError("temporal_attention requires EDCFullBatchSampler (grouped contiguous slices).")
                    T = int(getattr(train_loader.dataset, "max_frames", 60))
                    fwd_batch = self._pack_grouped_slice_batch(batch, T=T)

                    gt = fwd_batch["stft"].to(self.device).float()         # [B_rir,1,F,T]
                    pred = self._forward_batch(fwd_batch, mode_full=True)  # [B_rir,C,F,T]
                    is_slice = False  # downstream loss code should treat this as full-mode
                else:
                    gt = (batch["stft_slice"] if is_slice else batch["stft"]).to(self.device).float()
                    if is_slice:
                        gt = gt.unsqueeze(-1)  # [B_slices, 1, F, 1]
                    pred = self._forward_batch(batch, mode_full=not is_slice)

                # spectral loss (pass a string or bool to keep API simple)
                total, parts, edc_val = self._loss(pred, gt)

                # inside train() loop, after total, parts, edc_val = self._loss(...)
                # EDC and/or global envelope RMS both need full STFTs
                if (self.lambda_edc > 0.0) or (self.w_env_rms > 0.0):
                    bs_obj = getattr(train_loader, "batch_sampler", None)
                    sampler_name = type(bs_obj).__name__ if bs_obj is not None else type(train_loader.sampler).__name__

                    if is_slice:
                        # slice mode: we expect EDCFullBatchSampler to give contiguous T slices per RIR
                        if sampler_name != "EDCFullBatchSampler":
                            if self.w_env_rms > 0.0:
                                raise RuntimeError(
                                    "env_rms > 0 requires EDCFullBatchSampler (contiguous T slices per RIR)."
                                )
                            # only EDC is on -> just warn once
                            if self.lambda_edc > 0.0 and not getattr(self, "_warned_sampler", False):
                                print("[warn] EDC loss expects EDCFullBatchSampler (contiguous T slices per SID).")
                                self._warned_sampler = True
                        else:
                            T = int(getattr(train_loader.dataset, "max_frames", 60))
                            if (gt.shape[0] % T) == 0:
                                # pack slices -> full [B, C, F, T]
                                pred_full = self._pack_slices_to_full(pred, T)
                                gt_full   = self._pack_slices_to_full(gt,   T)

                                # EDC term (optional)
                                if self.lambda_edc > 0.0:
                                    edc_term = self._edc_loss(pred_full, gt_full, valid_mask=None)
                                    total = total + self.lambda_edc * edc_term
                                    edc_val = edc_term.detach()

                                # Envelope RMS term (optional)
                                # Envelope RMS term (optional)
                                if self.w_env_rms > 0.0:
                                    dbg = getattr(self.model, "debug_outputs", None)
                                    log_env_full = None
                                    if isinstance(dbg, dict):
                                        log_env_full = dbg.get("log_env_full", None)

                                    if log_env_full is None:
                                        raise RuntimeError(
                                            "env_rms > 0 requires model.debug_outputs['log_env_full'] "
                                            "to be populated by the envelope head."
                                        )

                                    if log_env_full.dim() != 2:
                                        raise RuntimeError(
                                            f"env_rms expects log_env_full to be 2D [B, T_env], "
                                            f"got shape {tuple(log_env_full.shape)}."
                                        )

                                    B_env, T_env = log_env_full.shape
                                    B_full       = gt_full.shape[0]   # number of RIRs in this batch

                                    # Case 1: model already gives one envelope per RIR → [B_full, T_env]
                                    if B_env == B_full:
                                        pred_env_for_loss = log_env_full

                                    # Case 2: slice-mode output: one envelope per SLICE → [B_full * T, T_env]
                                    elif (B_env % B_full) == 0:
                                        T_frames = T  # max_frames from dataset, used above for _pack_slices_to_full
                                        expected = B_full * T_frames
                                        if B_env != expected:
                                            raise RuntimeError(
                                                "env_rms: envelope batch size does not match [B_full * T_frames]. "
                                                f"Got B_env={B_env}, expected {expected} (B_full={B_full}, T_frames={T_frames})."
                                            )

                                        # reshape [B_full * T, T_env] -> [B_full, T, T_env]
                                        env_reshaped = log_env_full.view(B_full, T_frames, T_env)

                                        # Aggregate per-slice envelopes into a single envelope per RIR.
                                        # You can also use .mean(dim=1) or [:,0,:]; mean is safer.
                                        pred_env_for_loss = env_reshaped.mean(dim=1)   # [B_full, T_env]

                                    else:
                                        raise RuntimeError(
                                            "env_rms: mismatch between envelope batch (B_env="
                                            f"{B_env}) and full STFT batch (B_full={B_full}). "
                                            "Expected either per-RIR [B_full, T_env] or per-slice "
                                            "[B_full * T_frames, T_env] with contiguous slices per RIR."
                                        )

                                    env_term = self._env_rms_loss(
                                        pred_env_for_loss, gt_full, loss_type=self.env_rms_loss_type
                                    )
                                    parts["env_rms"] = env_term
                                    total = total + self.w_env_rms * env_term

                            else:
                                # batch not divisible by T -> cannot pack full RIRs
                                if (self.w_env_rms > 0.0) or (self.lambda_edc > 0.0):
                                    raise RuntimeError(
                                        "[warn] cannot pack slices (batch not multiple of T)."
                                        + " EDC/env_rms require full RIRs per batch when using slice mode."
                                    )
                    else:
                        # full batches: pred and gt already [B, C, F, T]
                        if self.lambda_edc > 0.0:
                            edc_term = self._edc_loss(pred, gt, valid_mask=None)
                            total = total + self.lambda_edc * edc_term
                            edc_val = edc_term.detach()

                        if self.w_env_rms > 0.0:
                            dbg = getattr(self.model, "debug_outputs", None)
                            log_env_full = None
                            if isinstance(dbg, dict):
                                log_env_full = dbg.get("log_env_full", None)

                            if log_env_full is None:
                                raise RuntimeError(
                                    "env_rms > 0 requires model.debug_outputs['log_env_full'] "
                                    "to be populated by the envelope head."
                                )

                            env_term = self._env_rms_loss(
                                log_env_full, gt, loss_type=self.env_rms_loss_type
                            )
                            parts["env_rms"] = env_term
                            total = total + self.w_env_rms * env_term
                            
                # ---- finalize loss and backward ----
                total_unscaled = total.detach()
                total = total * self.loss_factor  # APPLY GLOBAL SCALE
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()

                grad_norm_preclip = None
                if self.grad_clip_enabled:
                    grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_max_norm,
                        norm_type=self.grad_clip_norm_type,
                    )
                
                # ================================
                #   HARD DEBUG: LOSS WENT NaN
                # ================================
                if not torch.isfinite(total):
                    print("\n" + "="*80)
                    print("🔥🔥🔥  TRAIN LOSS BECAME NON-FINITE  🔥🔥🔥")
                    print("="*80)

                    # ---- basic checks ----
                    print("pred finite:", torch.isfinite(pred).all().item())
                    print("gt   finite:", torch.isfinite(gt).all().item())

                    # ---- ranges ----
                    try:
                        print("pred range:", pred.min().item(), pred.max().item())
                    except Exception:
                        print("pred range: FAILED")

                    try:
                        print("gt range:", gt.min().item(), gt.max().item())
                    except Exception:
                        print("gt range: FAILED")

                    # ---- show the batch IDs ----
                    if isinstance(batch, dict) and "id" in batch:
                        print("batch ids:", batch["id"])
                    else:
                        print("batch has no 'id' field")

                    # ---- check STFT and WAV inputs ----
                    if "stft" in batch:
                        st = batch["stft"]
                        print("batch stft finite:", torch.isfinite(st).all().item())
                        try:
                            print("batch stft range:", st.min().item(), st.max().item())
                        except Exception:
                            print("batch stft range: FAILED")

                    if "wav" in batch:
                        w = batch["wav"]
                        print("batch wav finite:", torch.isfinite(w).all().item())
                        try:
                            print("batch wav range:", w.min().item(), w.max().item())
                        except Exception:
                            print("batch wav range: FAILED")

                    # ---- print loss parts ----
                    print("Loss parts:")
                    for k, v in (parts or {}).items():
                        if torch.is_tensor(v):
                            print(f"  {k}: finite={torch.isfinite(v).all().item()}  value={v.item()}")
                        else:
                            print(f"  {k}: {v}")

                    # ---- check gradients before backward ----
                    print("\nChecking model activations:")
                    for name, p in self.model.named_parameters():
                        if p.requires_grad and torch.isfinite(p.data).all() is False:
                            print(f"PARAM DATA NOT FINITE → {name}")
                        if p.grad is not None and torch.isfinite(p.grad).all() is False:
                            print(f"PARAM GRAD NOT FINITE → {name}")
                            break

                    print("\n!!! ABORTING TRAINING (NaN detected) !!!")
                    print("="*80)
                    raise RuntimeError("Stopping due to NaN loss.")

                # ---- NEW: gradient norms before step (every self.log_every) ----
                if wandb_run is not None and (step % self.log_every == 0) and self.rag_module is not None:
                    # ---- single-pass module attribution via id(p) ----
                    enc_ids = {id(p) for p in self.model.encoder.parameters()}
                    dec_ids = {id(p) for p in self.model.decoder.parameters()}
                    gen_ids = {id(p) for p in self.model.rag_gen.parameters()} if hasattr(self.model, "rag_gen") else set()

                    tot_sq = torch.zeros((), device="cuda", dtype=torch.float32); tot_cnt = 0
                    enc_sq = torch.zeros((), device="cuda", dtype=torch.float32); enc_cnt = 0
                    dec_sq = torch.zeros((), device="cuda", dtype=torch.float32); dec_cnt = 0
                    gen_sq = torch.zeros((), device="cuda", dtype=torch.float32); gen_cnt = 0

                    for p in self.model.parameters():
                        if p.grad is None: 
                            continue
                        g = p.grad.detach().float()
                        s = (g * g).sum()
                        n = g.numel()
                        tot_sq += s; tot_cnt += n
                        pid = id(p)
                        if pid in enc_ids: enc_sq += s; enc_cnt += n
                        elif pid in dec_ids: dec_sq += s; dec_cnt += n
                        elif pid in gen_ids: gen_sq += s; gen_cnt += n

                    eps = 1e-12
                    tot_l2  = torch.sqrt(tot_sq + eps)
                    enc_l2  = torch.sqrt(enc_sq + eps)
                    dec_l2  = torch.sqrt(dec_sq + eps)
                    gen_l2  = torch.sqrt(gen_sq + eps)
                    tot_rms = torch.sqrt(tot_sq / max(tot_cnt, 1) + eps)
                    enc_rms = torch.sqrt(enc_sq / max(enc_cnt, 1) + eps)
                    dec_rms = torch.sqrt(dec_sq / max(dec_cnt, 1) + eps)
                    gen_rms = torch.sqrt(gen_sq / max(gen_cnt, 1) + eps)

                    enc_share = (enc_l2 / (tot_l2 + eps)).item()
                    dec_share = (dec_l2 / (tot_l2 + eps)).item()
                    gen_share = (gen_l2 / (tot_l2 + eps)).item()

                    wandb_run.log({
                        "grads/encoder_rms": float(enc_rms.item()),
                        "grads/decoder_rms": float(dec_rms.item()),
                        "grads/generator_rms": float(gen_rms.item()),
                        "grads/total_rms":    float(tot_rms.item()),
                        "grads/encoder_share": enc_share,
                        "grads/decoder_share": dec_share,
                        "grads/generator_share": gen_share,
                        "grads/log10_total_rms": float(torch.log10(tot_rms + eps).item()),
                    })

                
                self.optimizer.step()

                # ---- richer logging ----
                parts_log = self._format_parts(parts, edc_val)
                if wandb_run:
                    wandb_log = {
                        "train/loss": float(total.item()),
                        "train/lr":   float(lr),
                    }
                    if grad_norm_preclip is not None:
                        wandb_log["train/grad_norm_preclip"] = float(grad_norm_preclip.item())
                    for k, v in parts_log.items():
                        wandb_log[f"train/{k}"] = v

                    # ----- per-term shares in TOTAL UN-SCALED loss -----
                    eps = 1e-12
                    denom = float(total_unscaled.item()) + eps

                    # EDC share (if present)
                    if edc_val is not None and self.lambda_edc != 0.0:
                        edc_w = float(self.lambda_edc)
                        edc_share = float((edc_w * edc_val.item()) / denom)
                        wandb_log["train/edc_loss"]  = float(edc_val.item())
                        wandb_log["train/edc_share"] = edc_share

                    # Envelope RMS share
                    if ("env_rms" in parts) and (self.w_env_rms != 0.0):
                        env_term  = float(parts["env_rms"].item())
                        env_share = float((self.w_env_rms * env_term) / denom)
                        wandb_log["train/env_rms_share"] = env_share

                    # Residual L2 regularizer share
                    if ("res_l2" in parts) and (self.w_res_l2 != 0.0):
                        res_term  = float(parts["res_l2"].item())
                        res_share = float((self.w_res_l2 * res_term) / denom)
                        wandb_log["train/res_l2_share"] = res_share

                    wandb_run.log(wandb_log)
                # ---- progress bar update ----
                pbar.set_postfix({"loss": float(total.item())})
                pbar.update(1)
                step += 1
            pbar.close()

            # ---- eval each epoch ----
            eval_metrics = self.eval(val_loader)
            if wandb_run:
                wandb_run.log({f"eval/{k}": float(v) for k, v in eval_metrics.items()})

            # ---- periodic checkpoint ----
            if save_dir and (epoch % save_every == 0):
                ck = os.path.join(save_dir, f"epoch_{epoch}.pt")
                self.save_model(ck, extra={"epoch": epoch})

    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        # val loader MUST be dataset_mode='full'
        e_sum, n = None, 0
        for batch in tqdm(val_loader, desc="[eval]", leave=False):
            gt = batch["stft"].to(self.device).float()  # [B,1,F,60] or [B,2,F,60]
            pred = self._forward_batch(batch, mode_full="full")
            m = self.evaluator.evaluate(pred, gt)
            if e_sum is None:
                e_sum = {k: float(v) for k, v in m.items()}
            else:
                for k, v in m.items():
                    e_sum[k] += float(v)
            n += 1
        self.model.train()
        return {k: (v / max(1, n)) for k, v in e_sum.items()}
