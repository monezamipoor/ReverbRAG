# trainer.py
import math
from copy import deepcopy
import os
from typing import Dict, Any, Optional

from decay_features import build_ref_decay_features_bank
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
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
        self.baseline = baseline.lower()
        self.run_cfg = run_cfg
        self.scaler = GradScaler(enabled=True)

        # STFT/ISTFT & Evaluator (parameters taken “from main.py”)
        fs = int(run_cfg.get("sample_rate", 48000))
        stftp = stft_params or fs_to_stft_params(fs)
        self.istft = GriffinLim(
            n_fft=(stftp["N_freq"] - 1) * 2,
            win_length=stftp["win_len"],
            hop_length=stftp["hop_len"],
            power=1,
        ).to(device)
        self.evaluator = UnifiedEvaluator(fs=fs, edc_bins=60, edc_dist="l1")

        # ---- ReverbRAG reference bank & decay features (CPU, once) ----
        self.ref_bank = ref_bank  # [R,1,F,60] log-mags (usually CPU)
        self.ref_bank_ids = ref_bank_ids
        hop_ms = (float(stftp["hop_len"]) / float(fs)) * 1000.0
        # Handle configs where `reverbrag` is a bool (e.g., `reverbrag: true`) or a dict (`{ num_bands: 32 }`)
        rv_cfg = run_cfg.get("reverbrag", None)
        if isinstance(rv_cfg, dict):
            num_bands = int(rv_cfg.get("num_bands", 32))
        else:
            # bool/None/anything else -> enabled with default bands
            num_bands = 32

        if ref_feats is not None:
            self.ref_feats = ref_feats.to(torch.float32)  # [R,B,4], float32 normalized
        elif self.ref_bank is not None:
            with torch.no_grad():
                # ensure CPU numpy for deterministic Numba; compute in float64 then return float32
                bank_np = self.ref_bank.cpu().numpy()
                feats_np, stats = build_ref_decay_features_bank(bank_np, num_bands=num_bands, hop_ms=hop_ms)
                # keep a copy on CPU; we’ll gather to device per batch
                self.ref_feats = torch.from_numpy(feats_np)  # [R, B, 4], float32
                # (optional) you can persist stats if you want; omitted here
        else:
            self.ref_feats = None

        # Loss (different per baseline)
        if self.baseline == "neraf":
            self.loss_fn = STFTLoss(loss_type='mse')
        else:  # "avnerf": keep it simple & faithful to their MSE-on-mags spirit
            self.loss_fn = None  # using a compact bi/mono-agnostic MSE on log-mags
        # EDC batching requirement (full-sequence batches only)
        self.lambda_edc = float(run_cfg.get("edc_loss", 0.0))  # e.g., 0.2
        self.use_edc_full = bool(run_cfg.get("edc_full", False))
        self._warned_slice_edc = False

        # Optimizer (abstract; two presets)
        if self.baseline == "avnerf":
            # 1:1 spirit of avnerf-trainer: Adam + warmup/exp decay inside train step
            lr = float(run_cfg.get("lr", 1e-3))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self._av_lr0 = lr
        else:
            # NeRAF (ICLR ’25 impl details): Adam(β1=0.9, β2=0.999, eps=1e-15), lr 1e-4→1e-8 exp
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1e-4,
                betas=(0.9, 0.999), eps=1e-15
            )
            # compute per-step decay factor to reach 1e-8 by last step
            self._neraf_lr_hi, self._neraf_lr_lo = 1e-4, 1e-8
            self._neraf_total_steps = None  # set on first train()

        self.visual_feat_builder = visual_feat_builder
        if ref_bank is None or ref_bank.numel() == 0:
            self.ref_bank = None         # disables RAG path
        else:
            self.ref_bank = ref_bank     # [R, 1, F, 60] (log-mag)
        self.ref_bank_ids = ref_bank_ids or []

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

    def _forward_batch(self, batch: Dict[str, torch.Tensor], dataset_mode: str):
        full = (dataset_mode == "full")
        mic = batch["receiver_pos"].to(self.device).float()
        src = batch["source_pos"].to(self.device).float()
        head = batch["orientation"].to(self.device).float()
        t_idx = self._make_t_idx(batch, self.device, full)
        B = mic.shape[0]
        vfeat = self.visual_feat_builder(batch, B).to(self.device).float()

        # ---- ReverbRAG refs for this batch (optional) ----
        ref_idx = batch.get("ref_indices", None)
        refs_logmag = refs_mask = refs_feats = None
        if ref_idx is not None:
            refs_logmag, refs_mask = self._gather_refs(ref_idx)
            refs_feats = self._gather_ref_feats(ref_idx)

        with autocast(True):
            pred = self.model(
                mic_xyz=mic, src_xyz=src, head_dir=head, t_idx=t_idx,
                visual_feat=vfeat,
                refs_logmag=refs_logmag, refs_mask=refs_mask,    # <— existing
                refs_feats=refs_feats,                            # <— NEW
            )
        return pred

    # ---- compact EDC loss (computed on full STFT batches only) ----
    def _edc_loss(self, pred_log: torch.Tensor, gt_log: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None,
                p: int = 1, eps: float = 1e-12) -> torch.Tensor:
        """
        Differentiable EDC loss on the 60-frame STFT grid.
        pred_log, gt_log: [B, C, F, 60] or [B, 1, F, 60] log-magnitude.
        Steps:
        - mag = exp(log) - 1e-3, clamp >= 0
        - per-frame energy: sum over F (and C if present)
        - Schroeder: reverse cumsum over time
        - dB, per-curve shape normalization (0 dB at t=0, unit std over valid frames)
        - L1 (p=1) or root-L2 (p=2) over time
        """
        assert pred_log.dim() == 4 and gt_log.dim() == 4 and pred_log.shape[-1] == 60, \
            "EDC loss expects full 60-frame STFT sequences."

        # magnitudes
        pred_mag = (pred_log.exp() - 1e-3).clamp_min(0.0)
        gt_mag   = (gt_log.exp()   - 1e-3).clamp_min(0.0)

        # energy over frequency (and channel)
        E_pred = (pred_mag ** 2).sum(dim=-2)  # [B,C,60]
        E_gt   = (gt_mag   ** 2).sum(dim=-2)
        if E_pred.dim() == 3:
            E_pred = E_pred.sum(dim=-3)      # [B,60]
            E_gt   = E_gt.sum(dim=-3)        # [B,60]

        # Schroeder (reverse cumsum)
        S_pred = torch.flip(torch.cumsum(torch.flip(E_pred, dims=[-1]), dim=-1), dims=[-1])
        S_gt   = torch.flip(torch.cumsum(torch.flip(E_gt,   dims=[-1]), dim=-1), dims=[-1])

        # dB
        s_pred = 10.0 * torch.log10(S_pred + eps)
        s_gt   = 10.0 * torch.log10(S_gt   + eps)

        # shape normalization per-curve: zero at t0, unit std on valid frames
        B, T = s_pred.shape
        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=s_pred.device)

        s_pred = s_pred - s_pred[:, :1]
        s_gt   = s_gt   - s_gt[:, :1]

        def _std_over_valid(x, m):
            cnt = m.sum(dim=-1, keepdim=True).clamp_min(1)
            mu  = (x * m).sum(dim=-1, keepdim=True) / cnt
            var = ((x - mu) ** 2 * m).sum(dim=-1, keepdim=True) / cnt
            return torch.sqrt(var + eps)

        std_p = _std_over_valid(s_pred, valid_mask)
        std_g = _std_over_valid(s_gt,   valid_mask)
        s_pred = s_pred / std_p
        s_gt   = s_gt   / std_g

        # distance
        diff = (s_pred - s_gt) * valid_mask
        if p == 1:
            per_t = diff.abs()
            per_s = per_t.sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1)
            return per_s.mean()
        else:  # p == 2
            per_t = diff.pow(2)
            per_s = per_t.sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1)
            return torch.sqrt(per_s + eps).mean()


    def _loss(self, pred_log, gt_log, dataset_mode: str):
        pred_log = pred_log.float(); gt_log = gt_log.float()
        if self.baseline == "neraf":
            parts = self.loss_fn(pred_log, gt_log)  # dict: audio_sc_loss, audio_mag_loss
            total = 0.1 * parts["audio_sc_loss"] + 1.0 * parts["audio_mag_loss"]
            edc_val = None
            if self.lambda_edc > 0.0:
                if self.use_edc_full and dataset_mode == "full":
                    edc_term = self._edc_loss(pred_log, gt_log)
                    total = total + self.lambda_edc * edc_term
                    edc_val = edc_term.detach()
                elif self.use_edc_full and dataset_mode != "full" and not self._warned_slice_edc:
                    print("[warn] edc_full=True but dataset_mode!='full' → skipping EDC loss on slices.")
                    self._warned_slice_edc = True
            return total, parts, edc_val
        # AV-NeRF fallback: single MSE on log-mags
        mse = F.mse_loss(pred_log, gt_log)
        return mse, {"mse": mse.detach()}, None

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
                lr = self._av_lr0 * (step_idx / max(1, warmup))
            else:
                total = max_epoch * steps_per_epoch
                lr = self._av_lr0 * (0.1 ** (2 * (step_idx - warmup) / max(1, (total - warmup))))
            self.optimizer.param_groups[0]["lr"] = lr
            return lr
        # NeRAF: per-step exponential from 1e-4 to 1e-8
        if self._neraf_total_steps is None:
            self._neraf_total_steps = max_epoch * steps_per_epoch
        t = step_idx / max(1, self._neraf_total_steps)
        lr = self._neraf_lr_hi * ((self._neraf_lr_lo / self._neraf_lr_hi) ** t)
        self.optimizer.param_groups[0]["lr"] = lr
        return lr

    def _pack_slices_to_full(x: torch.Tensor, T: int) -> torch.Tensor:
        """
        x: [B*T, C, F, 1]  ->  [B, C, F, T]
        Requires that the dataloader groups slices per SID contiguously in time
        (EDCFullBatchSampler does this).
        """
        B = x.shape[0] // T
        return (
            x.view(B, T, x.shape[1], x.shape[2], x.shape[3])  # [B,T,C,F,1]
            .permute(0, 2, 3, 1, 4)                          # [B,C,F,T,1]
            .squeeze(-1)                                     # [B,C,F,T]
            .contiguous()
        )

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

    # ------------------------- API -------------------------
    def train(
        self, train_loader, val_loader, dataset_mode: str, epochs: int, wandb_run=None,
        save_dir: str = None, save_every: int = 10, cfg_copy_path: str = None,
        resume_ckpt: str = None, resume_load_optimizer: bool = False, resume_lr_override=None,
    ):
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
            pbar = tqdm(total=len(train_loader), desc=f"[train] epoch {epoch}", leave=False)
            for batch in train_loader:
                gt = (batch["stft"] if dataset_mode == "full" else batch["stft_slice"]).to(self.device).float()
                if dataset_mode != "full":
                    gt = gt.unsqueeze(-1)  # [B*T, 1, F, 1] logically

                pred = self._forward_batch(batch, dataset_mode)
                if pred.dtype != gt.dtype:
                    pred = pred.to(gt.dtype)

                # --- normal spectral parts ---
                total, parts, edc_val = self._loss(pred, gt, dataset_mode)

                # inside train() loop, after total, parts, edc_val = self._loss(...)
                if self.lambda_edc > 0.0 and getattr(train_loader.sampler, "__class__", None).__name__ != "EDCFullBatchSampler":
                    print("[warn] EDC loss expects EDCFullBatchSampler (contiguous T slices per SID).")
                if self.lambda_edc > 0.0:
                    T = getattr(train_loader.dataset, "max_frames", None)  # expect 60
                    if T is not None and (gt.shape[0] % T == 0):
                        pred_full = self._pack_slices_to_full(pred, T)
                        gt_full   = self._pack_slices_to_full(gt,   T)
                        edc_term  = self._edc_loss(pred_full, gt_full, valid_mask=None, p=1)
                        total = total + self.lambda_edc * edc_term
                        edc_val = edc_term.detach()
                    else:
                        if not self._warned_slice_edc:
                            print("[warn] cannot pack slices into full T=60 — skipping EDC term this step.")
                            self._warned_slice_edc = True

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                lr = self._update_lr(step, len(train_loader), epochs, self.baseline)

                # ---- richer logging & printing ----
                parts_log = self._format_parts(parts, edc_val)
                if wandb_run:
                    wandb_log = {"train/loss": float(total.item()), "train/lr": float(lr)}
                    for k, v in parts_log.items():
                        wandb_log[f"train/{k}"] = v
                    if edc_val is not None:
                        wandb_log["train/edc_loss"] = float(edc_val)
                    wandb_run.log(wandb_log)
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
            pred = self._forward_batch(batch, dataset_mode="full")
            m = self.evaluator.evaluate(pred, gt)
            if e_sum is None:
                e_sum = {k: float(v) for k, v in m.items()}
            else:
                for k, v in m.items():
                    e_sum[k] += float(v)
            n += 1
        self.model.train()
        return {k: (v / max(1, n)) for k, v in e_sum.items()}

    def save_model(self, path: str, extra: Optional[Dict[str, Any]] = None):
        state = {
            "model": self.model.state_dict(),
            "baseline": self.baseline,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)