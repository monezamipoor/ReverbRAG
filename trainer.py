# trainer.py
import math
from copy import deepcopy
from typing import Dict, Any, Optional

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
        self.evaluator = UnifiedEvaluator(fs=fs, edc_bins=60, edc_dist="l2")

        # Loss (different per baseline)
        if self.baseline == "neraf":
            self.loss_fn = STFTLoss(loss_type='mse')
            self.lambda_edc = float(run_cfg.get("edc_loss", 0.0))  # e.g., 0.2
        else:  # "avnerf": keep it simple & faithful to their MSE-on-mags spirit
            self.loss_fn = None  # using a compact bi/mono-agnostic MSE on log-mags

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

    def _forward_batch(self, batch: Dict[str, torch.Tensor], dataset_mode: str):
        full = (dataset_mode == "full")
        # Common pose tensors
        mic = batch["receiver_pos"].to(self.device).float()
        src = batch["source_pos"].to(self.device).float()
        head = batch["orientation"].to(self.device).float()
        t_idx = self._make_t_idx(batch, self.device, full)
        # Visual features (unified)
        B = mic.shape[0]
        vfeat = self.visual_feat_builder(batch, B).to(self.device).float()
        # Forward
        with autocast(True):
            pred = self.model(mic_xyz=mic, src_xyz=src, head_dir=head, t_idx=t_idx, visual_feat=vfeat)
        return pred  # [B,C,F,T or 1]

    # ---- compact EDC loss (computed on full STFT batches only) ----
    def _edc_loss(self, pred_log: torch.Tensor, gt_log: torch.Tensor):
        # ISTFT → wav → EDC inside evaluator helpers
        # We use evaluator’s internal GriffinLim (32 iters). Here we re-use self.istft (fast).
        with torch.no_grad():
            gt_mag = torch.exp(gt_log) - 1e-3
            wav_gt = self.istft(gt_mag.squeeze(1))  # [B, T_wav], mono or each-ch collapsed
        pred_mag = torch.exp(pred_log) - 1e-3
        wav_pr = self.istft(pred_mag.squeeze(1))
        # EDC curve L2 on dB-normalized Schroeder (batch mean)
        # Use evaluator’s public API by “evaluating” and just reading the 'edc' scalar.
        met = self.evaluator.evaluate(pred_log, gt_log)
        return torch.as_tensor(met["edc"], device=pred_log.device, dtype=pred_log.dtype)

    def _loss(self, pred_log: torch.Tensor, gt_log: torch.Tensor, dataset_mode: str):
        # run loss in fp32 to avoid AMP dtype clashes
        pred_log = pred_log.float()
        gt_log   = gt_log.float()
        if self.baseline == "neraf":
            parts = self.loss_fn(pred_log, gt_log)
            loss = parts["audio_sc_loss"] + parts["audio_mag_loss"]
            if self.lambda_edc > 0.0 and dataset_mode == "full":
                loss = loss + self.lambda_edc * self._edc_loss(pred_log, gt_log)
            return loss
        return F.mse_loss(pred_log, gt_log)


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

    # ------------------------- API -------------------------
    def train(self, train_loader, val_loader, dataset_mode: str, epochs: int, wandb_run=None):
        self.model.train()
        step = 0
        for epoch in range(1, epochs + 1):
            pbar = tqdm(total=len(train_loader), desc=f"[train] epoch {epoch}", leave=False)
            for batch in train_loader:
                # bring GT STFT log to device depending on mode
                gt = (batch["stft"] if dataset_mode == "full" else batch["stft_slice"]).to(self.device).float()
                if dataset_mode != "full":  # [B,1,F] -> [B,1,F,1]
                    gt = gt.unsqueeze(-1)

                pred = self._forward_batch(batch, dataset_mode)
                if pred.dtype != gt.dtype:
                    pred = pred.to(gt.dtype)
                loss = self._loss(pred, gt, dataset_mode)
                loss = self._loss(pred, gt, dataset_mode)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                lr = self._update_lr(step, len(train_loader), epochs, self.baseline)

                if wandb_run:
                    wandb_run.log({"train/loss": float(loss.item()), "train/lr": float(lr)})
                pbar.set_postfix(loss=float(loss.item()))
                pbar.update(1)
                step += 1
            pbar.close()

            # eval every epoch
            eval_metrics = self.eval(val_loader)
            if wandb_run:
                wandb_run.log({f"eval/{k}": float(v) for k, v in eval_metrics.items()})

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