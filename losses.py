import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConvergenceLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-12)


class LogSTFTMagnitudeLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, x_log, y_log):
        if self.loss_type == "mse":
            return F.mse_loss(x_log, y_log)
        return F.l1_loss(x_log, y_log)


class STFTLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.sc = SpectralConvergenceLoss()
        self.lm = LogSTFTMagnitudeLoss(loss_type)

    def forward(self, x_log, y_log):
        x_mag = torch.exp(x_log) - 1e-3
        y_mag = torch.exp(y_log) - 1e-3
        return {
            "audio_sc_loss": self.sc(x_mag, y_mag),
            "audio_mag_loss": self.lm(x_log, y_log),
        }


class ReverbRAGLosses:
    """Holds all training-loss logic (STFT, EDC, envelope, MR-STFT)."""

    def __init__(
        self,
        loss_cfg: Dict[str, Any],
        run_cfg: Dict[str, Any],
        fs: int,
        device: torch.device,
        base_n_fft: int,
        base_win_length: int,
        base_hop_length: int,
    ):
        self.fs = int(fs)
        self.device = device
        self._base_n_fft = int(base_n_fft)
        self._base_win_length = int(base_win_length)
        self._base_hop_length = int(base_hop_length)

        # STFT core losses
        self.mag_loss_type = str(loss_cfg.get("mag_loss_type", "mse")).lower()
        self.loss_fn = STFTLoss(loss_type=self.mag_loss_type)
        self.w_sc = float(loss_cfg.get("sc", 0.0))
        self.w_mag = float(loss_cfg.get("mag", 0.0))
        self.w_mse = float(loss_cfg.get("mse", 0.0))

        # Envelope/residual
        self.w_env_rms = float(loss_cfg.get("env_rms", 0.0))
        self.env_rms_loss_type = str(loss_cfg.get("env_rms_type", "l1"))
        self.w_res_l2 = float(loss_cfg.get("res_l2", 0.0))

        # Time-weighting over STFT frame axis
        tw_cfg = loss_cfg.get("time_weight", {}) or {}
        self.time_weight_enabled = bool(tw_cfg.get("enabled", False))
        self.tw_early_frames = int(tw_cfg.get("early_frames", 12))
        self.tw_early_gain = float(tw_cfg.get("early_gain", 2.0))
        self.tw_normalize_mean = bool(tw_cfg.get("normalize_mean", True))
        self.tw_apply_to_mag = bool(tw_cfg.get("apply_to_mag", True))
        self.tw_apply_to_mse = bool(tw_cfg.get("apply_to_mse", True))
        self.tw_max_frames = int(tw_cfg.get("max_frames", 60))

        # Multi-resolution waveform STFT loss
        mr_cfg = loss_cfg.get("mrstft", {}) or {}
        self.mrstft_enabled = bool(mr_cfg.get("enabled", False))
        self.mrstft_weight = float(mr_cfg.get("weight", 0.0))
        self.mrstft_loss_type = str(mr_cfg.get("loss_type", "l1")).lower()
        self.mrstft_freq_focus = str(mr_cfg.get("freq_focus", "none")).lower()
        self.mrstft_low_hz = float(mr_cfg.get("low_hz", 300.0))
        self.mrstft_scales = []
        self._mr_windows = []
        if self.mrstft_enabled and self.mrstft_weight > 0.0:
            scales = mr_cfg.get("scales", None)
            if not scales:
                scales = [
                    {"n_fft": 512, "win_length": 256, "hop_length": 128, "w": 1.0},
                    {"n_fft": 1024, "win_length": 512, "hop_length": 256, "w": 1.0},
                    {"n_fft": 2048, "win_length": 1024, "hop_length": 512, "w": 1.2},
                ]
            for s in scales:
                n_fft = int(s["n_fft"])
                win_length = int(s.get("win_length", n_fft // 2))
                hop_length = int(s.get("hop_length", max(1, win_length // 2)))
                w = float(s.get("w", 1.0))
                if win_length > n_fft:
                    raise ValueError(f"mrstft scale invalid: win_length({win_length}) > n_fft({n_fft})")
                self.mrstft_scales.append(
                    {"n_fft": n_fft, "win_length": win_length, "hop_length": hop_length, "w": w}
                )
                self._mr_windows.append(
                    torch.hann_window(win_length, device=self.device, dtype=torch.float32)
                )

        # EDC + optional banding
        self.lambda_edc = float(loss_cfg.get("edc", run_cfg.get("edc_loss", 0.0)))
        edc_band_cfg = loss_cfg.get("edc_band", {}) or {}
        self.edc_band_enabled = bool(edc_band_cfg.get("enabled", False))
        self.edc_band_type = str(edc_band_cfg.get("type", "8")).lower()
        self._edc_band_cache = {}

    def _make_time_weights(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        w = torch.ones(T, device=device, dtype=dtype)
        if T <= 0:
            return w
        ef = max(0, min(int(self.tw_early_frames), int(T)))
        if ef > 0 and self.tw_early_gain != 1.0:
            w[:ef] = float(self.tw_early_gain)
        if self.tw_normalize_mean:
            w = w / w.mean().clamp_min(1e-12)
        return w

    @staticmethod
    def _weighted_mean(err: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w_exp = w.expand_as(err)
        return (err * w_exp).sum() / w_exp.sum().clamp_min(1e-12)

    def base_loss(
        self,
        pred_log: torch.Tensor,
        gt_log: torch.Tensor,
        slice_t: Optional[torch.Tensor] = None,
        is_slice: bool = False,
        model_debug_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], None]:
        pred_log = pred_log.float()
        gt_log = gt_log.float()

        parts: Dict[str, torch.Tensor] = {}
        total = pred_log.new_zeros(())

        time_w = None
        if self.time_weight_enabled:
            if is_slice:
                if slice_t is None:
                    raise RuntimeError("time_weight.enabled=True in slice mode but slice_t is missing.")
                idx = slice_t.to(pred_log.device).long().view(-1)
                base_w = self._make_time_weights(
                    T=int(self.tw_max_frames), device=pred_log.device, dtype=pred_log.dtype
                )
                idx = idx.clamp(0, base_w.shape[0] - 1)
                time_w = base_w.index_select(0, idx).view(-1, 1, 1, 1)
            else:
                T_cur = int(pred_log.shape[-1])
                time_w = self._make_time_weights(
                    T=T_cur, device=pred_log.device, dtype=pred_log.dtype
                ).view(1, 1, 1, T_cur)

        if self.w_sc != 0.0:
            x_mag = torch.exp(pred_log) - 1e-3
            y_mag = torch.exp(gt_log) - 1e-3
            sc_val = self.loss_fn.sc(x_mag, y_mag)
            parts["audio_sc_loss"] = sc_val
            total = total + self.w_sc * sc_val

        if self.w_mag != 0.0:
            mag_err = (pred_log - gt_log).pow(2) if self.mag_loss_type == "mse" else (pred_log - gt_log).abs()
            mag_val = self._weighted_mean(mag_err, time_w) if (self.time_weight_enabled and self.tw_apply_to_mag) else mag_err.mean()
            parts["audio_mag_loss"] = mag_val
            total = total + self.w_mag * mag_val

        if self.w_mse != 0.0:
            mse_err = (pred_log - gt_log).pow(2)
            mse_val = self._weighted_mean(mse_err, time_w) if (self.time_weight_enabled and self.tw_apply_to_mse) else mse_err.mean()
            parts["mse"] = mse_val
            total = total + self.w_mse * mse_val

        if self.w_res_l2 != 0.0 and isinstance(model_debug_outputs, dict):
            r_log = model_debug_outputs.get("residual_log", None)
            if r_log is not None:
                res_l2 = (r_log ** 2).mean()
                parts["res_l2"] = res_l2
                total = total + self.w_res_l2 * res_l2

        return total, parts, None

    def env_rms_loss(
        self,
        pred_env_log: torch.Tensor,
        gt_log: torch.Tensor,
        loss_type: str = "l1",
    ) -> torch.Tensor:
        assert gt_log.shape[-1] == 60, "env_rms expects full GT STFT with T=60 frames"
        with torch.cuda.amp.autocast(enabled=False):
            gt_log32 = gt_log.float()
            mag_gt = (gt_log32.exp() - 1e-3).clamp_min(0.0)
            E_gt = (mag_gt ** 2).sum(dim=-2)
            if E_gt.dim() == 3:
                E_gt = E_gt.mean(dim=1)
            rms_gt = torch.sqrt(E_gt + 1e-8)
            log_rms_gt = torch.log(rms_gt + 1e-6)

        B_pred, T_env = pred_env_log.shape
        B_gt, T_gt = log_rms_gt.shape
        if T_gt != T_env:
            raise RuntimeError(
                f"env_rms loss: time length mismatch between pred_env_log (T_env={T_env}) "
                f"and GT log-RMS envelope (T={T_gt})."
            )
        if B_pred != B_gt:
            raise RuntimeError(
                f"env_rms loss: batch mismatch between pred_env_log (B={B_pred}) and GT log-RMS envelope (B={B_gt})."
            )
        if loss_type == "mse":
            return F.mse_loss(pred_env_log, log_rms_gt)
        if loss_type == "l1":
            return F.l1_loss(pred_env_log, log_rms_gt)
        raise ValueError(f"Unknown env_rms loss_type='{loss_type}'")

    def _build_edc_band_matrix(self, n_freq: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (int(n_freq), str(device), str(dtype), self.edc_band_type)
        if cache_key in self._edc_band_cache:
            return self._edc_band_cache[cache_key]

        fs = float(self.fs)
        freqs = torch.linspace(0.0, fs / 2.0, steps=n_freq, device=device, dtype=torch.float32)
        masks = []
        btype = self.edc_band_type

        if btype in ("8", "16"):
            n_band = int(btype)
            edges = torch.linspace(0.0, fs / 2.0, steps=n_band + 1, device=device, dtype=torch.float32)
            for i in range(n_band):
                lo, hi = edges[i], edges[i + 1]
                m = (freqs >= lo) & (freqs < hi) if i < n_band - 1 else (freqs >= lo) & (freqs <= hi)
                if bool(m.any()):
                    masks.append(m.to(torch.float32))
        else:
            if btype == "octave":
                centers = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
                ratio = math.sqrt(2.0)
            elif btype == "third_octave":
                centers = [
                    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
                    1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
                ]
                ratio = 2.0 ** (1.0 / 6.0)
            else:
                raise ValueError(
                    f"Unknown losses.edc_band.type='{self.edc_band_type}'. Supported: 8, 16, octave, third_octave."
                )
            for fc in centers:
                lo, hi = fc / ratio, fc * ratio
                if hi < 0.0 or lo > fs / 2.0:
                    continue
                m = (freqs >= lo) & (freqs < hi)
                if bool(m.any()):
                    masks.append(m.to(torch.float32))

        band_mat = torch.ones((1, n_freq), device=device, dtype=dtype) if len(masks) == 0 else torch.stack(masks, dim=0).to(device=device, dtype=dtype)
        self._edc_band_cache[cache_key] = band_mat
        return band_mat

    def edc_loss(
        self,
        pred_log: torch.Tensor,
        gt_log: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        assert pred_log.shape[-1] == 60
        with torch.cuda.amp.autocast(enabled=False):
            pred_log32 = pred_log.float()
            gt_log32 = gt_log.float()
            pred_mag = (pred_log32.exp() - 1e-3).clamp_min(0.0)
            gt_mag = (gt_log32.exp() - 1e-3).clamp_min(0.0)

            if pred_mag.dim() == 4:
                if self.edc_band_enabled:
                    band_mat = self._build_edc_band_matrix(
                        n_freq=int(pred_mag.shape[-2]), device=pred_mag.device, dtype=pred_mag.dtype
                    )
                    E_pred = torch.einsum("nf,bcft->bcnt", band_mat, pred_mag**2)
                    E_gt = torch.einsum("nf,bcft->bcnt", band_mat, gt_mag**2)
                else:
                    E_pred = (pred_mag**2).sum(dim=-2)
                    E_gt = (gt_mag**2).sum(dim=-2)
            else:
                if self.edc_band_enabled:
                    band_mat = self._build_edc_band_matrix(
                        n_freq=int(pred_mag.shape[-2]), device=pred_mag.device, dtype=pred_mag.dtype
                    )
                    E_pred = torch.einsum("nf,bft->bnt", band_mat, pred_mag**2)
                    E_gt = torch.einsum("nf,bft->bnt", band_mat, gt_mag**2)
                else:
                    E_pred = (pred_mag**2).sum(dim=-2)
                    E_gt = (gt_mag**2).sum(dim=-2)

            S_pred = torch.flip(torch.cumsum(torch.flip(E_pred, dims=[-1]), dim=-1), dims=[-1])
            S_gt = torch.flip(torch.cumsum(torch.flip(E_gt, dims=[-1]), dim=-1), dims=[-1])
            s_pred = torch.log10(S_pred.clamp_min(eps))
            s_gt = torch.log10(S_gt.clamp_min(eps))

            if valid_mask is not None:
                m = valid_mask.to(s_pred.dtype)
                while m.dim() < s_pred.dim():
                    m = m.unsqueeze(1)
            else:
                m = torch.ones_like(s_pred, dtype=s_pred.dtype)

            m = m.expand_as(s_pred)
            dims = tuple(range(1, s_pred.dim()))
            denom = m.sum(dim=dims).clamp_min(1.0)
            per_b = ((s_pred - s_gt).abs() * m).sum(dim=dims) / denom
            return per_b.mean()

    @staticmethod
    def _match_stft_timebins(x: torch.Tensor, T_target: int) -> torch.Tensor:
        T_cur = int(x.shape[-1])
        if T_cur == T_target:
            return x
        if T_cur > T_target:
            return x[..., :T_target]
        pad = T_target - T_cur
        last = x[..., -1:].expand(*x.shape[:-1], pad)
        return torch.cat([x, last], dim=-1)

    def reconstruct_wav_with_gt_phase(self, pred_log: torch.Tensor, gt_wav: torch.Tensor) -> torch.Tensor:
        pred_log = pred_log.float()
        gt_wav = gt_wav.float()
        B, C, F, T = pred_log.shape
        B_w, Cg, Tw = gt_wav.shape
        if B_w != B:
            raise RuntimeError(f"mrstft: batch mismatch pred(B={B}) vs gt_wav(B={B_w}).")
        if Cg == 1 and C > 1:
            gt_wav = gt_wav.expand(B, C, Tw)
            Cg = C
        if Cg != C:
            raise RuntimeError(f"mrstft: channel mismatch pred(C={C}) vs gt_wav(C={Cg}).")

        gt_flat = gt_wav.reshape(B * C, Tw)
        base_window = torch.hann_window(self._base_win_length, device=gt_wav.device, dtype=gt_wav.dtype)
        gt_c = torch.stft(
            gt_flat,
            n_fft=self._base_n_fft,
            hop_length=self._base_hop_length,
            win_length=self._base_win_length,
            window=base_window,
            center=True,
            return_complex=True,
        )
        gt_c = gt_c.view(B, C, gt_c.shape[-2], gt_c.shape[-1])
        gt_c = self._match_stft_timebins(gt_c, T_target=T)
        if gt_c.shape[-2] != F:
            raise RuntimeError(f"mrstft: freq mismatch pred(F={F}) vs gt_phase(F={gt_c.shape[-2]}).")

        phase_unit = gt_c / gt_c.abs().clamp_min(1e-8)
        pred_mag = (pred_log.exp() - 1e-3).clamp_min(0.0)
        pred_c = pred_mag.to(phase_unit.dtype) * phase_unit
        pred_c_flat = pred_c.reshape(B * C, F, T)
        wav_pred_flat = torch.istft(
            pred_c_flat,
            n_fft=self._base_n_fft,
            hop_length=self._base_hop_length,
            win_length=self._base_win_length,
            window=base_window,
            center=True,
            length=Tw,
        )
        return wav_pred_flat.view(B, C, Tw)

    def mrstft_loss(self, wav_pred: torch.Tensor, wav_gt: torch.Tensor) -> torch.Tensor:
        if wav_pred.shape != wav_gt.shape:
            raise RuntimeError(f"mrstft: wav shape mismatch {tuple(wav_pred.shape)} vs {tuple(wav_gt.shape)}")
        B, C, Tw = wav_pred.shape
        wp = wav_pred.reshape(B * C, Tw).float()
        wg = wav_gt.reshape(B * C, Tw).float()

        total = wp.new_zeros(())
        wsum = 0.0
        for sc, win in zip(self.mrstft_scales, self._mr_windows):
            Xp = torch.stft(
                wp,
                n_fft=sc["n_fft"],
                hop_length=sc["hop_length"],
                win_length=sc["win_length"],
                window=win,
                center=True,
                return_complex=True,
            )
            Xg = torch.stft(
                wg,
                n_fft=sc["n_fft"],
                hop_length=sc["hop_length"],
                win_length=sc["win_length"],
                window=win,
                center=True,
                return_complex=True,
            )
            mp = Xp.abs().clamp_min(1e-8)
            mg = Xg.abs().clamp_min(1e-8)
            lp = torch.log(mp + 1e-6)
            lg = torch.log(mg + 1e-6)
            err = (lp - lg).abs() if self.mrstft_loss_type == "l1" else (lp - lg).pow(2)

            if self.mrstft_freq_focus == "low":
                n_freq = int(mp.shape[-2])
                freqs = torch.linspace(0.0, self.fs / 2.0, steps=n_freq, device=err.device, dtype=err.dtype)
                mask = (freqs <= self.mrstft_low_hz).to(err.dtype).view(1, -1, 1)
                f_w = 1.0 + mask
                err = err * f_w
                err = err.sum() / f_w.expand_as(err).sum().clamp_min(1e-12)
            else:
                err = err.mean()

            sw = float(sc["w"])
            total = total + sw * err
            wsum += sw
        return total / max(wsum, 1e-12)
