import math
import numpy as np
import torch
import torch.nn.functional as Fnn
import torchaudio

# You said we can import from NeRAF_helper
from helper import (
    compute_t60,           # T60 (advanced=True matches RAF)
    evaluate_edt,          # EDT
    evaluate_clarity,      # C50
    compute_edc_db_np,     # Schroeder EDC (NumPy)
    normalize_edc_db_np,   # normalize EDC curve (shape emphasis)
)


class UnifiedEvaluator:
    """
    Unified NVAS evaluator for RAF/SoundSpaces-style batches.

    Inputs to evaluate():
      pred_log: Tensor [B, C, F, T] - log magnitude STFT (model output)
      gt_log  : Tensor [B, C, F, T] - log magnitude STFT (ground truth)

    Returns batch means in a dict:
      - 'stft' : STFT (spectral) MSE on linear magnitude * 2  (NeRAF style)
      - 'edc'  : EDC curve distance (bins & distance selectable)
      - 't60'  : % error (RAF advanced variant), invalids counted as 100%
      - 'edt'  : absolute error (seconds)
      - 'c50'  : absolute error (dB)
      - 'invalid_t60' : count of invalid T60 estimates in batch
    """

    def __init__(self, fs: int = 48000, edc_bins: int = 60, edc_dist: str = "l2"):
        """
        Args:
          fs: sample rate (use 48000 for RAF, 16000 for SS)
          edc_bins: number of EDC samples (default 60)
          edc_dist: 'l2' | 'l1' (default 'l2')
        """
        self.fs = int(fs)
        self.edc_bins = int(edc_bins)
        self.edc_dist = edc_dist.lower()
        if self.edc_dist not in ("l2", "l1"):
            raise ValueError("edc_dist must be 'l2' or 'l1'")

        self._gl_cache = {}

    @staticmethod
    def _fft_params_from_nfreq(nfreq: int):
        # nfreq = n_fft//2 + 1
        if nfreq == 513:     # RAF (48k typical)
            n_fft, win_length, hop_length = 1024, 512, 256
        elif nfreq == 257:   # 16k configs common in SoundSpaces
            n_fft, win_length, hop_length = 512, 256, 128
        else:
            n_fft = (nfreq - 1) * 2
            win_length = n_fft // 2
            hop_length = win_length // 2
        return n_fft, win_length, hop_length

    def _get_griffin_lim(self, device, nfreq: int):
        key = (device, nfreq)
        if key in self._gl_cache:
            return self._gl_cache[key]
        n_fft, win_length, hop_length = self._fft_params_from_nfreq(nfreq)
        gl = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1.0, n_iter=32
        ).to(device)
        self._gl_cache[key] = gl
        return gl

    @staticmethod
    def _logmag_to_mag(logmag: torch.Tensor) -> torch.Tensor:
        # NeRAF uses log(mag + 1e-3); invert consistently
        return torch.exp(logmag) - 1e-3

    @torch.no_grad()
    def _stft_loss_neraf(self, pred_log: torch.Tensor, gt_log: torch.Tensor) -> float:
        """
        NeRAF-style 'STFT error' on RAF:
        1) Reconstruct waveforms via Griffin–Lim from magnitudes.
        2) Recompute STFTs with the same n_fft/win/hop.
        3) Take mean absolute difference in log-magnitude domain.
            STFT error = mean(| log(M) - log(M_gt) |).
        """
        # 1) Reconstruct waveforms from provided log-magnitude STFTs
        wav_pred = self._waveforms_from_logmag(pred_log)  # (B, C, Tw)
        wav_gt   = self._waveforms_from_logmag(gt_log)    # (B, C, Tw)

        B, C, nfreq, _ = pred_log.shape
        n_fft, win_length, hop_length = self._fft_params_from_nfreq(nfreq)

        # 2) Recompute STFTs (magnitude) from waveforms
        def stft_mag(x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, Tw) -> return (B, C, F, T)
            mags = []
            for b in range(x.shape[0]):
                ch = []
                for c in range(x.shape[1]):
                    Xi = torch.stft(
                        torch.from_numpy(x[b, c]).to(pred_log.device),
                        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        window=torch.hann_window(win_length, device=pred_log.device),
                        return_complex=True
                    ).abs()  # (F, T)
                    ch.append(Xi.unsqueeze(0))
                mags.append(torch.stack(ch, dim=0))
            return torch.stack(mags, dim=0)  # (B, C, F, T)

        M_pred = stft_mag(wav_pred)
        M_gt   = stft_mag(wav_gt)

        # 3) Mean absolute error in log domain
        eps = 1e-3  # same stabiliser as NeRAF
        err = (torch.log(M_pred + eps) - torch.log(M_gt + eps)).abs().mean()
        return float(err.item())


    @torch.no_grad()
    def _waveforms_from_logmag(self, logmag: torch.Tensor) -> np.ndarray:
        """
        ISTFT via Griffin–Lim. Input: [B, C, F, T] log magnitude.
        Output: np.ndarray [B, C, Tw] waveforms (mono for RAF, binaural for SS).
        """
        B, C, nfreq, T = logmag.shape
        device = logmag.device
        gl = self._get_griffin_lim(device, nfreq)
        mag = self._logmag_to_mag(logmag)  # (B,C,F,T)

        wavs = []
        for b in range(B):
            ch_wavs = []
            for c in range(C):
                x_mag = mag[b, c]  # (F, T)
                wav = gl(x_mag)    # (Tw,)
                ch_wavs.append(wav.cpu().numpy())
            wavs.append(np.stack(ch_wavs, axis=0))  # (C, Tw)
        return np.stack(wavs, axis=0)  # (B, C, Tw)

    @staticmethod
    def _pad_to_same_time(a: np.ndarray, b: np.ndarray):
        """
        Pad second axis (time) to max length along time for each pair.
        a,b: (C, T) waveforms
        """
        Tm = max(a.shape[1], b.shape[1])
        if a.shape[1] < Tm:
            a = np.pad(a, ((0, 0), (0, Tm - a.shape[1])), mode='constant')
        if b.shape[1] < Tm:
            b = np.pad(b, ((0, 0), (0, Tm - b.shape[1])), mode='constant')
        return a, b

    @torch.no_grad()
    def _edc_distance_pair(self, wav_pred: np.ndarray, wav_gt: np.ndarray) -> float:
        """
        Average EDC distance over channels.
        wav_* : (C, T)
        """
        C = wav_gt.shape[0]
        dists = []
        for c in range(C):
            edc_p = compute_edc_db_np(wav_pred[c], T_target=self.edc_bins)
            edc_g = compute_edc_db_np(wav_gt[c],   T_target=self.edc_bins)
            edc_p = normalize_edc_db_np(edc_p)
            edc_g = normalize_edc_db_np(edc_g)
            diff = edc_p - edc_g
            if self.edc_dist == "l2":
                d = np.linalg.norm(diff) / math.sqrt(len(diff))
            else:
                d = np.mean(np.abs(diff))
            dists.append(float(d))
        return float(np.mean(dists))

    @torch.no_grad()
    def _t60_pair(self, wav_pred: np.ndarray, wav_gt: np.ndarray) -> (float, int):
        """
        RAF advanced T60 % error; invalids (negative) mapped to 100%.
        Returns (percent_error, invalid_count_in_pair)
        """
        t60_gt, t60_pred = compute_t60(wav_gt, wav_pred, fs=self.fs, advanced=True)
        C = wav_gt.shape[0]
        # shape (C,), compute mean relative error per channel
        rel = np.abs(t60_pred - t60_gt) / (np.abs(t60_gt) + 1e-12)
        # invalid = any invalid channel (e.g., negative)
        invalid = np.any(np.stack([t60_gt, t60_pred]) < -0.5, axis=0)
        rel[invalid] = 1.0
        err_pct = float(np.mean(rel) * 100.0)
        invalids = int(np.sum(invalid))
        return err_pct, invalids

    @torch.no_grad()
    def _edt_pair(self, wav_pred: np.ndarray, wav_gt: np.ndarray) -> float:
        edt_gt, edt_pred = evaluate_edt(wav_pred, wav_gt, fs=self.fs)
        return float(np.mean(np.abs(edt_pred - edt_gt)))

    @torch.no_grad()
    def _c50_pair(self, wav_pred: np.ndarray, wav_gt: np.ndarray) -> float:
        c50_gt, c50_pred = evaluate_clarity(wav_pred, wav_gt, fs=self.fs)
        return float(np.mean(np.abs(c50_pred - c50_gt)))

    @torch.no_grad()
    def evaluate(self, pred_log: torch.Tensor, gt_log: torch.Tensor) -> dict:
        """
        pred_log, gt_log: [B, C, F, T] log-magnitude STFT

        Returns dict of batch-mean metrics:
            stft, edc, t60, edt, c50, invalid_t60
        """
        assert pred_log.shape == gt_log.shape, "pred_log and gt_log must have the same shape"
        B, C, nfreq, T = pred_log.shape

        # 1) STFT loss (NeRAF style)
        stft_loss = self._stft_loss_neraf(pred_log, gt_log)

        # 2) Waveform reconstruction for temporal metrics
        wav_pred = self._waveforms_from_logmag(pred_log)  # (B, C, Tw)
        wav_gt   = self._waveforms_from_logmag(gt_log)    # (B, C, Tw)

        edc_vals = []
        t60_vals = []
        edt_vals = []
        c50_vals = []
        invalids = 0

        for b in range(B):
            wp, wg = wav_pred[b], wav_gt[b]   # (C, Tw)
            wp, wg = self._pad_to_same_time(wp, wg)

            edc_vals.append(self._edc_distance_pair(wp, wg))

            t60_err, inv = self._t60_pair(wp, wg)
            t60_vals.append(t60_err)
            invalids += inv

            edt_vals.append(self._edt_pair(wp, wg))
            c50_vals.append(self._c50_pair(wp, wg))

        res = {
            "stft": float(stft_loss),
            "edc":  float(np.mean(edc_vals)),
            "t60":  float(np.mean(t60_vals)),
            "edt":  float(np.mean(edt_vals)),
            "c50":  float(np.mean(c50_vals)),
            "invalid_t60": int(invalids),
        }
        return res


# -------------------------
# Minimal how-to (example)
# -------------------------
if __name__ == "__main__":
    # Dummy example on CPU with RAF-like shapes
    B, C, NF, T = 4, 1, 513, 60
    pred_log = torch.randn(B, C, NF, T)
    gt_log   = torch.randn(B, C, NF, T)

    ev = UnifiedEvaluator(fs=48000, edc_bins=60, edc_dist="l2")
    out = ev.evaluate(pred_log, gt_log)
    print(out)