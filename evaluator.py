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

    def __init__(self, fs: int = 48000, edc_bins: int = 60, edc_dist: str = "l1"):
        """
        Args:
          fs: sample rate (use 48000 for RAF, 16000 for SS)
          edc_bins: number of EDC samples (default 60)
          edc_dist: 'l2' | 'l1' (default 'l1')
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


    
def hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    B,T = x.shape
    Xf = torch.fft.fft(x, dim=1)
    h = torch.zeros(T, dtype=torch.complex64, device=x.device)
    if T%2==0:
        h[0]=1; h[1:T//2]=2; h[T//2]=1
    else:
        h[0]=1; h[1:(T+1)//2]=2
    return torch.abs(torch.fft.ifft(Xf * h, dim=1))

def _as_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure [B] -> [B,1]; pass through [B,T]."""
    if x.dim() == 1:
        return x.unsqueeze(1)
    return x

def _normalize_edc(edc: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize EDC curves per-sample so distances reflect *shape* not absolute level.
    Assumes EDC is already in dB (recommended). If you stored linear EDC, convert before calling:
        edc_db = 10*log10(edc_lin.clamp_min(eps))
    Steps: make 0 dB at t=0, then z-score over time.
    """
    # [B, T]
    edc0 = edc[:, :1]                          # first frame per sample
    edc_rel = edc - edc0                       # 0 dB at start
    std = edc_rel.std(dim=1, keepdim=True).clamp_min(eps)
    edc_norm = edc_rel / std                   # shape emphasis
    return edc_norm

def compute_edc_db(wav_1d: torch.Tensor, T_target: int = 60) -> torch.Tensor:
    """
    Schroeder EDC in dB, downsampled to T_target points.
    """
    x = wav_1d.float()
    e = x * x
    edc = torch.flip(torch.cumsum(torch.flip(e, dims=[0]), dim=0), dims=[0])
    edc = edc / (edc[0] + 1e-12)
    edc_db = 10.0 * torch.log10(edc + 1e-12)
    idx = torch.linspace(0, edc_db.numel() - 1, steps=T_target).long()
    return edc_db[idx]

def compute_dr(ir, fs=16000, direct_ms=5):
    ir = np.asarray(ir)
    # Find direct-path arrival (max abs sample)
    idx_direct = np.argmax(np.abs(ir))
    win_samples = int(direct_ms * fs / 1000)
    
    # Direct energy = ±win around arrival (or just after arrival)
    start = max(idx_direct - win_samples//2, 0)
    end   = min(idx_direct + win_samples//2, len(ir))
    direct_energy = np.sum(ir[start:end]**2)
    
    # Reverb energy = after that window
    reverb_energy = np.sum(ir[end:]**2)
    
    # Avoid log(0)
    return 10 * np.log10((direct_energy + 1e-12) / (reverb_energy + 1e-12))

def compute_audio_distance(
    stft: torch.Tensor,          # [B, M] or [B, F, T]
    wavs: torch.Tensor = None,   # [B, T]    
    edc_curves: torch.Tensor = None,  # [B, T_edc] precomputed EDC in dB
    decay_feats: torch.Tensor = None, # [B, D] e.g. [T60, C50, EDT]
    metric: str = 'SPL',
    fs: int = 16000,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute [B, B] distance matrix on GPU for various metrics.

    stft: either a flattened spectrogram [B, M] or a full 3D STFT [B, F, T].
    metric: one of 'MAG', 'MAG_HELPER', 'MAG2', 'MSE', 'SC', 'LSD', 'ENV', 'SPL'.
    """
    device = stft.device
    B = stft.size(0)

    # If stft is 3D, reshape to 2D for pairwise cdist
    if stft.dim() == 3:
        B, F, T = stft.shape
        flat = stft.reshape(B, F * T)
    else:
        flat = stft
        F = T = None

    # Choose metric
    if metric == 'MAG':
        # Spectral L₂ Distance
        lin = torch.exp(flat) - eps        # [B, M]
        D = torch.cdist(lin, lin, p=2)

    elif metric == 'MAG_HELPER':
        # Time-Averaged Spectral MSE
        # Sum over frequencies of per-frame MSE (like helper.Magnitude_distance)
        assert stft.dim() == 3, "MAG_HELPER requires stft shape [B, F, T]"
        lin3 = torch.exp(stft) - eps        # [B, F, T]
        flat = lin3.reshape(B, F * T)
        D2 = torch.cdist(flat, flat, p=2) ** 2  # sum of squared diffs
        D = D2 / T                                # divide by T (frames)

    elif metric == 'MAG2':
        # Normalized Spectral MSE×2
        # Full MSE on linear magnitude times 2 (like get_stft_metrics)
        assert stft.dim() == 3, "MAG2 requires stft shape [B, F, T]"
        lin3 = torch.exp(stft) - eps
        flat = lin3.reshape(B, F * T)
        D2 = torch.cdist(flat, flat, p=2) ** 2
        D = D2 / (F * T) * 2

    elif metric == 'MSE':
        if wavs is None:
            raise ValueError("`wavs` required for MSE metric")
        B, T_w = wavs.shape
        D2 = torch.cdist(wavs, wavs, p=2) ** 2
        D = D2 / T_w

    elif metric == 'SC':
        lin = torch.exp(flat) - eps
        D = torch.cdist(lin, lin, p=2)
        norm = lin.norm(dim=1, keepdim=True)
        D = D / (norm + eps)

    elif metric == 'LSD':
        D = torch.cdist(flat, flat, p=2)
        D = D / math.sqrt(flat.size(1))

    elif metric == 'ENV':
        if wavs is None:
            raise ValueError("`wavs` required for ENV metric")
        env = hilbert_envelope(wavs)
        D = torch.cdist(env, env, p=2)

    elif metric == 'SPL':
        # MSE on log-magnitude (recomputed from wavs)
        D2 = torch.cdist(flat, flat, p=2) ** 2
        D = D2 / flat.size(1)
    
    elif metric == 'EDC':
        # Prefer precomputed curves; else derive from wavs
        if edc_curves is None:
            if wavs is None:
                raise ValueError("Metric 'EDC' needs edc_curves or wavs.")
            # build [B, T_edc] in dB on the same device
            edc_list = [compute_edc_db(wavs[i].detach().cpu()) for i in range(B)]
            edc_db = torch.stack(edc_list, dim=0).to(device)      # [B, T_edc]
        else:
            edc_db = edc_curves.to(device)                         # already dB

        # normalize shape (same _normalize_edc you already use)
        edc = _normalize_edc(edc_db, eps=eps)                      # [B, T_edc]
        D = torch.cdist(edc, edc, p=2) / math.sqrt(edc.size(1))

    elif metric == 'T60':
        # Raw-seconds distance OR fall back to helpers
        if decay_feats is not None:
            t60 = decay_feats[:, 0:1].to(device)
            D = torch.cdist(t60, t60, p=2)
        else:
            if wavs is None:
                raise ValueError("Metric 'T60' needs decay_feats or (wavs + helpers).")
            # pairwise raw |ΔT60| (seconds)
            D = torch.zeros(B, B, device=device)
            for i in range(B):
                wi = wavs[i].detach().cpu().numpy()[None, :]
                for j in range(i + 1, B):
                    wj = wavs[j].detach().cpu().numpy()[None, :]
                    t60_i, t60_j = compute_t60(wi, wj, fs=fs, advanced=True)
                    t60_i = float(np.atleast_1d(t60_i)[0])
                    t60_j = float(np.atleast_1d(t60_j)[0])
                    val = abs(t60_i - t60_j) if (t60_i >= 0 and t60_j >= 0) else float('inf')
                    D[i, j] = D[j, i] = val

    elif metric == 'T60PCT':
        if decay_feats is not None:
            t60 = decay_feats[:, 0:1].to(device)
            diff  = (t60 - t60.t()).abs()
            denom = torch.maximum(t60.abs(), t60.abs().t())
            D = (diff / (denom + eps)) * 100.0
            invalid = (t60 < 0) | (t60.t() < 0)
            D = torch.where(invalid, torch.full_like(D, 100.0), D)
        else:
            if wavs is None:
                raise ValueError("Metric 'T60PCT' needs decay_feats or (wavs + helpers).")
            D = torch.zeros(B, B, device=device)
            for i in range(B):
                wi = wavs[i].detach().cpu().numpy()[None, :]
                for j in range(i + 1, B):
                    wj = wavs[j].detach().cpu().numpy()[None, :]
                    t60_i, t60_j = compute_t60(wi, wj, fs=fs, advanced=True)
                    t60_i = float(np.atleast_1d(t60_i)[0])
                    t60_j = float(np.atleast_1d(t60_j)[0])
                    if (t60_i < 0) or (t60_j < 0):
                        val = 100.0
                    else:
                        denom = max(abs(t60_i), abs(t60_j)) + 1e-12
                        val = abs(t60_i - t60_j) / denom * 100.0
                    D[i, j] = D[j, i] = val

    elif metric == 'C50':
        if decay_feats is not None:
            c50 = decay_feats[:, 1:2].to(device)
            D = torch.cdist(c50, c50, p=2)
        else:
            if wavs is None:
                raise ValueError("Metric 'C50' needs decay_feats or (wavs + helpers).")
            D = torch.zeros(B, B, device=device)
            for i in range(B):
                wi = wavs[i].detach().cpu().numpy()[None, :]
                for j in range(i + 1, B):
                    wj = wavs[j].detach().cpu().numpy()[None, :]
                    c_gt, c_x = evaluate_clarity(wj, wi, fs=fs)  # returns per-sample values
                    c_gt = float(np.atleast_1d(c_gt)[0])
                    c_x  = float(np.atleast_1d(c_x)[0])
                    val = abs(c_x - c_gt)
                    D[i, j] = D[j, i] = val

    elif metric == 'EDT':
        if decay_feats is not None:
            edt = decay_feats[:, 2:3].to(device)
            D = torch.cdist(edt, edt, p=2)
        else:
            if wavs is None:
                raise ValueError("Metric 'EDT' needs decay_feats or (wavs + helpers).")
            D = torch.zeros(B, B, device=device)
            for i in range(B):
                wi = wavs[i].detach().cpu().numpy()[None, :]
                for j in range(i + 1, B):
                    wj = wavs[j].detach().cpu().numpy()[None, :]
                    e_gt, e_x = evaluate_edt(wj, wi, fs=fs)
                    e_gt = float(np.atleast_1d(e_gt)[0])
                    e_x  = float(np.atleast_1d(e_x)[0])
                    val = abs(e_x - e_gt)
                    D[i, j] = D[j, i] = val
    elif metric == 'DR':
        if decay_feats is not None:
            dr = decay_feats[:, 3:4].to(device)  # assuming DR is 4th column
            D = torch.cdist(dr, dr, p=2)
        else:
            if wavs is None:
                raise ValueError("Metric 'DR' needs decay_feats or (wavs + helpers).")
            print("Calculating D/R")
            D = torch.zeros(B, B, device=device)
            for i in range(B):
                wi = wavs[i].detach().cpu().numpy()
                for j in range(i + 1, B):
                    wj = wavs[j].detach().cpu().numpy()
                    dr_i = compute_dr(wi, fs=fs)  # implement or import this
                    dr_j = compute_dr(wj, fs=fs)
                    val = abs(dr_i - dr_j)
                    D[i, j] = D[j, i] = val
    elif metric == 'SPECENV':
        # Time-averaged log-magnitude spectral envelope distance
        assert stft.dim() == 3, "SPECENV requires stft shape [B, F, T]"
        # Convert from log-magnitude back to linear, then average over time
        lin_mag = torch.exp(stft) - eps            # [B, F, T]
        env = lin_mag.mean(dim=2)                  # [B, F] mean over time
        log_env = torch.log(env + eps)             # log spectral envelope
        # Euclidean distance between envelopes
        D = torch.cdist(log_env, log_env, p=2) / math.sqrt(env.size(1))

    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Mask self-distances
    idx = torch.arange(B, device=device)
    D[idx, idx] = float('inf')
    return D


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