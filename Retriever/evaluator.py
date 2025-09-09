import math
import os
# --- NEW imports for NPY eval ---
import torch
from torchaudio.transforms import GriffinLim
import sys
sys.path.append('../NeRAF')
from NeRAF_helper import compute_t60 as _helper_compute_t60
from NeRAF_helper import evaluate_edt as _helper_evaluate_edt
from NeRAF_helper import evaluate_clarity as _helper_evaluate_clarity
import numpy as np
from torch.utils.data import Dataset
import glob, numpy as _np
from torchaudio.transforms import GriffinLim

# ======================= NPY vs Model Evaluation ============================

# ---------------- Distance Utilities ----------------
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
                    t60_i, t60_j = _helper_compute_t60(wi, wj, fs=fs, advanced=True)
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
                    t60_i, t60_j = _helper_compute_t60(wi, wj, fs=fs, advanced=True)
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
                    c_gt, c_x = _helper_evaluate_clarity(wj, wi, fs=fs)  # returns per-sample values
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
                    e_gt, e_x = _helper_evaluate_edt(wj, wi, fs=fs)
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

# ---------------- Eval-NPY helpers -----------------------------------
class _EvalNPYCache:
    """
    Loads a deterministic subset of eval npy files once, reconstructs wav_pred once,
    and holds GT/pred tensors for quick per-epoch comparisons.
    """
    def __init__(self, npy_ds: Dataset, sample_rate: int, pattern: str, max_files: int, seed: int):

        self.sample_rate = sample_rate
        self.npy_ds = npy_ds

        paths = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        if not paths:
            print(f"[EvalNPY] No files matched: {pattern}")
            self.records = []
            return

        # deterministic subset
        rng = np.random.default_rng(seed)
        sel_idx = np.arange(len(paths))
        rng.shuffle(sel_idx)
        if max_files is not None:
            sel_idx = sel_idx[:max_files]
        paths = [paths[i] for i in sorted(sel_idx.tolist())]

        # ISTFT consistent with dataset STFT
        if sample_rate == 48000:
            n_fft, win_length, hop_length = 1024, 512, 256
        elif sample_rate == 16000:
            n_fft, win_length, hop_length = 512, 256, 128
        else:
            raise ValueError(f"Unsupported sample rate {sample_rate}")
        self.istft = GriffinLim(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1)

        self.records = []
        for p in paths:
            d = _np.load(p, allow_pickle=True).item()
            idx = int(d["audio_idx"])
            gt = npy_ds[idx]  # pulls GT stft/wav/edc/decay

            # rebuild prediction wav from stored log-mag STFT
            stft_pred_log = torch.from_numpy(d["pred_stft"]).float().squeeze(0)  # [F,T]
            mag_pred = torch.exp(stft_pred_log) - 1e-3
            wav_pred = self.istft(mag_pred.unsqueeze(0)).squeeze(0)               # [T']

            # align for time-domain metrics
            wav_gt = gt['wav']
            L = min(wav_gt.shape[0], wav_pred.shape[0])
            wav_pred = wav_pred[:L]
            wav_gt   = wav_gt[:L]

            rec = {
                'id': gt['id'],
                'ds_idx': idx,
                'stft_gt': gt['stft'], 'wav_gt': wav_gt,
                'stft_pred': stft_pred_log, 'wav_pred': wav_pred,
                'edc_gt': gt.get('edc'), 'decay_gt': gt.get('decay_feats'),
                # filled-once metrics to avoid recompute across epochs
                'metrics_pred_vs_gt': None
            }
            self.records.append(rec)

        print(f"[EvalNPY] Cached {len(self.records)} eval npy samples.")

def _room_metric_diffs_np(gt_wav: torch.Tensor, x_wav: torch.Tensor, fs: int):
    """Returns {'EDT','C50','T60'} using your NeRAF helpers with robust handling."""
    # Convert to numpy [1,T]
    gt = gt_wav.detach().cpu().numpy()[None, :]
    xx = x_wav.detach().cpu().numpy()[None, :]
    t60_gt, t60_x = _helper_compute_t60(gt, xx, fs=fs, advanced=True)
    t60_gt = float(np.atleast_1d(t60_gt)[0]); t60_x = float(np.atleast_1d(t60_x)[0])
    if (t60_gt < 0) or (t60_x < 0):
        t60_err_pct = 100.0
    else:
        denom = max(abs(t60_gt), abs(t60_x)) + 1e-12
        t60_err_pct = abs(t60_x - t60_gt) / denom * 100.0

    c50_gt, c50_x = _helper_evaluate_clarity(xx, gt, fs=fs)
    edt_gt, edt_x = _helper_evaluate_edt(xx, gt, fs=fs)
    c50_mae = float(np.mean(np.abs(np.atleast_1d(c50_x) - np.atleast_1d(c50_gt))))
    edt_mae = float(np.mean(np.abs(np.atleast_1d(edt_x)  - np.atleast_1d(edt_gt))))
    return {'EDT': edt_mae, 'C50': c50_mae, 'T60': t60_err_pct}

def _pair_metrics_with_edc(stft_a, wav_a, stft_b, wav_b, edc_a, edc_b, fs: int):
    """
    Compute MSE/SPL/MAG/MAG2/EDC + room metrics for a pair. Uses your compute_audio_distance.
    """
    # align WAVs for time metrics
    L = min(wav_a.shape[0], wav_b.shape[0])
    wav_a = wav_a[:L]; wav_b = wav_b[:L]

    # EDC curves (align to same length if both present; else compute from wavs)
    pair_stft = torch.stack([stft_a, stft_b], dim=0)               # [2,F,T] or flat
    pair_wavs = torch.stack([wav_a, wav_b], dim=0)                 # [2,T]

    if edc_a is not None and edc_b is not None:
        T_edc = edc_a.shape[0]
        edc_b = (edc_b[:T_edc] if edc_b.shape[0] >= T_edc else F.pad(edc_b, (0, T_edc - edc_b.shape[0])))
        pair_edc = torch.stack([edc_a, edc_b], dim=0)
    else:
        pair_edc = None

    mse  = compute_audio_distance(pair_stft, wavs=pair_wavs, metric='MSE', fs=fs)[0,1].item()
    spl  = compute_audio_distance(pair_stft, wavs=pair_wavs, metric='SPL', fs=fs)[0,1].item()
    mag  = compute_audio_distance(pair_stft, metric='MAG')[0,1].item()
    mag2 = compute_audio_distance(pair_stft, metric='MAG2')[0,1].item()
    edcD = compute_audio_distance(pair_stft, wavs=pair_wavs if pair_edc is None else None,
                                  edc_curves=pair_edc, metric='EDC', fs=fs)[0,1].item()
    out = {'MSE': mse, 'SPL': spl, 'MAG': mag, 'MAG2': mag2, 'EDC': edcD}
    out.update(_room_metric_diffs_np(wav_a, wav_b, fs=fs))
    return out
