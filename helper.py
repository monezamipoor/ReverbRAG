import os 
import torch
import numpy as np
import pyroomacoustics
from scipy.signal import hilbert
import torchaudio
from torch.nn import functional as F
from torch.utils.data import BatchSampler
import torch.nn as nn
import math
from typing import Optional, List, Tuple
import random


# ---------- helper conversions ----------
def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def _hz_to_erbrate(hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1.0 + 0.00437 * hz)

def _erbrate_to_hz(erbrate: torch.Tensor) -> torch.Tensor:
    return (10.0**(erbrate / 21.4) - 1.0) / 0.00437

""" Helper methods for evaluation """

class SpectralLoss(nn.Module):
    """
    Compute a loss between two log power-spectrograms.
    From  https://github.com/facebookresearch/SING/blob/main/sing/dsp.py#L79 modified

    Arguments:
        base_loss (function): loss used to compare the log power-spectrograms.
            For instance :func:`F.mse_loss`
        epsilon (float): offset for the log, i.e. `log(epsilon + ...)`
        **kwargs (dict): see :class:`STFT`
    """

    def __init__(self, base_loss=F.mse_loss, reduction='mean', epsilon=1, dB=False, stft_input_type='mag', **kwargs):
        super(SpectralLoss, self).__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon
        self.dB = dB
        self.stft_input_type = stft_input_type
        self.reduction = reduction

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def _log_spectrogram(self, STFT):
        if self.dB and self.stft_input_type == 'mag':
            return 10*torch.log10(self.epsilon + STFT) 
        elif not self.dB and self.stft_input_type == 'mag':
            return torch.log(self.epsilon + STFT)
        elif self.stft_input_type == 'log mag': 
            return STFT

    def forward(self, a, b):
        spec_a = self._log_spectrogram(a)
        spec_b = self._log_spectrogram(b)
        return self.base_loss(spec_a, spec_b, reduction=self.reduction)


def make_tri_filterbank(Fbins: int, n_bands: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Triangular filterbank over Fbins producing n_bands coarse bands.
    Returns (n_bands, Fbins), rows sum to 1.
    """
    centers = torch.linspace(0, Fbins - 1, n_bands)
    width = Fbins / n_bands
    f = torch.arange(Fbins).unsqueeze(0)                 # (1, F)
    c = centers.unsqueeze(1)                             # (n_bands, 1)
    fb = 1.0 - (f - c).abs() / width                     # triangular
    fb = fb.clamp_min(0.0)
    fb = fb / (fb.sum(dim=1, keepdim=True) + eps)
    return fb                                            # (n_bands, Fbins)

def make_erb_filterbank(
    Fbins: int,
    n_bands: int,
    nyquist_hz: float = 24000,
    f_min: float = 20.0,
    eps: float = 1e-8,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    ERB-spaced triangular filterbank over Fbins -> (n_bands, Fbins). Rows sum to 1.
    Requires Nyquist frequency to map bins <-> Hz.
    """
    if dtype is None: dtype = torch.get_default_dtype()
    if device is None: device = "cpu"

    # frequency axis for STFT bins [0..F-1] mapped to Hz
    f_hz = torch.linspace(0.0, nyquist_hz, Fbins, device=device, dtype=dtype)  # (F,)

    # ERB-rate axis and evenly spaced ERB centers between f_min..Nyquist
    # (skip the very-low region: f_min avoids a giant DC band)
    E_min = _hz_to_erbrate(torch.tensor(f_min, device=device, dtype=dtype))
    E_max = _hz_to_erbrate(torch.tensor(nyquist_hz, device=device, dtype=dtype))
    E_centers = torch.linspace(E_min, E_max, n_bands, device=device, dtype=dtype)     # (B,)
    f_centers = _erbrate_to_hz(E_centers)                                              # (B,)

    # Convert ERB-spaced center freqs to *bin* positions (float)
    # bin_idx = f / nyquist * (F-1)
    bin_centers = (f_centers / nyquist_hz) * (Fbins - 1)                               # (B,)

    # Triangle supports via midpoints between adjacent centers (in *bin* space)
    # For edges, extrapolate a half-step so every triangle has left/right support.
    mids = 0.5 * (bin_centers[1:] + bin_centers[:-1])                                  # (B-1,)
    lefts  = torch.empty_like(bin_centers)
    rights = torch.empty_like(bin_centers)
    lefts[1:]  = mids
    rights[:-1] = mids
    # edge extrapolation
    lefts[0]   = bin_centers[0] - (rights[0] - bin_centers[0])
    rights[-1] = bin_centers[-1] + (bin_centers[-1] - lefts[-1])

    # Build triangles in bin domain (like your linear version, but variable widths)
    f_bins = torch.arange(Fbins, device=device, dtype=dtype).unsqueeze(0)              # (1,F)
    c  = bin_centers.unsqueeze(1)                                                      # (B,1)
    L  = lefts.unsqueeze(1)
    R  = rights.unsqueeze(1)

    # left ramp: from L -> C, right ramp: from C -> R
    left_ramp  = (f_bins - L) / (c - L + eps)
    right_ramp = (R - f_bins) / (R - c + eps)
    fb = torch.minimum(left_ramp, right_ramp).clamp_min(0.0)                           # (B,F)

    # Row-normalize to sum=1 per band
    fb = fb / (fb.sum(dim=1, keepdim=True) + eps)
    return fb

class ERBPool(nn.Module):
    """Pools (B, T, F) or (B, F) -> (B, T, n_bands) or (B, n_bands) via fixed filterbank."""
    def __init__(self, Fbins: int, n_bands: int = 24):
        super().__init__()
        fb = make_erb_filterbank(Fbins, n_bands)
        self.register_buffer("fb", fb)  # (n_bands, Fbins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:  # (B, F)
            return x @ self.fb.T        # (B, n_bands)
        elif x.ndim == 3: # (B, T, F)
            B, T, F = x.shape
            y = x.reshape(B*T, F) @ self.fb.T
            return y.reshape(B, T, -1)  # (B, T, n_bands)
        else:
            raise ValueError(f"Expected (B,F) or (B,T,F), got {x.shape}")

class SpectralEnvelopePreservationLoss(nn.Module):
    """
    Coarse-band spectral envelope preservation.
    If input is (B,F): matches current frame envelope.
    If input is (B,T,F): uses mean over time (uniform; with T=1 it's identical).
    """
    def __init__(self, Fbins: int, n_bands: int = 24, reduction: str = "mean"):
        super().__init__()
        self.pool = ERBPool(Fbins, n_bands)
        self.reduction = reduction

    def forward(self, x_log: torch.Tensor, base_log: torch.Tensor) -> torch.Tensor:
        # ERB pooling
        x_bands   = self.pool(x_log)     # (B, n_bands) or (B, T, n_bands)
        base_bands = self.pool(base_log)

        # If time present, average over T (for T=1 this is a no-op)
        if x_bands.ndim == 3:
            x_env = x_bands.mean(dim=1)          # (B, n_bands)
            with torch.no_grad():
                base_env = base_bands.mean(dim=1)
        else:
            x_env = x_bands                      # (B, n_bands)
            with torch.no_grad():
                base_env = base_bands

        return F.mse_loss(x_env, base_env, reduction=self.reduction)



def compute_t60(true_in, gen_in, fs, advanced = False):
    ch = true_in.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        try:
            if advanced: 
                true = measure_rt60_advance(true_in[c], sr=fs)
                gen = measure_rt60_advance(gen_in[c], sr=fs)
            else:
                true = pyroomacoustics.experimental.measure_rt60(true_in[c], fs=fs, decay_db=30)
                gen = pyroomacoustics.experimental.measure_rt60(gen_in[c], fs=fs, decay_db=30)
        except:
            true = -1
            gen = -1
        gt.append(true)
        pred.append(gen)
    return np.array(gt), np.array(pred)

def measure_rt60_advance(signal, sr, decay_db=10, cutoff_freq=200):
    # following RAF implementation
    signal = torch.from_numpy(signal)
    signal = torchaudio.functional.highpass_biquad(
        waveform=signal,
        sample_rate=sr,
        cutoff_freq=cutoff_freq
    )
    signal = signal.cpu().numpy()
    rt60 = pyroomacoustics.experimental.measure_rt60(signal, sr, decay_db=decay_db, plot=False)
    return rt60

def Envelope_distance(predicted, gt):
    ch = predicted.shape[0]
    envelope_distance=0
    for c in range(ch):
        pred_env = np.abs(hilbert(predicted[c,:]))
        gt_env = np.abs(hilbert(gt[c,:]))
        distance = np.sqrt(np.mean((gt_env - pred_env)**2))
        envelope_distance += distance
    return float(envelope_distance)

def SNR(predicted, gt):
    mse_distance = np.mean(np.power((predicted - gt), 2))
    snr = 10. * np.log10((np.mean(gt**2) + 1e-4) / (mse_distance + 1e-4))
    return float(snr)

def normalize(samples):
    return samples / np.maximum(1e-20, np.max(np.abs(samples)))

def Magnitude_distance(predicted_mag, gt_mag):
    ch = predicted_mag.shape[0]
    stft_mse = 0
    for c in range(ch): 
        stft_mse += np.mean(np.power(predicted_mag[c] - gt_mag[c], 2))
    return float(stft_mse)

def Magnitude_distance_L2(predicted_mag, gt_mag):
    ch = predicted_mag.shape[0]
    l2_sum = 0
    for c in range(ch):
        diff = predicted_mag[c] - gt_mag[c]
        l2_sum += np.sqrt(np.sum(diff ** 2))  # L2 norm
    return float(l2_sum)
    
def measure_clarity(signal, time=50, fs=44100):
    h2 = signal**2
    t = int((time/1000)*fs + 1) 
    return 10*np.log10(np.sum(h2[:t])/np.sum(h2[t:]))

def evaluate_clarity(pred_ir, gt_ir, fs):
    np_pred_ir = pred_ir
    np_gt_ir = gt_ir

    # manage multiple channels IR
    ch = gt_ir.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        pred_clarity = measure_clarity(np_pred_ir[c,...], fs=fs)
        gt_clarity = measure_clarity(np_gt_ir[c,...], fs=fs)
        gt.append(gt_clarity)
        pred.append(pred_clarity)
    return np.array(gt), np.array(pred)

def measure_edt(h, fs=44100, decay_db=10):
    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    if np.all(energy == 0):
        return np.nan

    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs
    # compute the decay time
    decay_time = t_decay
    est_edt = (60 / decay_db) * decay_time 
    return est_edt

def evaluate_edt(pred_ir, gt_ir, fs):
    np_pred_ir = pred_ir
    np_gt_ir = gt_ir

    # manage multiple channels IR
    ch = gt_ir.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        pred_edt = measure_edt(np_pred_ir[c], fs=fs)
        gt_edt = measure_edt(np_gt_ir[c], fs=fs)
        gt.append(gt_edt)
        pred.append(pred_edt)
    return np.array(gt), np.array(pred)

def compute_edc_db_np(wav_1d: np.ndarray, T_target: int = 60) -> np.ndarray:
    """
    Schroeder EDC in dB, downsampled to T_target points (NumPy version).
    """
    x = wav_1d.astype(np.float32)
    e = x * x
    edc = np.flip(np.cumsum(np.flip(e, axis=0), axis=0), axis=0)
    if edc[0] <= 1e-12:
        edc = edc + 1e-12
    edc = edc / edc[0]
    edc_db = 10.0 * np.log10(edc + 1e-12)

    idx = np.linspace(0, edc_db.shape[0] - 1, num=T_target).astype(np.int64)
    return edc_db[idx]

def normalize_edc_db_np(edc_db: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Make 0 dB at t=0, then z-score over time so distance emphasizes *shape*.
    """
    edc_rel = edc_db - edc_db[0]
    std = np.maximum(np.std(edc_rel), eps)
    return edc_rel / std

def edc_distance_np(pred_wav: np.ndarray,
                    gt_wav: np.ndarray,
                    T_target: int = 60,
                    normalize: bool = True) -> float:
    """
    L2 distance between (optionally normalized) EDC curves, normalized by sqrt(T).
    Works on 1D waveforms. For multichannel, call per-channel and average.
    """
    edc_p = compute_edc_db_np(pred_wav, T_target)
    edc_g = compute_edc_db_np(gt_wav,   T_target)
    if normalize:
        edc_p = normalize_edc_db_np(edc_p)
        edc_g = normalize_edc_db_np(edc_g)
    diff = edc_p - edc_g
    return float(np.linalg.norm(diff) / np.sqrt(len(diff)))

def _schroeder_edc_db_torch(wav: torch.Tensor, T_target: int = 60) -> torch.Tensor:
    """
    wav: [B, C, T] or [B, T] waveform in float32/float64, on any device.
    Returns: [B, C, T_target] (or [B, 1, T_target]) EDC in dB.
    """
    if wav.dim() == 2:
        wav = wav.unsqueeze(1)            # [B, 1, T]
    B, C, T = wav.shape
    e = wav * wav                         # energy
    # Schroeder integration: reverse -> cumsum -> reverse
    edc = torch.flip(torch.cumsum(torch.flip(e, dims=[-1]), dim=-1), dims=[-1])  # [B, C, T]
    edc = edc / (edc[..., :1] + 1e-12)    # normalize by initial energy
    edc_db = 10.0 * torch.log10(edc + 1e-12)

    # Downsample to T_target with linear indexing (no gradient through indices)
    idx = torch.linspace(0, T - 1, steps=T_target, device=wav.device)
    idx_long = idx.long()
    # gather along time
    edc_db_ds = edc_db[..., idx_long]     # [B, C, T_target]
    return edc_db_ds

def _normalize_edc_shape_torch(edc_db: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Zero at t=0 per (B,C), then z-score across time to emphasize *shape*.
    edc_db: [B, C, T]
    """
    edc_rel = edc_db - edc_db[..., :1]                    # 0 dB at start
    std = edc_rel.float().std(dim=-1, keepdim=True).clamp_min(eps)
    return edc_rel / std

def edc_l2_loss_torch(
    pred_wav: torch.Tensor, gt_wav: torch.Tensor,
    T_target: int = 60, normalize: bool = True, reduction: str = "mean"
) -> torch.Tensor:
    """
    L2 distance between (optionally normalized) EDC curves, normalized by sqrt(T).
    Returns scalar if reduction='mean' else per-sample [B].
    """
    # shape: [B, C, T] alignment
    if pred_wav.dim() == 2: pred_wav = pred_wav.unsqueeze(1)
    if gt_wav.dim() == 2:   gt_wav   = gt_wav.unsqueeze(1)

    E_p = _schroeder_edc_db_torch(pred_wav, T_target)     # [B, C, Tt]
    E_g = _schroeder_edc_db_torch(gt_wav,   T_target)

    if normalize:
        E_p = _normalize_edc_shape_torch(E_p)
        E_g = _normalize_edc_shape_torch(E_g)

    diff = E_p - E_g                                       # [B, C, Tt]
    # Frobenius over (C,Tt), then / sqrt(Tt)
    per_sample = torch.linalg.vector_norm(diff.flatten(1), ord=2, dim=1) / (T_target ** 0.5)
    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "none":
        return per_sample
    else:
        return per_sample.sum()


class EDCLoss(nn.Module):
    def __init__(self, T_target: int = 60, normalize: bool = True, reduction: str = "mean", eps: float = 1e-12):
        super().__init__()
        self.T_target = int(T_target)
        self.normalize = bool(normalize)
        self.reduction = reduction
        self.eps = eps
        # optional: a buffer you could reuse if T is fixed; we’ll build idx on the fly since T can vary
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    @staticmethod
    def _schroeder_edc_db(wav: torch.Tensor, eps: float) -> torch.Tensor:
        # wav: [B, T] or [B, C, T]
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)                    # [B, 1, T]
        e = wav.float() * wav.float()
        edc = torch.flip(torch.cumsum(torch.flip(e, dims=[-1]), dim=-1), dims=[-1])  # [B, C, T]
        edc = edc / (edc[..., :1] + eps)
        return 10.0 * torch.log10(edc + eps)         # [B, C, T]

    def forward(self, pred_wav: torch.Tensor, gt_wav: torch.Tensor) -> torch.Tensor:
        Ep = self._schroeder_edc_db(pred_wav, self.eps)
        Eg = self._schroeder_edc_db(gt_wav,   self.eps)

        B, C, T = Ep.shape
        idx = torch.linspace(0, T - 1, steps=self.T_target, device=Ep.device).long()
        Ep = Ep[..., idx]                            # [B, C, Tt]
        Eg = Eg[..., idx]

        if self.normalize:
            Ep = (Ep - Ep[..., :1]) / Ep.std(dim=-1, keepdim=True).clamp_min(1e-6)
            Eg = (Eg - Eg[..., :1]) / Eg.std(dim=-1, keepdim=True).clamp_min(1e-6)

        diff = Ep - Eg                               # [B, C, Tt]
        per_sample = torch.linalg.vector_norm(diff.flatten(1), ord=2, dim=1) / (self.T_target ** 0.5)

        if self.reduction == "none":
            return per_sample
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()

# ---------- unified builder ----------
def build_band2freq_matrix(
    Fbins: int = 513,
    n_fft: int = 1024,
    sr: int = 48000,
    n_bands: int = 32,
    scale: str = "erb",   # "erb" or "mel"
    fmin: float = 20.0,
    fmax: float = None,
    device="cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Build a band-to-frequency triangular matrix for upsampling band masks to FFT bins.

    Returns:
        W: (Fbins, n_bands)
    """

    if fmax is None:
        fmax = sr / 2.0

    freqs_hz = torch.linspace(0.0, sr/2.0, Fbins, device=device, dtype=dtype)

    if scale == "mel":
        to_scale = _hz_to_mel
        from_scale = _mel_to_hz
    elif scale == "erb":
        to_scale = _hz_to_erbrate
        from_scale = _erbrate_to_hz
    else:
        raise ValueError("scale must be 'mel' or 'erb'")

    # edges in scale domain
    s_min = to_scale(torch.tensor([fmin], device=device, dtype=dtype))
    s_max = to_scale(torch.tensor([fmax], device=device, dtype=dtype))
    s_edges = torch.linspace(s_min.item(), s_max.item(), n_bands + 2, device=device, dtype=dtype)
    f_edges = from_scale(s_edges)  # Hz positions of band edges

    # build triangular weights
    W = torch.zeros(Fbins, n_bands, device=device, dtype=dtype)
    for b in range(n_bands):
        f_l, f_c, f_r = f_edges[b], f_edges[b+1], f_edges[b+2]
        left  = (freqs_hz >= f_l) & (freqs_hz <= f_c)
        right = (freqs_hz >= f_c) & (freqs_hz <= f_r)
        W[left,  b] = (freqs_hz[left]  - f_l) / (f_c - f_l + 1e-12)
        W[right, b] = (f_r - freqs_hz[right]) / (f_r - f_c + 1e-12)

    # normalize so each band contributes equally
    col_sums = W.sum(dim=0, keepdim=True) + 1e-12
    W = W / col_sums

    return W  # (Fbins, n_bands)


class BandQueryMaskHead(nn.Module):
    """
    Shared-trunk (W -> 64) + per-band queries (dot with 64-d h_shared) + tiny Conv1D smoothing.
    Outputs tanh-bounded band mask in [-1, 1].

    forward(q_W, g_W=None) expects (B, W) each.
    If g_W is given, the head uses [q_W || g_W] -> Linear(2W->W) pre-projection.
    """
    def __init__(self, W: int, bands: int = 32, use_g: bool = True, conv_kernel: int = 3):
        super().__init__()
        self.W = W
        self.bands = bands
        self.use_g = use_g

        # Optional 2W->W compressor when using both q_W and g_W
        self.pre = nn.Linear(2 * W, W) if use_g else nn.Identity()

        # Shared trunk: LN -> GELU -> Linear(W->64) -> GELU -> Linear(64->64) -> Residual
        self.ln = nn.LayerNorm(W)
        self.fc1 = nn.Linear(W, 64)
        self.fc2 = nn.Linear(64, 64)
        self.act = nn.GELU()

        # Per-band queries (bands x 64) + bias (bands)
        self.band_queries = nn.Parameter(torch.empty(bands, 64))
        self.band_bias = nn.Parameter(torch.zeros(bands))
        nn.init.normal_(self.band_queries, mean=0.0, std=0.02)

        # Tiny Conv1D over bands for coherence (identity-initialized)
        if conv_kernel is not None and conv_kernel >= 3 and conv_kernel % 2 == 1:
            self.smooth = nn.Conv1d(1, 1, kernel_size=conv_kernel, padding=conv_kernel // 2, bias=True)
            with torch.no_grad():
                self.smooth.weight.zero_()
                # identity kernel: center=1, others=0
                self.smooth.weight[0, 0, conv_kernel // 2] = 1.0
                self.smooth.bias.zero_()
        else:
            self.smooth = None

    def forward(self, q_W: torch.Tensor, g_W: torch.Tensor = None) -> torch.Tensor:
        """
        Returns: (B, bands) in [-1, 1] via tanh.
        """
        if self.use_g and g_W is not None:
            xW = self.pre(torch.cat([q_W, g_W], dim=-1))   # (B, W)
        else:
            xW = q_W                                      # (B, W)

        # Shared trunk to 64-d
        h = self.act(self.fc1(self.act(self.ln(xW))))     # LN -> GELU -> Linear(W->64) -> GELU
        h = h + self.fc2(h)                               # Residual (64)

        # Band logits via per-band queries
        # logits[b] = band_queries @ h + band_bias
        logits = F.linear(h, self.band_queries, self.band_bias)  # (B, bands)

        # Optional smoothing over bands
        if self.smooth is not None:
            logits = self.smooth(logits.unsqueeze(1)).squeeze(1)  # (B, bands)

        return logits

# The main EDC Loss I used in my training loop
def _edc_from_mag(mag, T_target=60, eps=1e-12):
    # mag: [C, F, T]
    E = (mag ** 2).sum(dim=1)                          # [C, T] frame energy
    S = torch.flip(torch.cumsum(torch.flip(E, [-1]), -1), [-1])  # Schroeder [C, T]
    S = S / (S[..., :1] + eps)
    S_db = 10.0 * torch.log10(S + eps)                 # [C, T]
    # downsample to T_target (integer indexing keeps it simple)
    T = S_db.shape[-1]
    idx = torch.linspace(0, T - 1, steps=T_target, device=S_db.device).long()
    Sd = S_db[..., idx]                                # [C, T_target]
    # normalize shape (zero at t=0, z-score over time)
    Sd = Sd - Sd[..., :1]
    Sd = Sd / Sd.float().std(dim=-1, keepdim=True).clamp_min(1e-6)
    return Sd

# samplers.py

class WindowBatchSampler(BatchSampler):
    """
    Groups pre-existing per-slice items (with id & t_idx) into fixed-length windows.
    - Dataset stays unchanged.
    - Batch size remains your current "slices per batch" (e.g., 2048).
    - Each batch is a concatenation of many windows (e.g., 128 windows × 16 slices).
    """
    def __init__(self, dataset, batch_size: int, window_size: int,
                 shuffle: bool = True, drop_last: bool = True,
                 get_id_fn=None, get_tidx_fn=None, seed=127):
        """
        dataset: existing dataset yielding single slices
        batch_size: total #slices per batch (e.g., 2048)
        window_size: T_w (e.g., 16)
        get_id_fn, get_tidx_fn: callables (ds, i)-> id, t_idx. If None, we use ds.get_id(i)/ds.get_time_index(i).
        """
        self.ds = dataset
        self.B = batch_size
        self.Tw = max(int(window_size), 1)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch_offset = 0
        self.epoch = 0
        self.shuffle_start = False
        self.seed = seed
        self._get_id = get_id_fn or (lambda ds, i: ds.get_id(i))
        self._get_t  = get_tidx_fn or (lambda ds, i: ds.get_time_index(i))

        # Build {id: [(t_idx, global_idx), ...]}
        self._by_id = {}
        for i in range(len(self.ds)):
            rid = self._get_id(self.ds, i)
            t   = self._get_t(self.ds, i)
            self._by_id.setdefault(rid, []).append((t, i))
        for rid in self._by_id:
            self._by_id[rid].sort(key=lambda x: x[0])

        self._windows_all = []  # list[list[global_idx]] each of length Tw
        self._rebuild_windows()

    def set_epoch(self, epoch: int, shuffle_start: bool = False):
        self.epoch = int(epoch)
        self.shuffle_start = bool(shuffle_start)
        if self.shuffle_start and self.Tw > 1:
            # derive offset from (seed + epoch) so it's reproducible per epoch
            rng = random.Random(self.seed + self.epoch)
            self.epoch_offset = rng.randrange(0, self.Tw)
        else:
            # keep window starts fixed → same slices every epoch
            self.epoch_offset = 0
        self._rebuild_windows()


    def _rebuild_windows(self):
        self._windows_all.clear()
        off, Tw = self.epoch_offset, self.Tw
        for _, seq in self._by_id.items():
            idx_list = [i for (_, i) in seq]
            T_total  = len(idx_list)
            t0 = off
            while t0 < T_total:
                win = idx_list[t0 : min(t0 + Tw, T_total)]
                self._windows_all.append(list(win))
                t0 += Tw

        # shuffle order ONLY (windows themselves already built above)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(self._windows_all)

    def __iter__(self):
        batch = []
        for win in self._windows_all:
            if len(batch) + len(win) > self.B:
                if len(batch) == self.B or not self.drop_last:
                    yield batch
                batch = []
            batch.extend(win)
            if len(batch) == self.B:
                yield batch
                batch = []
        if (not self.drop_last) and batch:
            yield batch

    def __len__(self):
        n_slices = sum(len(w) for w in self._windows_all)
        return n_slices // self.B

# collate_windowptr.py

# def collate_with_windowptr_varlen(samples, window_size: int = 20):
#     """
#     Collate a list of dataset samples (dicts) into one batch dict.
#     Ensures that `id`, `t_idx`, and `time_query` come out as tensors.
#     """
#     # 1. Gather lists per key
#     batch = {}
#     for key in samples[0].keys():
#         batch[key] = [s[key] for s in samples]

#     # 2. Convert certain fields to tensors
#     for k in ["id", "t_idx", "time_query"]:
#         if k in batch:
#             batch[k] = torch.as_tensor(batch[k], dtype=torch.long)

#     # Everything else can be stacked if they’re tensors already
#     for k, v in batch.items():
#         if isinstance(v, list) and isinstance(v[0], torch.Tensor):
#             batch[k] = torch.stack(v, dim=0)

#     return batch

# --- NEW: collate that emits window_ptr for grouped EDC ---
def collate_with_windowptr_varlen(samples, min_len_keep: int = 8):
    """
    Collate for windowed training when each dataset item is a SINGLE STFT slice.

    Inputs:
      samples: list of dicts from the Dataset, each dict must contain at least:
               - 'id'   : int or Tensor scalar (sequence id / utterance id)
               - 't_idx': int or Tensor scalar (frame index within the sequence)
               plus whatever tensors you already use (e.g., 'g_stft', etc.)

    Output:
      batch: dict with usual stacked tensors, plus:
             - 'window_ptr': LongTensor of shape (W, 2) with rows (start_index, length)
                              indexing into the *flat* batch dimension.
    """
    # 0) keep sampler order; do NOT sort or shuffle here
    B = len(samples)
    batch = {}

    # 1) gather lists per key
    for key in samples[0].keys():
        vals = [s[key] for s in samples]
        batch[key] = vals

    # 2) convert scalar-ish fields to tensors
    def _as_1d_long(x):
        return torch.as_tensor(x, dtype=torch.long)

    # Common scalar meta
    for k in ("audio_idx", "time_query"):
        if k in batch:
            batch[k] = _as_1d_long([int(v) if not hasattr(v, "item") else int(v.item()) for v in batch[k]])

    # 3) stack tensor fields (leave non-tensors alone)
    for k, v in list(batch.items()):
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            # Stack along a new leading dim -> (B_flat, ...)
            batch[k] = torch.stack(v, dim=0)

    # 4) rebuild windows from consecutive (id, t_idx)
    ids  = batch["audio_idx"].tolist()
    tids = batch["time_query"].tolist()

    window_ptr = []
    s, L = 0, 1
    for i in range(1, B):
        same_seq   = (ids[i] == ids[i-1])
        consec_time = (tids[i] == tids[i-1] + 1)
        if same_seq and consec_time:
            L += 1
        else:
            if L >= min_len_keep:
                window_ptr.append((s, L))
            s, L = i, 1
    # flush last
    if L >= min_len_keep:
        window_ptr.append((s, L))

    if len(window_ptr) == 0:
        # fall back: keep at least one tiny window to avoid empty batches
        window_ptr.append((0, min(B, max(1, min_len_keep))))

    batch["window_ptr"] = torch.as_tensor(window_ptr, dtype=torch.long)  # (W,2)

    return batch


# edc_losses.py

def _schroeder_curve(E_t):
    """
    E_t: (..., T) non-negative energy over time
    returns: (..., T) log Schroeder curve normalized to start at 0 dB
    """
    eps = 1e-12
    Sd = torch.flip(torch.cumsum(torch.flip(E_t, dims=[-1]), dim=-1), dims=[-1])  # reverse cumsum
    Sd = torch.log(Sd + eps)
    Sd = Sd - Sd[..., :1]
    return Sd

@torch.no_grad()
def _zscore_inplace(x, mask=None):
    # NOTE: we do not need grads for normalization stats
    eps = 1e-8
    if mask is None:
        m = x.mean(dim=-1, keepdim=True)
        v = x.var(dim=-1, unbiased=False, keepdim=True)
    else:
        m = (x * mask).sum(dim=-1, keepdim=True) / (mask.sum(dim=-1, keepdim=True) + eps)
        v = ((x - m) ** 2 * mask).sum(dim=-1, keepdim=True) / (mask.sum(dim=-1, keepdim=True) + eps)
    return (x - m) / (v.sqrt() + eps)

def edc_loss_full(pred_mag: torch.Tensor, gt_mag: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    pred_mag, gt_mag: (B,F,T) in *linear magnitude* (or keep consistent transform on both)
    valid_mask: (B,T) bool, True=valid (optional)
    Implements: skip if T<2; ignore invalid frames; and if padded tail > 4 -> drop that sample from EDC.
    """
    B, F, T = pred_mag.shape
    if T < 2:
        return pred_mag.new_zeros(())

    losses = []
    for b in range(B):
        pm = pred_mag[b]  # (F,T)
        gm = gt_mag[b]    # (F,T)

        if valid_mask is not None:
            m = valid_mask[b].to(pm.dtype)  # (T,)
            # determine padding tail length
            # tail = trailing False count
            inv = (~valid_mask[b]).to(torch.int32)
            pad_len = int(inv.flip(-1).cumprod(-1).sum().item())  # trailing falses
            if pad_len > 4:
                # Discard overly padded item
                continue
            # zero invalid frames’ energy
            m = m[None, :]
            pm = pm * m
            gm = gm * m

        # energy over freq: (T,)
        E_pred = (pm ** 2).sum(dim=0)
        E_gt   = (gm ** 2).sum(dim=0)

        if E_pred.numel() < 2:
            continue

        Sd_p = _schroeder_curve(E_pred)
        Sd_g = _schroeder_curve(E_gt)

        # z-score per item (optional but stabilizes scale)
        Sd_p = _zscore_inplace(Sd_p)
        Sd_g = _zscore_inplace(Sd_g)

        losses.append(torch.mean((Sd_p - Sd_g) ** 2))

    if not losses:
        return pred_mag.new_zeros(())
    return torch.stack(losses).mean()

def edc_loss_grouped_from_slices(
    pred_mag_flat: torch.Tensor,
    gt_mag_flat: torch.Tensor,
    window_ptr: List[Tuple[int, int]],
    allow_small_windows: bool = False,
    pad_discard_thresh: int = 4,   # keep if pad_len <= 4
    min_len_keep: Optional[int] = 10  # e.g., 8 means drop L < 8
) -> torch.Tensor:
    eps = 1e-12
    losses = []
    # more robust nominal Tw: use the max length observed (or pass from collate)
    Tw_nominal = max((l for _, l in window_ptr), default=0)

    for (s, L) in window_ptr:
        if min_len_keep is not None and L < min_len_keep and not allow_small_windows:
            # explicit "group too short" filter
            continue

        pad_len = max(0, Tw_nominal - L)
        if pad_len > pad_discard_thresh:
            continue
        if L < 2 and not allow_small_windows:
            continue

        pm = pred_mag_flat[s:s+L].transpose(0, 1)  # (F,L)
        gm = gt_mag_flat[s:s+L].transpose(0, 1)    # (F,L)
        
        E_pred = (pm ** 2).sum(dim=0)
        E_gt   = (gm ** 2).sum(dim=0)

        Sd_p = _zscore_inplace(_schroeder_curve(E_pred))
        Sd_g = _zscore_inplace(_schroeder_curve(E_gt))

        losses.append(torch.mean((Sd_p - Sd_g) ** 2))

    if not losses:
        return pred_mag_flat.new_zeros(())
    return torch.stack(losses).mean()


def set_global_seed(seed: int = 1234):
    """Force deterministic runs (same shuffling, init, etc)."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
