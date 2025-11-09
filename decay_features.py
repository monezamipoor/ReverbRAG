# decay_features.py
# CPU-only (NumPy + Numba) feature builder for reference bank
# Features per band (B): [late_slope, knee_time_ms, edt_ms, early_energy]
# All computations in float64; final normalized features in float32.

import numpy as np
import numba
from tqdm import tqdm

def _make_linear_bands(F: int, B: int):
    # contiguous equal-width bins; last band absorbs remainder
    edges = np.linspace(0, F, B+1, dtype=np.int64)
    idxs = [np.arange(edges[b], edges[b+1], dtype=np.int64) for b in range(B)]
    return idxs

@numba.njit(cache=True, fastmath=True)
def _ols_slope(x, y):
    n = x.shape[0]
    sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
    for i in range(n):
        xi = x[i]; yi = y[i]
        sx += xi; sy += yi; sxx += xi*xi; sxy += xi*yi
    den = (n * sxx - sx * sx)
    if den == 0.0:
        return 0.0, 0.0
    a = (n * sxy - sx * sy) / den   # slope
    b = (sy - a * sx) / n           # intercept
    return a, b

@numba.njit(cache=True, fastmath=True)
def _piecewise_knee(x, y, k_min, k_max):
    # brute-force 2-seg least squares knee index in [k_min, k_max]
    best_k = k_min
    best_sse = 1e308
    for k in range(k_min, k_max+1):
        # fit [0..k] and [k..N-1]
        a1,b1 = _ols_slope(x[:k+1], y[:k+1])
        a2,b2 = _ols_slope(x[k:],   y[k:])
        sse = 0.0
        for i in range(k+1):
            di = y[i] - (a1*x[i] + b1)
            sse += di*di
        for i in range(k, x.shape[0]):
            di = y[i] - (a2*x[i] + b2)
            sse += di*di
        if sse < best_sse:
            best_sse = sse
            best_k = k
    return best_k

@numba.njit(cache=True, fastmath=True)
def _schroeder_db(E):
    # E: (T,) non-negative energy
    T = E.shape[0]
    S = np.empty_like(E)
    # reverse cumsum
    acc = 0.0
    for i in range(T-1, -1, -1):
        acc += E[i]
        S[i] = acc
    if S[0] <= 1e-12:
        S = S + 1e-12
    S = S / S[0]
    out = np.empty_like(S)
    for i in range(T):
        out[i] = 10.0 * np.log10(S[i] + 1e-12)
    return out

@numba.njit(cache=True, fastmath=True)
def _interp_time_at_drop(l_db, drop_db):
    # earliest t where l_db <= l0 - drop_db ; linear interp
    T = l_db.shape[0]
    l0 = l_db[0]
    target = l0 - drop_db
    for t in range(1, T):
        if l_db[t] <= target:
            # interpolate between t-1 and t
            y1 = l_db[t-1]; y2 = l_db[t]
            if y1 == y2:
                return t
            frac = (target - y1) / (y2 - y1)
            if frac < 0.0: frac = 0.0
            if frac > 1.0: frac = 1.0
            return (t-1) + frac
    return float(T-1)

def _band_energy_from_logmag(S_log_band):
    # S_log_band: (T,) log-mag (natural log) of a single band (already pooled)
    # Convert back to magnitude then energy
    # NeRAF convention: log(mag + 1e-3) -> mag = exp(logmag) - 1e-3
    mag = np.exp(S_log_band) - 1e-3
    mag[mag < 0.0] = 0.0
    return mag*mag

def _pool_bands_logmag(S_log_FT, bands):
    # S_log_FT: (F,T), bands: list of arrays of freq indices
    F,T = S_log_FT.shape
    B = len(bands)
    S_band = np.zeros((B, T), dtype=np.float64)
    for b in range(B):
        idx = bands[b]
        # average linear magnitude within band, then convert back to log-mag
        lin = np.exp(S_log_FT[idx, :]) - 1e-3
        lin[lin < 0.0] = 0.0
        m = lin.mean(axis=0)  # (T,)
        S_band[b, :] = np.log(m + 1e-12)
    return S_band

def compute_decay_features_for_ref(S_log_FT, num_bands: int, hop_ms: float):
    """
    S_log_FT: (F,T) log-magnitude STFT
    Returns: feats_band (B,4) in float64: [slope_db_per_ms, knee_ms, edt_ms, early_energy]
    """
    F,T = S_log_FT.shape
    bands = _make_linear_bands(F, num_bands)
    S_band = _pool_bands_logmag(S_log_FT, bands)   # (B,T)

    B = num_bands
    feats = np.zeros((B, 4), dtype=np.float64)

    t_ms = np.arange(T, dtype=np.float64) * hop_ms

    for b in range(B):
        E = _band_energy_from_logmag(S_band[b])          # (T,)
        ldb = _schroeder_db(E)                           # (T,) dB

        # (1) Late slope: OLS on last half
        t1 = T//2
        x = t_ms[t1:]
        y = ldb[t1:]
        if x.shape[0] >= 2:
            slope, intercept = _ols_slope(x, y)
        else:
            slope, intercept = 0.0, ldb[0]
        slope_db_per_ms = slope

        # (2) Knee time (2-segment OLS) on 10%..70%
        kmin = max(1, int(0.1*T))
        kmax = max(kmin+1, int(0.7*T))
        kidx = _piecewise_knee(t_ms, ldb, kmin, min(kmax, T-2))
        knee_ms = t_ms[kidx]

        # (3) EDT: 0–10 dB
        t_star = _interp_time_at_drop(ldb, 10.0)
        edt_ms = (t_star - 0.0) * hop_ms  # reference is first frame

        # (4) Early energy 0–80 ms
        end_early = 80.0
        last_i = int(np.floor(end_early / hop_ms))
        if last_i < 1: last_i = 1
        if last_i >= T: last_i = T-1
        early_e = E[:last_i+1].sum()

        feats[b, 0] = slope_db_per_ms
        feats[b, 1] = knee_ms
        feats[b, 2] = edt_ms
        feats[b, 3] = early_e

    return feats

def build_ref_decay_features_bank(ref_bank_logmag, num_bands: int, hop_ms: float):
    """
    ref_bank_logmag: numpy array [R, 1, F, T] in log-magnitude
    Returns:
      feats_norm: float32 [R, B, 4]
      stats: dict {means: [4], stds: [4]}
    """
    R, C, F, T = ref_bank_logmag.shape
    assert C == 1
    feats = np.zeros((R, num_bands, 4), dtype=np.float64)

    # Progress bar for feature computation
    for r in tqdm(range(R), desc="Building decay features (CPU)", unit="ref"):
        S_log = ref_bank_logmag[r, 0]
        feats[r] = compute_decay_features_for_ref(S_log, num_bands=num_bands, hop_ms=hop_ms)

    # z-score per feature across all refs and bands
    feats_reshaped = feats.reshape(R * feats.shape[1], 4)
    means = feats_reshaped.mean(axis=0)
    stds = feats_reshaped.std(axis=0) + 1e-12
    feats_norm = (feats - means[None, None, :]) / stds[None, None, :]

    return feats_norm.astype(np.float32), {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "num_bands": int(num_bands),
        "hop_ms": float(hop_ms),
    }
