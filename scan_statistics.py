"""
scan_statistics_runner.py
=========================
Single-file modular implementation of the CosmicAI NWKR scan statistics pipeline.

Sections
--------
  1.  Imports & constants
  2.  Public types          (Input, Output)
  3.  Atmospheric detection (load_transmission, detect_atm_ranges)
  4.  SR / superresolution  (_sr_factor, _superresolve, ...)
  5.  Kernel primitives     (_truncated_kernel_vector, _calculate_gaussian_sra_trunc)
  6.  Numba JIT state       (_is_inside, _nin_din_*, _sse_out_*, _buf_*)
  7.  Core scan             (_scan_single_row)
  8.  Public API            (compute_scan_statistics_scores)
  9.  CLI runner            (main)

Library usage
-------------
    from scan_statistics_runner import compute_scan_statistics_scores, Input, Output

    results = compute_scan_statistics_scores({
        "key": Input(amplitude=amp, frequency=freq, flag_array=flags,
                     atm_ranges=[(s1,e1), (s2,e2)])
    })
    out = results["key"]["masked"]
    print(out.score, out.win_start, out.win_end)

CLI usage
---------
    python scan_statistics_runner.py \
        --amplitude  amp.npy \
        --frequency  freq.npy \
        --interference full_spectrum.gzip \
        --key my_baseline

    # with optional flag array:
    python scan_statistics_runner.py \
        --amplitude  amp.npy \
        --frequency  freq.npy \
        --flag-array flags.npy \
        --interference full_spectrum.gzip \
        --key my_baseline

Arguments
---------
    --amplitude     (required) .npy file, 1-D float64 amplitude array.
    --frequency     (required) .npy file, 1-D float64 frequency array in GHz.
    --flag-array    (optional) .npy file, 1-D bool array (True = flagged).
                    Defaults to all-False if omitted.
    --interference  (optional) Transmission parquet/gzip file with columns
                    "Frequency (GHz)" and "Transmission (%)".
                    Used to detect atmospheric absorption line ranges.
                    If omitted, atm_ranges is empty.
    --key           Label for this spectrum (default: "spectrum").
    --kernel        "gaussian" (default) or "laplace".
    --log-level     Logging verbosity (default: INFO).
"""

from __future__ import annotations

# =============================================================================
# 1. Imports & constants
# =============================================================================

import argparse
import logging
import math
import pandas as pd
from itertools import groupby
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, TypeAlias

import numpy as np
from numba import njit
from scipy.signal import find_peaks, peak_widths

# Pipeline constants — match cosmicai/config.py
_REF_FREQ:           float = 0.0625   # GHz reference spectral-line channel width
_SUPER_RESOLVE_BASE: int   = 450      # L0 for SR factor computation
_BUFFER_COEFF:       int   = 20       # BUFFER = n // _BUFFER_COEFF
_DEFAULT_KERNEL:     str   = "gaussian"

logger = logging.getLogger(__name__)


# =============================================================================
# 2. Public types
# =============================================================================

class Input(NamedTuple):
    """
    Input data for a single bandpass calibration spectrum.

    Parameters
    ----------
    amplitude  : 1-D float64 array of amplitude values.
    frequency  : 1-D float64 array of frequency values (GHz).
    flag_array : 1-D bool array; True = flagged (excluded from both scans).
    atm_ranges : List of (start, end) inclusive channel index pairs marking
                 atmospheric absorption lines.  Excluded from the *masked*
                 scan only — the unmasked scan sees them.
    """
    amplitude:  np.ndarray
    frequency:  np.ndarray
    flag_array: np.ndarray
    atm_ranges: list = []


class Output(NamedTuple):
    """Scan statistic result for one scan mode."""
    score:     float
    win_start: int
    win_end:   int


ScanMode:   TypeAlias = Literal["masked", "unmasked", "fixed"]
ScanResult: TypeAlias = Dict[ScanMode, Output]


# =============================================================================
# 3. Atmospheric interference detection
# =============================================================================

def load_transmission(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the atmospheric transmission table from a parquet file.
    The file extension may be .parquet or .gzip (both are parquet format).
    Expected columns: "Frequency (GHz)", "Transmission (%)".

    Returns
    -------
    trans_freqs : 1-D float64 array of frequency values (GHz), sorted ascending.
    trans_vals  : 1-D float64 array of transmission percentages (0–100).
    """
    df = pd.read_parquet(path)
    return df["Frequency (GHz)"].to_numpy(), df["Transmission (%)"].to_numpy()


def _match_transmission(
    freq_array:  np.ndarray,
    trans_freqs: np.ndarray,
    trans_vals:  np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour interpolation of transmission onto freq_array."""
    idxs    = np.searchsorted(trans_freqs, freq_array)
    idxs[idxs == len(trans_freqs)] = len(trans_freqs) - 1
    left    = np.maximum(idxs - 1, 0)
    right   = idxs
    dl      = np.abs(freq_array - trans_freqs[left])
    dr      = np.abs(trans_freqs[right] - freq_array)
    nearest = np.where(dl <= dr, left, right)
    return trans_vals[nearest]


def detect_atm_ranges(
    freq_array:  np.ndarray,
    trans_freqs: np.ndarray,
    trans_vals:  np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Detect atmospheric absorption line channel ranges.

    Replicates the find_peaks logic in io_preprocess.load_data_by_length:
      1. Match transmission to freq_array via nearest-neighbour.
      2. Find troughs in transmission (prominence > 1%).
      3. Measure trough width at 75% relative height.
      4. Convert frequency-width bounds to channel index pairs.

    Parameters
    ----------
    freq_array  : Spectrum frequency axis (GHz), length n.
    trans_freqs : Transmission table frequency axis (GHz).
    trans_vals  : Transmission table values (%).

    Returns
    -------
    List of (start_idx, end_idx) inclusive channel ranges in freq_array coords.
    """
    trans = _match_transmission(freq_array, trans_freqs, trans_vals)

    troughs, _ = find_peaks(-trans, prominence=1)
    if len(troughs) == 0:
        return []

    _, _, left_ips, right_ips = peak_widths(-trans, troughs, rel_height=0.75)

    left_freqs  = np.interp(left_ips,  np.arange(len(freq_array)), freq_array)
    right_freqs = np.interp(right_ips, np.arange(len(freq_array)), freq_array)
    widths_freq = right_freqs - left_freqs

    trough_freqs  = freq_array[troughs]
    trough_ranges = np.column_stack(
        (trough_freqs - widths_freq / 2.0, trough_freqs + widths_freq / 2.0)
    )

    result: List[Tuple[int, int]] = []
    for start_f, end_f in trough_ranges:
        s_idx = int(np.abs(freq_array - start_f).argmin())
        e_idx = int(np.abs(freq_array - end_f).argmin())
        result.append((s_idx, e_idx))

    return result


# =============================================================================
# 4. Superresolution helpers
# =============================================================================

def _sr_factor(n: int) -> int:
    """
    Compute the block-average downsampling factor for an array of length n.
    Matches cosmicai.superres.sr_factor(L, r=2, q=2).

    Formula: SR = 2^ceil(log2(ceil((n+1)/450)))
    Example: n=1920 → SR=8  (1920 channels → 240 after downsampling)
    """
    s  = math.ceil((n + 1) / _SUPER_RESOLVE_BASE)
    kk = math.ceil(math.log(s, 2)) if s > 1 else 0
    return 2 ** kk


def _superresolve(x: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample: groups every `factor` consecutive channels and
    averages them, producing an array of length n // factor.
    Matches cosmicai.superres.superresolve applied to a single row."""
    if factor <= 1:
        return x.copy()
    n_blk = len(x) // factor
    return x[: n_blk * factor].reshape(n_blk, factor).mean(axis=1)


def _superresolve_freq(freq: np.ndarray, factor: int) -> np.ndarray:
    """Downsample frequency axis by taking the first element of each block of
    `factor` channels.  Preserves the frequency spacing meaning of the SR array."""
    if factor <= 1:
        return freq.copy()
    n_blk = len(freq) // factor
    return freq[: n_blk * factor].reshape(n_blk, factor)[:, 0]


def _superresolve_ranges(
    ranges: List[Tuple[int, int]], factor: int
) -> List[Tuple[int, int]]:
    """
    Downsample channel index ranges.
    Matches cosmicai.superres.superresolve_ranges (divide by factor, merge).
    """
    if factor <= 1 or not ranges:
        return list(ranges)
    adjusted = sorted({(s // factor, e // factor) for s, e in ranges})
    merged: List[Tuple[int, int]] = []
    for s, e in adjusted:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# =============================================================================
# 5. Kernel primitives
# =============================================================================

def _truncated_kernel_vector(w: float, r: int, kind: str = "gaussian") -> np.ndarray:
    """k[d] for d = 0..r.  Matches cosmicai.kernels.truncated_kernel_vector."""
    d = np.arange(r + 1, dtype=np.float64)
    if kind == "gaussian":
        return np.exp(-(d * d) / (w * w))
    if kind == "laplace":
        return np.exp(-d / max(float(w), 1e-12))
    raise ValueError(f"Unknown kernel kind: {kind!r}")


@njit(cache=True, fastmath=True)
def _calculate_gaussian_sra_trunc(
    array: np.ndarray,
    k:     np.ndarray,
):
    """
    Exact port of cosmicai.scoring.calculate_gaussian_sra_trunc.
    Returns (sra, numer_all, denom_all).
    """
    n = array.shape[0]; r = k.shape[0] - 1; eps = 1e-12
    numer     = np.empty(n, dtype=np.float64)
    denom     = np.empty(n, dtype=np.float64)
    ssr_array = np.empty(n, dtype=np.float64)

    for i in range(n):
        j0 = i - r if i - r >= 0 else 0
        j1 = i + r if i + r <= n - 1 else n - 1
        num = 0.0; den = 0.0
        for j in range(j0, j1 + 1):
            d = i - j
            if d < 0: d = -d
            wgt = k[d]; den += wgt; num += wgt * array[j]
        numer[i] = num; denom[i] = den
        pi = num / (den if den > eps else eps)
        ssr_array[i] = (array[i] - pi) ** 2

    sra = 0.0
    for i in range(n):
        sra += ssr_array[i]

    return sra, numer, denom


# =============================================================================
# 6. Numba JIT state management
# =============================================================================

@njit(cache=True, fastmath=True)
def _is_inside(buf_idxs: np.ndarray, m: int, idx: int) -> bool:
    """Binary search: True if idx is in buf_idxs[0..m-1] (sorted)."""
    lo = 0; hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < idx: lo = mid + 1
        else: hi = mid
    return lo < m and buf_idxs[lo] == idx


@njit(cache=True, fastmath=True)
def _nin_din_init_full(x, idxs_in, k, nin, din):
    """Zero and rebuild nin/din from inside set. O(n + m*r)."""
    n = x.shape[0]; r = k.shape[0] - 1
    for i in range(n): nin[i] = 0.0; din[i] = 0.0
    for p in range(idxs_in.shape[0]):
        j = int(idxs_in[p]); xj = x[j]; i = j
        while i >= 0:
            d = j - i
            if d > r: break
            nin[i] += k[d] * xj; din[i] += k[d]; i -= 1
        i = j + 1
        while i < n:
            d = i - j
            if d > r: break
            nin[i] += k[d] * xj; din[i] += k[d]; i += 1


@njit(cache=True, fastmath=True)
def _nin_din_add(x, nin, din, idx, k):
    """Add idx's contribution to nin/din. O(r)."""
    r = k.shape[0] - 1; n = nin.shape[0]; xj = x[idx]; i = idx
    while i >= 0:
        d = idx - i
        if d > r: break
        nin[i] += k[d] * xj; din[i] += k[d]; i -= 1
    i = idx + 1
    while i < n:
        d = i - idx
        if d > r: break
        nin[i] += k[d] * xj; din[i] += k[d]; i += 1


@njit(cache=True, fastmath=True)
def _nin_din_remove(x, nin, din, idx, k):
    """Remove idx's contribution from nin/din. O(r)."""
    r = k.shape[0] - 1; n = nin.shape[0]; xj = x[idx]; i = idx
    while i >= 0:
        d = idx - i
        if d > r: break
        nin[i] -= k[d] * xj; din[i] -= k[d]; i -= 1
    i = idx + 1
    while i < n:
        d = i - idx
        if d > r: break
        nin[i] -= k[d] * xj; din[i] -= k[d]; i += 1


@njit(cache=True, fastmath=True)
def _sse_out_from_nin_din(x, numer_all, denom_all, nin, din, buf_idxs, m):
    """Exact sse_out recompute from nin/din. O(n)."""
    n = x.shape[0]; eps = 1e-12; sse = 0.0; p = 0
    for i in range(n):
        while p < m and buf_idxs[p] < i: p += 1
        if p < m and buf_idxs[p] == i: continue
        den_out = denom_all[i] - din[i]; num_out = numer_all[i] - nin[i]
        pred    = num_out / den_out if den_out > eps else 0.0
        sse    += (x[i] - pred) ** 2
    return sse


@njit(cache=True, fastmath=True)
def _sse_out_add(x, numer_all, denom_all, nin, din, buf_idxs, m_new,
                  new_idx, k, sse_out):
    """Update sse_out after new_idx moves outside→inside. O(r)."""
    r = k.shape[0] - 1; eps = 1e-12; n = nin.shape[0]; x_new = x[new_idx]
    nin_old = nin[new_idx] - k[0] * x_new; din_old = din[new_idx] - k[0]
    d_old   = denom_all[new_idx] - din_old; n_old = numer_all[new_idx] - nin_old
    sse_out -= (x_new - (n_old / d_old if d_old > eps else 0.0)) ** 2
    for sign in ((-1, new_idx - 1), (1, new_idx + 1)):
        step, start = sign
        i = start
        while 0 <= i < n:
            d = new_idx - i if step == -1 else i - new_idx
            if d > r: break
            if not _is_inside(buf_idxs, m_new, i):
                w = k[d]
                ni_o = nin[i] - w * x_new; di_o = din[i] - w
                do   = denom_all[i] - di_o; no = numer_all[i] - ni_o
                po   = no / do if do > eps else 0.0
                dn   = denom_all[i] - din[i]; nn = numer_all[i] - nin[i]
                pn   = nn / dn if dn > eps else 0.0
                sse_out -= (x[i] - po) ** 2; sse_out += (x[i] - pn) ** 2
            i += step
    return sse_out


@njit(cache=True, fastmath=True)
def _sse_out_remove(x, numer_all, denom_all, nin, din, buf_idxs, m_new,
                     rem_idx, k, sse_out):
    """Update sse_out after rem_idx moves inside→outside. O(r)."""
    r = k.shape[0] - 1; eps = 1e-12; n = nin.shape[0]; x_rem = x[rem_idx]
    dn = denom_all[rem_idx] - din[rem_idx]; nn = numer_all[rem_idx] - nin[rem_idx]
    sse_out += (x_rem - (nn / dn if dn > eps else 0.0)) ** 2
    for sign in ((-1, rem_idx - 1), (1, rem_idx + 1)):
        step, start = sign
        i = start
        while 0 <= i < n:
            d = rem_idx - i if step == -1 else i - rem_idx
            if d > r: break
            if not _is_inside(buf_idxs, m_new, i):
                w = k[d]
                ni_o = nin[i] + w * x_rem; di_o = din[i] + w
                do   = denom_all[i] - di_o; no = numer_all[i] - ni_o
                po   = no / do if do > eps else 0.0
                dn2  = denom_all[i] - din[i]; nn2 = numer_all[i] - nin[i]
                pn   = nn2 / dn2 if dn2 > eps else 0.0
                sse_out -= (x[i] - po) ** 2; sse_out += (x[i] - pn) ** 2
            i += step
    return sse_out


@njit(cache=True, fastmath=True)
def _buf_init(x, idxs_in, k, buf_idxs, buf_num, buf_den):
    """Initialise inside buf/sse_in. O(m*r). Does NOT touch nin/din."""
    m0 = idxs_in.shape[0]; r = k.shape[0] - 1; eps = 1e-12
    for t in range(m0): buf_idxs[t] = idxs_in[t]
    for u in range(m0):
        i = int(buf_idxs[u]); sn = 0.0; sd = 0.0; v = u
        while v >= 0:
            d = i - int(buf_idxs[v])
            if d > r: break
            sn += k[d] * x[int(buf_idxs[v])]; sd += k[d]; v -= 1
        v = u + 1
        while v < m0:
            d = int(buf_idxs[v]) - i
            if d > r: break
            sn += k[d] * x[int(buf_idxs[v])]; sd += k[d]; v += 1
        buf_num[u] = sn; buf_den[u] = sd
    sse_in = 0.0
    for u in range(m0):
        d_ = buf_den[u]; pred = buf_num[u] / d_ if d_ > eps else 0.0
        sse_in += (x[int(buf_idxs[u])] - pred) ** 2
    return m0, float(sse_in)


@njit(cache=True, fastmath=True)
def _buf_add(x, new_idx, k, buf_idxs, buf_num, buf_den, m, sse_in):
    """Add new_idx to inside buf/sse_in. O(r). Does NOT touch nin/din."""
    r = k.shape[0] - 1; eps = 1e-12; x_new = x[new_idx]
    lo = 0; hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < new_idx: lo = mid + 1
        else: hi = mid
    ins = lo
    for t in range(m - 1, ins - 1, -1):
        buf_idxs[t+1] = buf_idxs[t]; buf_num[t+1] = buf_num[t]; buf_den[t+1] = buf_den[t]
    buf_idxs[ins] = new_idx; m_new = m + 1
    v = ins - 1
    while v >= 0:
        d = new_idx - int(buf_idxs[v])
        if d > r: break
        w = k[d]; od = buf_den[v]; op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] += w * x_new; buf_den[v] += w
        nd = buf_den[v]; np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2; v -= 1
    v = ins + 1
    while v < m_new:
        d = int(buf_idxs[v]) - new_idx
        if d > r: break
        w = k[d]; od = buf_den[v]; op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] += w * x_new; buf_den[v] += w
        nd = buf_den[v]; np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2; v += 1
    sn = k[0] * x_new; sd = k[0]; v = ins - 1
    while v >= 0:
        d = new_idx - int(buf_idxs[v])
        if d > r: break
        sn += k[d] * x[int(buf_idxs[v])]; sd += k[d]; v -= 1
    v = ins + 1
    while v < m_new:
        d = int(buf_idxs[v]) - new_idx
        if d > r: break
        sn += k[d] * x[int(buf_idxs[v])]; sd += k[d]; v += 1
    buf_num[ins] = sn; buf_den[ins] = sd
    sse_in += (x_new - (sn / sd if sd > eps else 0.0)) ** 2
    return m_new, float(sse_in)


@njit(cache=True, fastmath=True)
def _buf_remove(x, rem_idx, k, buf_idxs, buf_num, buf_den, m, sse_in):
    """Remove rem_idx from inside buf/sse_in. O(r). Does NOT touch nin/din."""
    r = k.shape[0] - 1; eps = 1e-12; x_rem = x[rem_idx]
    lo = 0; hi = m
    while lo < hi:
        mid = (lo + hi) >> 1
        if buf_idxs[mid] < rem_idx: lo = mid + 1
        else: hi = mid
    pos = lo
    if pos >= m or buf_idxs[pos] != rem_idx: return m, sse_in
    od = buf_den[pos]; op = buf_num[pos] / od if od > eps else 0.0
    sse_in -= (x_rem - op) ** 2
    v = pos - 1
    while v >= 0:
        d = rem_idx - int(buf_idxs[v])
        if d > r: break
        w = k[d]; od = buf_den[v]; op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] -= w * x_rem; buf_den[v] -= w
        nd = buf_den[v]; np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2; v -= 1
    v = pos + 1
    while v < m:
        d = int(buf_idxs[v]) - rem_idx
        if d > r: break
        w = k[d]; od = buf_den[v]; op = buf_num[v] / od if od > eps else 0.0
        sse_in -= (x[int(buf_idxs[v])] - op) ** 2
        buf_num[v] -= w * x_rem; buf_den[v] -= w
        nd = buf_den[v]; np_ = buf_num[v] / nd if nd > eps else 0.0
        sse_in += (x[int(buf_idxs[v])] - np_) ** 2; v += 1
    for t in range(pos, m - 1):
        buf_idxs[t] = buf_idxs[t+1]; buf_num[t] = buf_num[t+1]; buf_den[t] = buf_den[t+1]
    return m - 1, float(sse_in)


# =============================================================================
# 7. Core scan  (_scan_single_row)
# =============================================================================

def _flag_array_to_ranges(flag_array: np.ndarray) -> List[Tuple[int, int]]:
    """Convert boolean flag array to list of inclusive (start, end) ranges."""
    ranges: List[Tuple[int, int]] = []
    idx = 0
    for val, grp in groupby(flag_array.tolist()):
        length = sum(1 for _ in grp)
        if val:
            ranges.append((idx, idx + length - 1))
        idx += length
    return ranges


def _scan_single_row(
    amplitude:   np.ndarray,
    frequency:   np.ndarray,
    flag_ranges: List[Tuple[int, int]],
    atm_ranges:  List[Tuple[int, int]],
    kernel_kind: str,
) -> ScanResult:
    """
    Full pipeline scan for one spectrum row.

    Steps
    -----
    1. Compute SR_FACTOR and BUFFER.
    2. Downsample amplitude and frequency by SR_FACTOR (block average).
    3. Superresolve flag_ranges and atm_ranges to SR coordinates.
    4. Derive kernel params (w, r) from SR frequency axis.
    5. Compute SRA + numer_all/denom_all on trimmed SR array.
    6. Build valid index sets:
         - valid_masked:   excludes flag_ranges + atm_ranges
         - valid_unmasked: excludes flag_ranges only
    7. Run _varlen_search on each valid set.
    8. Run _fixedlen_sweep on all trimmed channels.
    9. Convert window coords back to original full-row space.

    Window coordinate mapping
    -------------------------
    The scan works in trimmed SR coordinates (0-indexed within row_trimmed).
    Returned windows are converted back to original full-row coordinates:
      trimmed SR → full SR:   i_trim + buf_scan
      full SR    → original:  i_sr   × SR_FACTOR
    """
    x    = amplitude
    freq = frequency
    n    = x.shape[0]
    _BAD = Output(score=-np.inf, win_start=0, win_end=0)

    if n < 4 or len(freq) < 2:
        return {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}

    SR       = _sr_factor(n)
    BUFFER   = n // _BUFFER_COEFF
    buf_scan = BUFFER // SR

    x_sr    = _superresolve(x, SR)
    freq_sr = _superresolve_freq(freq, SR)
    n_sr    = len(x_sr)

    fr_sr  = _superresolve_ranges(flag_ranges, SR)
    atm_sr = _superresolve_ranges(atm_ranges,  SR)

    if len(freq_sr) < 2:
        return {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}

    freq_step = abs(float(freq_sr[1] - freq_sr[0]))
    if freq_step <= 0.0: freq_step = _REF_FREQ
    R           = _REF_FREQ / freq_step
    L_sr        = len(freq_sr)
    w           = int(round(max(3.0, min(R, L_sr / 16.0))))
    range_cap   = 3 * w
    window_bins = int(math.floor(R)) + 1
    k_vec       = _truncated_kernel_vector(float(w), range_cap, kind=kernel_kind)

    row_trimmed = x_sr[buf_scan: n_sr - buf_scan]
    n_trimmed   = row_trimmed.shape[0]
    if n_trimmed <= 2:
        return {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}

    sra, numer_all, denom_all = _calculate_gaussian_sra_trunc(row_trimmed, k_vec)
    sra = max(float(sra), 1e-12)

    mask_masked   = np.ones(n_trimmed, dtype=np.bool_)
    mask_unmasked = np.ones(n_trimmed, dtype=np.bool_)

    for ranges, masks in [
        (fr_sr,  [mask_masked, mask_unmasked]),
        (atm_sr, [mask_masked]),
    ]:
        for s, e in ranges:
            s0 = max(int(s) - buf_scan, 0)
            e0 = min(int(e) - buf_scan, n_trimmed - 1)
            if s0 <= e0:
                for mask in masks:
                    mask[s0:e0 + 1] = False

    all_trimmed    = np.arange(n_trimmed, dtype=np.int64)
    valid_masked   = all_trimmed[mask_masked]
    valid_unmasked = all_trimmed[mask_unmasked]

    REFRESH = max(1, range_cap)

    def _varlen_search(valid: np.ndarray) -> Tuple[int, int, float]:
        best_sc = 0.0; best_i = best_j = 0
        n_valid = valid.shape[0]
        if n_valid < 2:
            return (best_i + buf_scan) * SR, (best_j + buf_scan) * SR, best_sc

        cap      = range_cap + 2
        buf_idxs = np.empty(cap, dtype=np.int64)
        buf_num  = np.empty(cap, dtype=np.float64)
        buf_den  = np.empty(cap, dtype=np.float64)
        nin      = np.zeros(n_trimmed, dtype=np.float64)
        din      = np.zeros(n_trimmed, dtype=np.float64)

        carry_valid = False; carry_left_idx = -1; steps = 0

        for pos_i in range(n_valid):
            i = int(valid[pos_i])
            if pos_i + 1 < n_valid and valid[pos_i + 1] != i + 1:
                carry_valid = False; continue

            use_carry = carry_valid and carry_left_idx == i - 1
            if use_carry:
                _nin_din_remove(row_trimmed, nin, din, i - 1, k_vec)
            else:
                for ii in range(n_trimmed): nin[ii] = 0.0; din[ii] = 0.0
                steps = 0

            g_init = False; m = 0; sse_in = 0.; sse_out = 0.
            max_k  = min(pos_i + range_cap, n_valid - 1)

            for kk in range(pos_i + 1, max_k + 1):
                if valid[kk] != valid[kk - 1] + 1: break
                j = int(valid[kk])

                if not g_init:
                    _nin_din_init_full(row_trimmed,
                                       np.array([i, j], dtype=np.int64),
                                       k_vec, nin, din)
                    m, sse_in = _buf_init(row_trimmed,
                                          np.array([i, j], dtype=np.int64),
                                          k_vec, buf_idxs, buf_num, buf_den)
                    sse_out = _sse_out_from_nin_din(
                        row_trimmed, numer_all, denom_all, nin, din, buf_idxs, m)
                    steps = 0; g_init = True
                else:
                    _nin_din_add(row_trimmed, nin, din, j, k_vec)
                    m, sse_in = _buf_add(row_trimmed, j, k_vec,
                                         buf_idxs, buf_num, buf_den, m, sse_in)
                    sse_out = _sse_out_add(row_trimmed, numer_all, denom_all,
                                           nin, din, buf_idxs, m, j, k_vec, sse_out)
                    steps += 1
                    if steps >= REFRESH:
                        sse_out = _sse_out_from_nin_din(
                            row_trimmed, numer_all, denom_all, nin, din, buf_idxs, m)
                        steps = 0

                sc = 1.0 - (sse_in + sse_out) / sra
                if sc > best_sc:
                    best_sc = sc; best_i, best_j = i, j

            if g_init:
                carry_valid = True; carry_left_idx = int(i)
            else:
                carry_valid = False

        return (best_i + buf_scan) * SR, (best_j + buf_scan) * SR, float(best_sc)

    def _fixedlen_sweep() -> Tuple[int, int, float]:
        best_sc = 0.; best_i = 0; best_j = max(window_bins - 1, 0)
        if window_bins <= 0 or window_bins > n_trimmed:
            return best_i * SR, best_j * SR, best_sc

        cap      = window_bins + 1
        buf_idxs = np.empty(cap, dtype=np.int64)
        buf_num  = np.empty(cap, dtype=np.float64)
        buf_den  = np.empty(cap, dtype=np.float64)
        nin      = np.zeros(n_trimmed, dtype=np.float64)
        din      = np.zeros(n_trimmed, dtype=np.float64)
        m = 0; sse_in = 0.; sse_out = 0.; g_init = False; steps = 0

        for i in range(n_trimmed - window_bins + 1):
            j      = i + window_bins - 1
            inside = np.arange(i, i + window_bins, dtype=np.int64)
            if not g_init:
                _nin_din_init_full(row_trimmed, inside, k_vec, nin, din)
                m, sse_in = _buf_init(row_trimmed, inside, k_vec,
                                      buf_idxs, buf_num, buf_den)
                sse_out = _sse_out_from_nin_din(
                    row_trimmed, numer_all, denom_all, nin, din, buf_idxs, m)
                steps = 0; g_init = True
            else:
                rem = np.int64(i - 1); add = np.int64(j)
                _nin_din_remove(row_trimmed, nin, din, rem, k_vec)
                m, sse_in = _buf_remove(row_trimmed, rem, k_vec,
                                        buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = _sse_out_remove(row_trimmed, numer_all, denom_all,
                                          nin, din, buf_idxs, m, rem, k_vec, sse_out)
                _nin_din_add(row_trimmed, nin, din, add, k_vec)
                m, sse_in = _buf_add(row_trimmed, add, k_vec,
                                     buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = _sse_out_add(row_trimmed, numer_all, denom_all,
                                       nin, din, buf_idxs, m, add, k_vec, sse_out)
                steps += 1
                if steps >= REFRESH:
                    sse_out = _sse_out_from_nin_din(
                        row_trimmed, numer_all, denom_all, nin, din, buf_idxs, m)
                    steps = 0

            sc = 1.0 - (sse_in + sse_out) / sra
            if sc > best_sc:
                best_sc = sc; best_i = i; best_j = j

        return (best_i + buf_scan) * SR, (best_j + buf_scan) * SR, float(best_sc)

    wm_s, wm_e, sm = _varlen_search(valid_masked)
    wu_s, wu_e, su = _varlen_search(valid_unmasked)
    wf_s, wf_e, sf = _fixedlen_sweep()

    return {
        "masked":   Output(score=sm, win_start=wm_s, win_end=wm_e),
        "unmasked": Output(score=su, win_start=wu_s, win_end=wu_e),
        "fixed":    Output(score=sf, win_start=wf_s, win_end=wf_e),
    }


# =============================================================================
# 8. Public API  (compute_scan_statistics_scores)
# =============================================================================

def compute_scan_statistics_scores(
    keyed_input: Dict[str, Input],
    *,
    kernel_kind: str = _DEFAULT_KERNEL,
) -> Dict[str, ScanResult]:
    """
    Compute NWKR scan statistics for a batch of spectra.

    Replicates the full CosmicAI pipeline:
      1. Downsample by SR_FACTOR = 2^ceil(log2(ceil((n+1)/450)))
      2. Scan on downsampled array with buffer = (n//20) // SR_FACTOR
      3. Score = 1 − (SSE_in + SSE_out) / SRA  using nin/din formulation
      4. Returned window coordinates are in original full-row space

    Parameters
    ----------
    keyed_input : Dict[str, Input]
        Mapping of unique key → Input namedtuple.
    kernel_kind : str
        "gaussian" (default) or "laplace".

    Returns
    -------
    Dict[str, ScanResult]
        Two-level dict: key → {"masked": Output, "unmasked": Output, "fixed": Output}.

    Notes
    -----
    flag_array vs atm_ranges
        - flag_array  : channels excluded from BOTH masked and unmasked scans.
        - atm_ranges  : atmospheric line channels excluded from masked scan only.
        - If flag_array is all False AND atm_ranges is empty, edge exclusion is
          handled by the buffer trim alone.
    """
    if not keyed_input:
        return {}

    _BAD        = Output(score=-np.inf, win_start=0, win_end=0)
    _BAD_RESULT: ScanResult = {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}
    results: Dict[str, ScanResult] = {}

    for key, inp in keyed_input.items():
        try:
            amplitude  = np.asarray(inp.amplitude,  dtype=np.float64)
            frequency  = np.asarray(inp.frequency,  dtype=np.float64)
            flag_array = np.asarray(inp.flag_array, dtype=bool)
        except Exception as exc:
            logger.warning("[%s] Could not coerce arrays: %s", key, exc)
            results[key] = _BAD_RESULT; continue

        if amplitude.ndim != 1 or frequency.ndim != 1 or flag_array.ndim != 1:
            raise ValueError(
                f"[{key}] All arrays must be 1-D. "
                f"Got shapes amplitude={amplitude.shape}, "
                f"frequency={frequency.shape}, flag_array={flag_array.shape}."
            )
        if not (amplitude.shape == frequency.shape == flag_array.shape):
            raise ValueError(
                f"[{key}] All arrays must have equal length. "
                f"Got amplitude={amplitude.shape[0]}, "
                f"frequency={frequency.shape[0]}, flag_array={flag_array.shape[0]}."
            )

        if np.all(amplitude == 0.0):
            logger.debug("[%s] Skipped: all-zero amplitude.", key)
            results[key] = _BAD_RESULT; continue
        if flag_array.all():
            logger.debug("[%s] Skipped: all channels flagged.", key)
            results[key] = _BAD_RESULT; continue

        flag_ranges = _flag_array_to_ranges(flag_array) if flag_array.any() else []
        atm_ranges  = list(inp.atm_ranges) if inp.atm_ranges else []

        logger.debug("[%s] n=%d  SR=%d  BUF=%d  n_flag=%d  n_atm=%d  kernel=%s",
                     key, len(amplitude), _sr_factor(len(amplitude)),
                     len(amplitude) // _BUFFER_COEFF,
                     len(flag_ranges), len(atm_ranges), kernel_kind)

        try:
            results[key] = _scan_single_row(
                amplitude, frequency, flag_ranges, atm_ranges, kernel_kind)
        except Exception as exc:
            logger.warning("[%s] Scan raised: %s", key, exc, exc_info=True)
            results[key] = _BAD_RESULT

    return results


# =============================================================================
# 9. CLI runner
# =============================================================================

def _flag_ranges_to_array(flag_ranges, n: int) -> np.ndarray:
    """Convert flag_ranges list-of-tuples to boolean array."""
    arr = np.zeros(n, dtype=bool)
    if flag_ranges is None:
        return arr
    try:
        for rng in flag_ranges:
            if rng is None: continue
            s, e = int(rng[0]), int(rng[1])
            s = max(0, s); e = min(n - 1, e)
            if s <= e: arr[s:e + 1] = True
    except (TypeError, ValueError, IndexError):
        pass
    return arr


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run NWKR scan statistics on a single spectrum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--amplitude",     required=True,
                   help="Path to .npy file containing 1-D amplitude array.")
    p.add_argument("--frequency",     required=True,
                   help="Path to .npy file containing 1-D frequency array (GHz).")
    p.add_argument("--flag-array",    default=None,
                   help="Path to .npy file containing 1-D boolean flag array "
                        "(True = flagged). Optional; defaults to all-False.")
    p.add_argument("--interference",  default=None,
                   help="Transmission table parquet/gzip for atm detection "
                        "(columns: 'Frequency (GHz)', 'Transmission (%)'). Optional.")
    p.add_argument("--key",           default="spectrum",
                   help="Label for this spectrum in the output.")
    p.add_argument("--kernel",        default="gaussian",
                   choices=["gaussian", "laplace"])
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> Dict[str, ScanResult]:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    # --- load arrays ---
    amp  = np.load(args.amplitude).astype(np.float64)
    freq = np.load(args.frequency).astype(np.float64)
    n    = len(amp)

    if args.flag_array:
        flag_array = np.load(args.flag_array).astype(bool)
    else:
        flag_array = np.zeros(n, dtype=bool)

    # --- atmospheric detection ---
    atm_ranges: List[Tuple[int, int]] = []
    if args.interference:
        trans_freqs, trans_vals = load_transmission(args.interference)
        atm_ranges = detect_atm_ranges(freq, trans_freqs, trans_vals)

    # --- build input and run ---
    keyed_input: Dict[str, Input] = {
        args.key: Input(
            amplitude  = amp,
            frequency  = freq,
            flag_array = flag_array,
            atm_ranges = atm_ranges,
        )
    }

    return compute_scan_statistics_scores(keyed_input, kernel_kind=args.kernel)


if __name__ == "__main__":
    main()
