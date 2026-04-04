"""
scan_statistics.py
==================
Public entry point for computing NWKR-based scan statistics on batched
bandpass calibration inputs.

This module is self-contained: it does not import from ``cosmicai``.
All core logic is re-implemented here to be dependency-free, but the
algorithm is identical to ``cosmicai.scan.scan_row_with_nwkr``.

Usage
-----
    from scan_statistics import compute_scan_statistics_scores, Input, Output

    results = compute_scan_statistics_scores({
        "baseline_1": Input(amplitude=amp, frequency=freq, flag_array=flags),
    })
    out = results["baseline_1"]["masked"]
    print(out.score, out.win_start, out.win_end)
"""

from __future__ import annotations

import math
import logging
from itertools import groupby
from typing import Dict, List, Literal, NamedTuple, Tuple, TypeAlias

import numpy as np

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class Input(NamedTuple):
    """Input data for a single bandpass calibration solution."""
    # Amplitude values (1D array)
    amplitude: np.ndarray
    # Frequency values (1D array)
    frequency: np.ndarray
    # 1D Boolean array of flags, same length as amplitude;
    # True denotes a flagged (excluded) channel.
    flag_array: np.ndarray


class Output(NamedTuple):
    """Scan statistic result for one scan mode."""
    score: float
    win_start: int
    win_end: int


ScanMode: TypeAlias = Literal["masked", "unmasked", "fixed"]
ScanResult: TypeAlias = Dict[ScanMode, Output]

# ---------------------------------------------------------------------------
# Internal constants  (mirror cosmicai/config.py)
# ---------------------------------------------------------------------------

_REF_FREQ: float = 0.0625       # GHz — reference spectral-line channel width
_BUFFER_DIVISOR: int = 20       # buffer = len(frequency) // _BUFFER_DIVISOR
_DEFAULT_KERNEL: str = "gaussian"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers: flag array → index ranges
# ---------------------------------------------------------------------------

def _flag_array_to_ranges(flag_array: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a boolean flag array to a list of inclusive (start, end) ranges."""
    ranges: List[Tuple[int, int]] = []
    idx = 0
    for val, grp in groupby(flag_array.tolist()):
        length = sum(1 for _ in grp)
        if val:
            ranges.append((idx, idx + length - 1))
        idx += length
    return ranges

# ---------------------------------------------------------------------------
# Helpers: truncated kernel vector (mirrors cosmicai.kernels)
# ---------------------------------------------------------------------------

def _truncated_kernel_vector(w: float, r: int, kind: str = "gaussian") -> np.ndarray:
    """
    Build k[d] for d = 0..r.

    Gaussian : exp(-(d^2)/(w^2))
    Laplace  : exp(-d/w)
    """
    d = np.arange(r + 1, dtype=np.float64)
    if kind == "gaussian":
        return np.exp(-(d * d) / (w * w))
    if kind == "laplace":
        return np.exp(-d / max(float(w), 1e-12))
    raise ValueError(f"Unknown kernel kind: {kind!r}")

# ---------------------------------------------------------------------------
# Helpers: NWKR SSE with a truncated kernel (pure NumPy, no Numba)
# ---------------------------------------------------------------------------

def _nwkr_sse_trunc(x: np.ndarray, idxs: np.ndarray, k_vec: np.ndarray) -> float:
    """
    NWKR sum-of-squared residuals over the subset ``idxs``.

    For each index i in ``idxs`` the prediction is::

        yhat[i] = sum_{j in idxs, |i-j|<=r}  k[|i-j|] * x[j]
                / sum_{j in idxs, |i-j|<=r}  k[|i-j|]

    ``idxs`` must be sorted ascending (as produced by np.arange or np.nonzero).
    """
    m = idxs.shape[0]
    if m == 0:
        return 0.0

    r = k_vec.shape[0] - 1
    sse = 0.0

    for ui in range(m):
        i = int(idxs[ui])
        s_num = 0.0
        s_den = 0.0

        # scan left neighbours
        vi = ui
        while vi >= 0:
            dist = i - int(idxs[vi])
            if dist > r:
                break
            w = k_vec[dist]
            s_num += w * x[int(idxs[vi])]
            s_den += w
            vi -= 1

        # scan right neighbours
        vi = ui + 1
        while vi < m:
            dist = int(idxs[vi]) - i
            if dist > r:
                break
            w = k_vec[dist]
            s_num += w * x[int(idxs[vi])]
            s_den += w
            vi += 1

        pred = s_num / s_den if s_den > 1e-12 else 0.0
        diff = x[i] - pred
        sse += diff * diff

    return float(sse)


def _nwkr_sra(x: np.ndarray, k_vec: np.ndarray) -> float:
    """Global NWKR SSE (SRA baseline) over the full array x."""
    return max(_nwkr_sse_trunc(x, np.arange(x.shape[0], dtype=np.int64), k_vec), 1e-12)

# ---------------------------------------------------------------------------
# Core per-row scan
# ---------------------------------------------------------------------------

def _scan_single_row(
    amplitude: np.ndarray,
    frequency: np.ndarray,
    flag_ranges: List[Tuple[int, int]],
    buffer: int,
    kernel_kind: str,
) -> ScanResult:
    """
    Run all three scan modes on one spectrum.

    Parameters
    ----------
    amplitude    : 1-D float64 spectrum values.
    frequency    : 1-D float64 frequency axis (GHz).
    flag_ranges  : Inclusive (start, end) index pairs for flagged regions.
    buffer       : Number of edge channels to exclude from varlen scans.
    kernel_kind  : "gaussian" or "laplace".

    Returns
    -------
    ScanResult
        ``{"masked": Output, "unmasked": Output, "fixed": Output}``
    """
    x = amplitude
    n = x.shape[0]
    _BAD = Output(score=-np.inf, win_start=0, win_end=0)

    if n < 4 or len(frequency) < 2:
        return {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}

    # ---- derive kernel/window parameters from frequency grid ----
    freq_step = abs(float(frequency[1] - frequency[0]))
    if freq_step <= 0.0:
        freq_step = _REF_FREQ
    R = _REF_FREQ / freq_step
    L = len(frequency)

    w = int(round(max(3.0, min(R, L / 16.0))))   # kernel half-width (channels)
    range_cap = 3 * w                             # max window size for varlen scan
    window_bins = int(math.floor(R)) + 1          # fixed window width

    # ---- trim buffer edges (varlen scans only) ----
    buf = max(buffer, 0)
    trim_start = buf
    trim_end = n - buf
    row_trimmed = x[trim_start:trim_end] if (trim_end - trim_start) > 2 else x
    n_trimmed = row_trimmed.shape[0]

    if n_trimmed <= 2:
        return {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}

    # ---- build truncated kernel vectors ----
    k_vec = _truncated_kernel_vector(float(w), range_cap, kind=kernel_kind)

    # ---- global SRA baselines ----
    sra = _nwkr_sra(row_trimmed, k_vec)        # for varlen scoring
    sra_full = _nwkr_sra(x, k_vec)             # for fixed-length scoring

    # ---- build valid (non-flagged) index sets on trimmed row ----
    ignore_trimmed: List[Tuple[int, int]] = []
    for (s, e) in flag_ranges:
        s0 = max(s - buf, 0)
        e0 = min(e - buf, n_trimmed - 1)
        if s0 <= e0:
            ignore_trimmed.append((s0, e0))

    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s0, e0 in ignore_trimmed:
        mask[s0:e0 + 1] = False

    all_trimmed = np.arange(n_trimmed, dtype=np.int64)
    valid_masked = all_trimmed[mask]

    # ----------------------------------------------------------------
    # Variable-length window search
    # ----------------------------------------------------------------
    def _score_varlen(i: int, j: int, valid: np.ndarray) -> float:
        """Score contiguous window [i,j] (trimmed coords) against valid set."""
        inside = valid[(valid >= i) & (valid <= j)]
        if inside.size == 0:
            return -np.inf
        outside = np.setdiff1d(all_trimmed, inside, assume_unique=False)
        sri = _nwkr_sse_trunc(row_trimmed, inside, k_vec)
        sro = _nwkr_sse_trunc(row_trimmed, outside, k_vec)
        return 1.0 - (sri + sro) / sra

    def _varlen_search(valid: np.ndarray) -> Tuple[int, int, float]:
        """
        Grow contiguous windows over ``valid`` (up to ``range_cap`` channels).
        Returns ``(win_start, win_end, score)`` in **full-row** coordinates.
        """
        best_sc = 0.0
        best_i = best_j = 0
        n_valid = valid.shape[0]

        if n_valid < 2:
            return best_i + buf, best_j + buf, best_sc

        for pos_i in range(n_valid):
            i = int(valid[pos_i])

            # require i is the start of a contiguous run in valid
            if pos_i + 1 < n_valid and valid[pos_i + 1] != i + 1:
                continue

            max_k = min(pos_i + range_cap, n_valid - 1)
            for k in range(pos_i + 1, max_k + 1):
                if valid[k] != valid[k - 1] + 1:
                    break
                j = int(valid[k])
                sc = _score_varlen(i, j, valid)
                if sc > best_sc:
                    best_sc = sc
                    best_i, best_j = i, j

        return best_i + buf, best_j + buf, float(best_sc)

    # ----------------------------------------------------------------
    # Fixed-length window sweep (full row, no buffer trimming)
    # ----------------------------------------------------------------
    def _fixedlen_sweep() -> Tuple[int, int, float]:
        """
        Slide a fixed window of ``window_bins`` channels over the full spectrum.
        Returns ``(win_start, win_end, score)`` in full-row coordinates.
        """
        best_sc = 0.0
        best_i = 0
        best_j = max(window_bins - 1, 0)

        if window_bins <= 0 or window_bins > n:
            return best_i, best_j, best_sc

        all_full = np.arange(n, dtype=np.int64)
        max_start = n - window_bins

        for i in range(max_start + 1):
            j = i + window_bins - 1
            inside = np.arange(i, j + 1, dtype=np.int64)
            outside = np.setdiff1d(all_full, inside, assume_unique=True)
            sri = _nwkr_sse_trunc(x, inside, k_vec)
            sro = _nwkr_sse_trunc(x, outside, k_vec)
            sc = 1.0 - (sri + sro) / sra_full
            if sc > best_sc:
                best_sc = sc
                best_i, best_j = i, j

        return best_i, best_j, float(best_sc)

    # ---- run all three modes ----
    m_start, m_end, m_score = _varlen_search(valid_masked)
    u_start, u_end, u_score = _varlen_search(all_trimmed)
    f_start, f_end, f_score = _fixedlen_sweep()

    return {
        "masked":   Output(score=m_score, win_start=m_start, win_end=m_end),
        "unmasked": Output(score=u_score, win_start=u_start, win_end=u_end),
        "fixed":    Output(score=f_score, win_start=f_start, win_end=f_end),
    }

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_scan_statistics_scores(
    keyed_input: Dict[str, Input],
    *,
    kernel_kind: str = _DEFAULT_KERNEL,
    buffer_divisor: int = _BUFFER_DIVISOR,
) -> Dict[str, ScanResult]:
    """
    Entry point for computing scores on batched inputs.

    Parameters
    ----------
    keyed_input : Dict[str, Input]
        A dictionary of unique key mapped to a set of input.
    kernel_kind : str, optional
        Kernel shape: ``"gaussian"`` (default) or ``"laplace"``.
    buffer_divisor : int, optional
        Edge buffer = ``len(frequency) // buffer_divisor``.  Default 20
        gives ~5% on each side.

    Returns
    -------
    Dict[str, ScanResult]
        Two-level results keyed by input key, then by scan mode.

        Example::

            {
                "key1": {
                    "masked":   Output(score=..., win_start=..., win_end=...),
                    "unmasked": Output(score=..., win_start=..., win_end=...),
                    "fixed":    Output(score=..., win_start=..., win_end=...),
                },
                "key2": { ... },
            }

    Notes
    -----
    **Score definition** (identical to ``cosmicai.scan.scan_row_with_nwkr``)::

        score = 1 - (SSE_inside + SSE_outside) / SRA_global

    where SRA_global is the NWKR sum-of-squared residuals on the full array
    and SSE is computed with a truncated Gaussian (or Laplace) kernel of
    width ``w = round(ref_freq / freq_step)`` channels.

    **Scan modes**

    - ``"masked"``   — variable-length search restricted to *non-flagged*
      channels.  Flagged channels (``flag_array=True``) are excluded from
      both candidate windows and SSE evaluation.
    - ``"unmasked"`` — same variable-length search over *all* channels.
    - ``"fixed"``    — fixed-width window of
      ``floor(ref_freq / freq_step) + 1`` channels swept over the full
      spectrum (no edge trimming).

    Rows that are all-zero, all-flagged, or too short return
    ``Output(score=-inf, win_start=0, win_end=0)`` for all modes.

    Examples
    --------
    >>> import numpy as np
    >>> from scan_statistics import compute_scan_statistics_scores, Input
    >>> rng  = np.random.default_rng(42)
    >>> freq = np.linspace(85.0, 87.0, 512)
    >>> amp  = rng.standard_normal(512)
    >>> amp[200:240] += 8.0          # inject a bright spectral line
    >>> flags = np.zeros(512, dtype=bool)
    >>> flags[100:115] = True        # flag an atmospheric absorption region
    >>> results = compute_scan_statistics_scores(
    ...     {"obs_01": Input(amp, freq, flags)}
    ... )
    >>> r = results["obs_01"]
    >>> print(r["masked"].score, r["masked"].win_start, r["masked"].win_end)
    """
    if not keyed_input:
        return {}

    results: Dict[str, ScanResult] = {}
    _BAD = Output(score=-np.inf, win_start=0, win_end=0)
    _BAD_RESULT: ScanResult = {"masked": _BAD, "unmasked": _BAD, "fixed": _BAD}

    for key, inp in keyed_input.items():
        # ---- coerce and validate ----
        try:
            amplitude = np.asarray(inp.amplitude, dtype=np.float64)
            frequency = np.asarray(inp.frequency, dtype=np.float64)
            flag_array = np.asarray(inp.flag_array, dtype=bool)
        except Exception as exc:
            logger.warning("[%s] Could not coerce arrays: %s", key, exc)
            results[key] = _BAD_RESULT
            continue

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

        # ---- skip degenerate rows ----
        if np.all(amplitude == 0.0):
            logger.debug("[%s] Skipped: all-zero amplitude.", key)
            results[key] = _BAD_RESULT
            continue
        if flag_array.all():
            logger.debug("[%s] Skipped: all channels flagged.", key)
            results[key] = _BAD_RESULT
            continue

        # ---- derive buffer and flag ranges ----
        buffer = max(1, len(frequency) // buffer_divisor)
        flag_ranges = _flag_array_to_ranges(flag_array)

        logger.debug(
            "[%s] n=%d  buffer=%d  n_flag_ranges=%d  kernel=%s",
            key, len(amplitude), buffer, len(flag_ranges), kernel_kind,
        )

        # ---- scan ----
        try:
            results[key] = _scan_single_row(
                amplitude=amplitude,
                frequency=frequency,
                flag_ranges=flag_ranges,
                buffer=buffer,
                kernel_kind=kernel_kind,
            )
        except Exception as exc:
            logger.warning("[%s] Scan raised an exception: %s", key, exc, exc_info=True)
            results[key] = _BAD_RESULT

    return results
