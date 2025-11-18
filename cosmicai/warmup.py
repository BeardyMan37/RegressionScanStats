from __future__ import annotations
import math, numpy as np
from typing import Dict, Tuple, List
from .config import ref_freq, get_kernel_kind, set_kernel
from .kernels import _get_kernel as _get_kernel
from .scoring import calculate_nwkr_sra
from .laplace_fast import calculate_laplace_sra_fast
from .predictors import predict_on_idxs
from .scoring import ssr_region_dispatch
from .scan import scan_row
from numba import njit

def _estimate_w_from_freqs(freqs_row: np.ndarray, sr_factor: int = 1) -> int:
    step = float(np.median(np.diff(freqs_row))) if len(freqs_row) > 1 else ref_freq
    R = ref_freq / (step if step > 0 else ref_freq)
    return int(round(max(3, min(R / sr_factor, len(freqs_row) / 16))))

def _jit_touch_laplace_path(n: int, sigma: float) -> None:
    x = np.random.rand(n).astype(np.float64)
    _ = calculate_laplace_sra_fast(x, sigma)

def _jit_touch_gaussian_path(n: int, w_bins: int) -> None:
    x = np.random.rand(n).astype(np.float64)
    W = _get_kernel(n, float(w_bins), "gaussian")
    _ = calculate_nwkr_sra(x, W)

def _jit_touch_predictors(n: int, w_bins: int, kind: str) -> None:
    x = np.random.rand(n).astype(np.float64)
    idxs = np.arange(0, n, max(1, n // 8), dtype=np.int64)
    if kind == "gaussian":
        W = _get_kernel(n, float(w_bins), "gaussian")
        _ = predict_on_idxs(x, idxs, W, kind_str="gaussian", sigma=None)
    else:
        W = None
        _ = predict_on_idxs(x, idxs, None, kind_str="laplace", sigma=float(max(w_bins, 1)))
    ssr_arr = (x - x.mean()).astype(np.float64) ** 2
    W = _get_kernel(n, float(w_bins), "gaussian")
    _ = ssr_region_dispatch(x, idxs, W, ssr_arr, 2, min(n - 1, 6), range_cap=3, kind="gaussian", sigma=None)

def _build_tiny_scan_param(L: int, kind: str) -> tuple:
    row = np.random.rand(L).astype(np.float64)
    freqs = (np.arange(L, dtype=np.float64) * ref_freq)
    ignore: List[Tuple[int, int]] = [(max(1, L//6), min(L-2, L//6 + 4))]
    buffer = max(1, L // 32)
    sr_factor = 1
    return (0, row, ignore, freqs, buffer, sr_factor)

def warmup_numba_and_caches(
    groups: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Tuple[int,int]]]]],
    kinds: Tuple[str, ...] = ("laplace", "gaussian"),
    sample_per_length: int = 1,
    small_n: int = 128,
) -> None:
    lengths = sorted(groups.keys())
    for L in lengths:
        specs, _, _, _, _, freqs, _ = groups[L]
        # choose a few rows to estimate widths that will actually be used
        for i in range(0, min(sample_per_length, specs.shape[0])):
            w_bins = _estimate_w_from_freqs(np.asarray(freqs[i], dtype=np.float64))
            for k in kinds:
                _ = _get_kernel(max(8, min(L, small_n)), float(max(3, w_bins)), k)

    _jit_touch_laplace_path(n=small_n, sigma=16.0)
    _jit_touch_gaussian_path(n=small_n, w_bins=16)

    for k in kinds:
        _jit_touch_predictors(n=small_n, w_bins=16, kind=k)

    old = get_kernel_kind().value
    for k in kinds:
        set_kernel(k)                     
        params = _build_tiny_scan_param(L=max(32, small_n), kind=k)
        _ = scan_row(params)
    set_kernel(old)

    _jit_touch_laplace_path(n=small_n, sigma=16.0)

def worker_warmup(kind: str, n: int = 64) -> None:
    if kind == "laplace":
        _jit_touch_laplace_path(n=n, sigma=8.0)
    else:
        _jit_touch_gaussian_path(n=n, w_bins=8)
    _jit_touch_predictors(n=n, w_bins=8, kind=kind)
