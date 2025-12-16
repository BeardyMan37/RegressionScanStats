from __future__ import annotations
import math
import numpy as np
from typing import Iterable, List, Tuple
from .scoring import (calculate_nwkr_sra, score_variance_nwkr)
from .laplace_fast import calculate_laplace_sra_fast
from .kernels import _get_kernel
from .config import get_kernel_kind, get_super_resolve_base, KernelKind

def sr_factor(L, r=2, q=2, cap=None):
        L0 = get_super_resolve_base()
        s = math.ceil((L + 1) / L0)
        k = math.ceil(math.log(s, r)) if s > 1 else 0
        f = q ** k
        return min(f, cap) if cap is not None else f

def superresolve_ranges(ranges_list, factor: int):
    if factor < 1:
        raise ValueError("superresolve_ranges: factor must be >= 1")

    def merge(rs: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for s, e in rs:
            if not out or s > out[-1][1] + 1:
                out.append((s, e))
            else:
                out[-1] = (out[-1][0], max(out[-1][1], e))
        return out

    new: List[List[Tuple[int, int]]] = []
    for sub in ranges_list:
        adjusted = [(s // factor, e // factor) for s, e in sub]
        adjusted = sorted(set(adjusted))
        merged = merge(adjusted)
        new.append(merged)
    return new

def superresolve(specs: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("superresolve: factor must be >= 1")
    n_rows, n_ch = specs.shape
    n_blk = n_ch // factor
    if n_blk == 0:
        return np.empty((n_rows, 0), dtype=specs.dtype)
    trimmed = specs[:, : n_blk * factor]
    return trimmed.reshape(n_rows, n_blk, factor).mean(axis=2)

def refine_all_windows_exact_for_length(spec_arrays, windows_masked_sr, windows_unmasked_sr,
                                        windows_fixed_sr, atm_interfs, ws, range_caps, sr_factor, buffer):
    N, L = spec_arrays.shape
    n_trimmed = L - 2 * buffer
    kernel_kind = get_kernel_kind()
    if n_trimmed <= 0:
        z = [(0, 0)] * N
        return z, z, z

    out_masked   : List[Tuple[int,int]] = []
    out_unmasked : List[Tuple[int,int]] = []
    out_fixed    : List[Tuple[int,int]] = []

    for i in range(N):
        range_cap = range_caps[i]                      # native-scale
        row_trimmed = spec_arrays[i, buffer:L-buffer]

        if kernel_kind == KernelKind.GAUSSIAN:
            sigma = 0.0
            W_trimmed = _get_kernel(n_trimmed, ws[i], KernelKind.GAUSSIAN)
            sra, ssr_array, _ = calculate_nwkr_sra(row_trimmed, W_trimmed)
        else:
            sigma = float(max(ws[i], 1))
            W_trimmed = np.empty((1,1), dtype=np.float64)
            sra, ssr_array, _, _ = calculate_laplace_sra_fast(row_trimmed, sigma)

        mask = np.ones(n_trimmed, dtype=np.bool_)
        for (s, e) in atm_interfs[i]:
            s0 = max(s - buffer, 0)
            e0 = min(e - buffer, n_trimmed - 1)
            if s0 <= e0:
                mask[s0:e0+1] = False
        valid_all = np.arange(n_trimmed, dtype=np.int64)
        valid_masked = valid_all[mask]

        def _score_varlen(a: int, b: int, valid: np.ndarray, sigma: float) -> float:
            inside = valid[(valid >= a) & (valid <= b)]
            if inside.size == 0:
                return -np.inf
            outside = np.setdiff1d(valid_all, inside, assume_unique=False)
            sc = score_variance_nwkr(row_trimmed, inside, outside, a, b, range_cap, W_trimmed, ssr_array, kernel_kind, sigma)
            return sc / sra + 1.0

        def _refine_varlen_from_sr(x_sr: int, y_sr: int, valid: np.ndarray, sigma: float) -> Tuple[int,int]:
            a_lo = max(x_sr * sr_factor - buffer, 0)
            a_hi = min((x_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)
            b_lo = max(y_sr * sr_factor - buffer, 0)
            b_hi = min((y_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)

            best_sc, best_ab = -np.inf, (a_lo, max(a_lo, b_lo))
            for a in range(a_lo, a_hi + 1):
                b_start = max(a, b_lo)
                for b in range(b_start, b_hi + 1):
                    sc = _score_varlen(a, b, valid, sigma)
                    if sc > best_sc:
                        best_sc, best_ab = sc, (a, b)
            a_t, b_t = best_ab
            return (a_t + buffer, b_t + buffer)

        def _refine_fixed_from_sr(x_sr: int, y_sr: int, sigma: float) -> Tuple[int,int]:
            fixed_bins_native = (y_sr - x_sr + 1) * sr_factor
            fixed_bins_native = max(1, min(fixed_bins_native, n_trimmed))

            a_lo = max(x_sr * sr_factor - buffer, 0)
            a_hi = min((x_sr + 1) * sr_factor - 1 - buffer, n_trimmed - fixed_bins_native)
            b_from_a = lambda a: a + fixed_bins_native - 1

            best_sc, best_a = -np.inf, a_lo
            for a in range(a_lo, a_hi + 1):
                b = b_from_a(a)
                sc = _score_varlen(a, b, valid_all, sigma)
                if sc > best_sc:
                    best_sc, best_a = sc, a
            a_t, b_t = best_a, b_from_a(best_a)
            return (a_t + buffer, b_t + buffer)

        xm, ym = windows_masked_sr[i]
        xu, yu = windows_unmasked_sr[i]
        xf, yf = windows_fixed_sr[i]

        out_masked.append(_refine_varlen_from_sr(xm, ym, valid_masked, sigma))
        out_unmasked.append(_refine_varlen_from_sr(xu, yu, valid_all, sigma))
        out_fixed.append(_refine_fixed_from_sr(xf, yf, sigma))

    return out_masked, out_unmasked, out_fixed
