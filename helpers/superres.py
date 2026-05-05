from __future__ import annotations
import math
import numpy as np
from typing import Iterable, List, Tuple

from .predictors import predict_on_idxs_trunc
from .scoring import calculate_gaussian_sra_trunc, calculate_laplace_sra_fast_trunc
from .kernels import truncated_kernel_vector
from .config import ref_freq, get_kernel_kind, get_super_resolve_base, KernelKind

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

def refine_all_windows_exact_for_length(
    spec_arrays,
    freq_arrays,
    windows_masked_sr,
    windows_unmasked_sr,
    windows_fixed_sr,
    atm_interfs,
    ws,
    range_caps,
    sr_factor,
    buffer
):
    N, L = spec_arrays.shape
    n_trimmed = L - 2 * buffer
    kernel_kind = get_kernel_kind()

    if n_trimmed <= 0:
        z = [(0, 0)] * N
        return z, z, z

    out_masked: List[Tuple[int, int]] = []
    out_unmasked: List[Tuple[int, int]] = []
    out_fixed: List[Tuple[int, int]] = []

    for i in range(N):
        w_i = int(ws[i])
        range_cap = int(range_caps[i])
        row_trimmed = np.asarray(spec_arrays[i, buffer:L-buffer], dtype=np.float64)
        freqs = np.asarray(freq_arrays[i], dtype=np.float64)

        # ------------------------------------------------------------
        # Build truncated kernel + SRA with the same kernel definition
        # ------------------------------------------------------------
        if kernel_kind == KernelKind.GAUSSIAN:
            k_vector = truncated_kernel_vector(
                w=float(w_i),
                r=range_cap,
                kind="gaussian"
            )
            sra, _, _, _, _, _ = calculate_gaussian_sra_trunc(row_trimmed, k_vector)

        else:
            sigma = float(max(w_i, 1))
            k_vector = truncated_kernel_vector(
                w=float(sigma),
                r=range_cap,
                kind="laplace"
            )
            sra, _, _, _ = calculate_laplace_sra_fast_trunc(row_trimmed, sigma, range_cap)

        sra = max(float(sra), 1e-12)

        # ------------------------------------------------------------
        # Build masked / unmasked valid sets
        # ------------------------------------------------------------
        mask = np.ones(n_trimmed, dtype=np.bool_)
        if len(atm_interfs) != 0:
            for (s, e) in atm_interfs[i]:
                s0 = max(s - buffer, 0)
                e0 = min(e - buffer, n_trimmed - 1)
                if s0 <= e0:
                    mask[s0:e0+1] = False

        valid_all = np.arange(n_trimmed, dtype=np.int64)
        valid_masked = valid_all[mask]

        # ------------------------------------------------------------
        # Exact subset SSE under the truncated kernel
        # ------------------------------------------------------------
        def _subset_sse(idxs: np.ndarray) -> float:
            if idxs.size == 0:
                return 0.0

            preds = predict_on_idxs_trunc(row_trimmed, idxs, k_vector)
            diff = row_trimmed[idxs] - preds
            return float(np.dot(diff, diff))

        def _score_varlen(a: int, b: int, valid: np.ndarray) -> float:
            if b < a:
                return -np.inf

            inside = valid[(valid >= a) & (valid <= b)]
            if inside.size == 0:
                return -np.inf

            # Keep same convention as your previous code:
            # outside is complement relative to all trimmed indices
            outside = np.setdiff1d(valid_all, inside, assume_unique=False)

            sri = _subset_sse(inside)
            sro = _subset_sse(outside)

            return 1.0 - (sri + sro) / sra

        # ------------------------------------------------------------
        # Local refinement around the super-res candidate
        # ------------------------------------------------------------
        def _refine_varlen_from_sr(x_sr: int, y_sr: int, valid: np.ndarray) -> Tuple[int, int]:
            a_lo = max(x_sr * sr_factor - buffer, 0)
            a_hi = min((x_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)
            b_lo = max(y_sr * sr_factor - buffer, 0)
            b_hi = min((y_sr + 1) * sr_factor - 1 - buffer, n_trimmed - 1)

            best_sc = -np.inf
            best_ab = (a_lo, max(a_lo, b_lo))

            for a in range(a_lo, a_hi + 1):
                b_start = max(a, b_lo)
                for b in range(b_start, b_hi + 1):
                    sc = _score_varlen(a, b, valid)
                    if sc > best_sc:
                        best_sc = sc
                        best_ab = (a, b)

            a_t, b_t = best_ab
            return (a_t + buffer, b_t + buffer)

        def _refine_fixed_from_sr(x_sr: int, y_sr: int) -> Tuple[int, int]:
            freq_step = abs(freqs[1] - freqs[0])
            L = len(freqs)
            R = ref_freq / (freq_step if freq_step > 0 else 1.0)
            fixed_bins_native = int(math.floor(R)) + 1
            fixed_bins_native = max(1, min(fixed_bins_native, n_trimmed))

            a_lo = max(x_sr * sr_factor - buffer, 0)
            a_hi = min((x_sr + 1) * sr_factor - 1 - buffer, n_trimmed - fixed_bins_native)

            def b_from_a(a: int) -> int:
                return a + fixed_bins_native - 1

            best_sc = -np.inf
            best_a = a_lo

            for a in range(a_lo, a_hi + 1):
                b = b_from_a(a)
                sc = _score_varlen(a, b, valid_all)
                if sc > best_sc:
                    best_sc = sc
                    best_a = a

            a_t = best_a
            b_t = b_from_a(best_a)

            # return in full-row coordinates (consistent with varlen)
            return (a_t + buffer, b_t + buffer)

        # ------------------------------------------------------------
        # Refine all three window types
        # ------------------------------------------------------------
        xm, ym = windows_masked_sr[i]
        xu, yu = windows_unmasked_sr[i]
        xf, yf = windows_fixed_sr[i]

        out_masked.append(_refine_varlen_from_sr(xm, ym, valid_masked))
        out_unmasked.append(_refine_varlen_from_sr(xu, yu, valid_all))
        out_fixed.append(_refine_fixed_from_sr(xf, yf))

    return out_masked, out_unmasked, out_fixed
