from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .config import ref_freq, get_kernel_kind, KernelKind
from .kernels import _get_kernel as _get_kernel
from .gaussian_state import (
    GaussianWindowState,
    gaussian_state_init,
    gaussian_state_add_point,
    gaussian_state_remove_point,
)
from .scoring import (
    calculate_nwkr_sra,
    ssr_region_dispatch,
    score_variance_nwkr,
)
from .laplace_fast import calculate_laplace_sra_fast
from .predictors import predict_on_idxs

def scan_row(params: Tuple[int, np.ndarray, List[Tuple[int,int]], np.ndarray, int, int]):
    row_idx, row, ignore, freqs, buffer, sr_factor = params
    n = row.shape[0]
    kernel_kind = get_kernel_kind()

    def _overlap_stats(a_orig: int, b_orig: int, ignore_ranges: List[Tuple[int, int]]) -> float:
        if b_orig <= a_orig:
            return 0.0
        win_len = (b_orig - a_orig + 1)
        if win_len <= 0:
            return 0.0
        overlap = 0
        for s, e in ignore_ranges:
            lo = max(a_orig, s)
            hi = min(b_orig, e)
            if hi >= lo:
                overlap += (hi - lo + 1)
        return overlap / win_len

    if len(freqs) < 2 or not np.isfinite(freqs[:2]).all():
        return (
            row_idx, (0,0), -np.inf, np.array([]), None, None, 0, 0,
            (0,0), -np.inf, 0.0, 0, None, None,
            (0,0), -np.inf, 0.0, 0, None, None, 0
        )

    freq_step = abs(freqs[1] - freqs[0])
    L = len(freqs)
    R = ref_freq / (freq_step if freq_step > 0 else 1.0)
    w = int(round(max(3, min(R, L / 16))))
    range_cap = 3 * w

    row_trimmed = row[buffer: len(row) - buffer]
    n_trimmed = row_trimmed.shape[0]
    if n_trimmed <= 0:
        return (
            row_idx, (0,0), -np.inf, np.array([]), None, None, 0, 0,
            (0,0), -np.inf, 0.0, 0, None, None,
            (0,0), -np.inf, 0.0, 0, None, None, 0
        )
    
    all_full = np.arange(n)
    if kernel_kind == KernelKind.GAUSSIAN:
        W_full = _get_kernel(n, w, KernelKind.GAUSSIAN)
        sra_full, ssr_full, _ = calculate_nwkr_sra(row, W_full)
        lap_sigma_full = None
    else:
        W_full = None
        lap_sigma_full = sigma = float(max(w, 1))
        sra_full, ssr_full, _ = calculate_laplace_sra_fast(row, lap_sigma_full)
    sra_full = max(sra_full, 1e-12)

    if kernel_kind == KernelKind.GAUSSIAN:
        W_trimmed = _get_kernel(n_trimmed, w, KernelKind.GAUSSIAN)
    else:
        W_trimmed = None
    if kernel_kind == KernelKind.GAUSSIAN:
        sra, ssr_array, pred_array = calculate_nwkr_sra(row_trimmed, W_trimmed)
        sigma = None
    elif kernel_kind == KernelKind.LAPLACE:
        sigma = float(max(w, 1))
        sra, ssr_array, pred_array = calculate_laplace_sra_fast(row_trimmed, sigma)
    sra = sra if sra > 1e-12 else 1e-12

    ignore_trimmed: List[Tuple[int, int]] = []
    for (start, end) in ignore:
        s0 = max(start - buffer, 0)
        e0 = min(end - buffer, n_trimmed - 1)
        if s0 <= e0:
            ignore_trimmed.append((s0, e0))

    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s0, e0 in ignore_trimmed:
        mask[s0:e0 + 1] = False

    all_trimmed = np.arange(n_trimmed)
    valid_masked = np.nonzero(mask)[0]

    def _varlen_search(valid: np.ndarray) -> Tuple[Tuple[int,int], float, np.ndarray | None, np.ndarray | None]:
        best_sc = -np.inf
        best_win = (0, 0)
        best_idx_full = None
        best_vals = None

        n_valid = valid.shape[0]
        for pos_i in range(n_valid):
            i = valid[pos_i]
            if pos_i < n_valid - 1 and (valid[pos_i + 1] - i) > 1:
                continue

            stop = min(pos_i + 1 + range_cap, n_valid)
            sub = valid[pos_i + 1: stop]

            gstate_in: GaussianWindowState | None = None

            for pos_j in range(sub.shape[0]):
                j = sub[pos_j]
                if pos_j > 0 and (sub[pos_j] - sub[pos_j - 1]) > 1:
                    break

                lo = pos_i
                hi = pos_i + 1 + pos_j
                inside = valid[lo:hi + 1]
                outside = np.setdiff1d(all_trimmed, inside, assume_unique=True)
                sigma = float(max(w, 1))

                if kernel_kind == KernelKind.GAUSSIAN:
                    new_idx = inside[-1]
                    if gstate_in is None:
                        gstate_in = gaussian_state_init(row_trimmed, inside, W_trimmed)
                    else:
                        if new_idx != gstate_in.idxs[-1]:
                            gstate_in = gaussian_state_add_point(row_trimmed, gstate_in, new_idx, W_trimmed)
                    sri_inc = gstate_in.sse

                    sro = ssr_region_dispatch(
                        row_trimmed,
                        outside,
                        W_trimmed,
                        ssr_array,
                        int(i),
                        int(j),
                        range_cap,
                        kernel_kind,
                        sigma,
                    )

                    sc = -(sri_inc + sro)

                else:
                    sc = score_variance_nwkr(
                        row_trimmed,
                        inside,
                        outside,
                        int(i),
                        int(j),
                        range_cap,
                        W_trimmed,
                        ssr_array,
                        kernel_kind,
                        sigma
                    )

                sc = sc / sra + 1.0
                if sc > best_sc:
                    best_sc = sc
                    best_win = (i, j)
                    best_idx_full = inside + buffer
                    sigma = float(max(w, 1))
                    best_vals = predict_on_idxs(
                        row_trimmed,
                        inside,
                        W_trimmed,
                        kernel_kind,
                        sigma,
                    )

        oi, oj = best_win
        return (oi + buffer, oj + buffer), best_sc, best_idx_full, best_vals

    window_bins = int(round(R))

    def _fixedlen_sweep() -> Tuple[Tuple[int,int], float, np.ndarray | None, np.ndarray | None]:
        best_sc = -np.inf
        best_win = (0, 0)
        best_idx_full = None
        best_vals = None

        if window_bins <= 0 or window_bins > n:
            return best_win, best_sc, best_idx_full, best_vals
        max_start = n - window_bins

        gstate_in: GaussianWindowState | None = None

        for i in range(max_start + 1):
            j = i + window_bins - 1

            inside  = np.arange(i, i + window_bins, dtype=np.int64)
            outside = np.setdiff1d(all_full, inside, assume_unique=False)

            if kernel_kind == KernelKind.GAUSSIAN:
                if gstate_in is None:
                    gstate_in = gaussian_state_init(row, inside, W_full)
                else:
                    rem_idx = i - 1
                    if rem_idx >= 0:
                        gstate_in = gaussian_state_remove_point(row, gstate_in, rem_idx, W_full)
                    gstate_in = gaussian_state_add_point(row, gstate_in, j, W_full)

                sri_inc = gstate_in.sse
                sro = ssr_region_dispatch(
                    row, outside, W_full, ssr_full, i, j, range_cap, kernel_kind,
                    0.0 if lap_sigma_full is None else lap_sigma_full
                )
                sc = -(sri_inc + sro)
            else:
                sc = score_variance_nwkr(
                    row, inside, outside, i, j, range_cap, W_full, ssr_full, kernel_kind,
                    0.0 if lap_sigma_full is None else lap_sigma_full
                )

            sc = sc / sra_full + 1.0

            if sc > best_sc:
                best_sc   = sc
                best_win  = (i, j)
                best_idx_full = inside
                best_vals = predict_on_idxs(
                    row, inside, W_full, kernel_kind,
                    0.0 if lap_sigma_full is None else lap_sigma_full
                )

        return best_win, best_sc, best_idx_full, best_vals


    best_win_masked, best_sc_masked, sri_idx_masked, sri_vals_masked = _varlen_search(valid_masked)

    best_win_unmasked, best_sc_unmasked, sri_idx_unmasked, sri_vals_unmasked = _varlen_search(all_trimmed)
    overlap_pct_unmasked = _overlap_stats(best_win_unmasked[0], best_win_unmasked[1], ignore)

    best_win_fixed, best_sc_fixed, sri_idx_fixed, sri_vals_fixed = _fixedlen_sweep()
    overlap_pct_fixed = _overlap_stats(best_win_fixed[0], best_win_fixed[1], ignore)

    return (
        row_idx, best_win_masked, best_sc_masked, pred_array, sri_idx_masked, sri_vals_masked, w * sr_factor, range_cap * sr_factor,
        best_win_unmasked, best_sc_unmasked, overlap_pct_unmasked, sri_idx_unmasked, sri_vals_unmasked,
        best_win_fixed, best_sc_fixed, overlap_pct_fixed, sri_idx_fixed, sri_vals_fixed, window_bins * sr_factor
    )
