from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .config import ref_freq, get_kernel_kind, KernelKind
from .kernels import get_kernel as get_kernel, get_kernel_and_denom
from .gaussian_state import (
    gaussian_state_init,
    gaussian_state_add_point,
    gaussian_state_remove_point,
)
from .scoring import (
    calculate_gaussian_sra_with_nd,
    gaussian_sro_nearband,
    laplace_sri,
    laplace_sro_nearband,
    laplace_ssr_window
)
from .regressors import (
    build_regressor_sra,
    predict_on_idxs_mean,
    predict_on_idxs_poly,
    predict_on_idxs_krr,
)
from .laplace_fast import calculate_laplace_sra_fast
from .predictors import predict_on_idxs

def predict_subset_using_regressor(x: np.ndarray, idxs: np.ndarray, family: str, **kwargs) -> np.ndarray:
    family = family.lower()
    if family == "mean":
        return predict_on_idxs_mean(x, idxs)
    elif family == "poly":
        degree = int(kwargs.get("degree", 1))
        reg = float(kwargs.get("reg", 1e-8))
        return predict_on_idxs_poly(x, idxs, degree=degree, reg=reg)
    elif family == "krr":
        kernel = kwargs.get("kernel", "rbf")
        kernel_param = float(kwargs.get("kernel_param", 1.0))
        reg = float(kwargs.get("reg", 1e-3))
        return predict_on_idxs_krr(x, idxs, kernel=kernel, kernel_param=kernel_param, reg=reg)
    else:
        raise ValueError(f"Unknown family {family!r}")
    
def scan_row_with_regressor(
    x: np.ndarray,
    w: int,
    family: str = "mean",
    **kwargs,
):
    x = x.astype(np.float64)
    n = x.shape[0]

    sra_all, _, _, _, _, _ = build_regressor_sra(x, family, **kwargs)

    best_score = -np.inf
    best_a = -1
    best_b = -1

    for a in range(n):
        max_b = min(n - 1, a + w - 1)
        for b in range(a, max_b + 1):
            inside = np.arange(a, b + 1, dtype=np.int64)

            if a == 0 and b == n - 1:
                continue
            if a == 0:
                outside = np.arange(b + 1, n, dtype=np.int64)
            elif b == n - 1:
                outside = np.arange(0, a, dtype=np.int64)
            else:
                outside = np.concatenate(
                    (np.arange(0, a, dtype=np.int64),
                     np.arange(b + 1, n, dtype=np.int64))
                )

            if inside.size > 0:
                pred_in = predict_subset_using_regressor(x, inside, family, **kwargs)
                resid_in = x[inside] - pred_in
                sse_in = float(np.dot(resid_in, resid_in))
            else:
                sse_in = 0.0

            if outside.size > 0:
                pred_out = predict_subset_using_regressor(x, outside, family, **kwargs)
                resid_out = x[outside] - pred_out
                sse_out = float(np.dot(resid_out, resid_out))
            else:
                sse_out = 0.0

            score = sra_all - (sse_in + sse_out)

            if score > best_score:
                best_score = score
                best_a = a
                best_b = b

    return best_score, best_a, best_b


def scan_row_with_nwkr(params: Tuple[int, np.ndarray, List[Tuple[int,int]], np.ndarray, int, int]):
    row_idx, row, ignore, freqs, buffer, sr_factor = params[:6]

    w_override = None
    range_cap_override = None
    fixed_bins_override = None

    if len(params) >= 7:
        w_override = params[6]
    if len(params) >= 8:
        range_cap_override = params[7]
    if len(params) >= 9:
        fixed_bins_override = params[8]

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

    w_auto = int(round(max(3, min(R, L / 16))))
    range_cap_auto = 3 * w_auto
    window_bins_auto = int(round(R)) + 1
    span_hz = freqs[min(window_bins_auto, len(freqs)) - 1] - freqs[0]
    
    fixed_len_flag = True
    if ref_freq != span_hz:
        fixed_len_flag = False
    

    if w_override is not None and int(w_override) > 0:
        w = int(w_override)
    else:
        w = w_auto

    if range_cap_override is not None and int(range_cap_override) > 0:
        range_cap = int(range_cap_override)
    else:
        range_cap = range_cap_auto

    if fixed_bins_override is not None and int(fixed_bins_override) > 0:
        window_bins = int(fixed_bins_override)
    else:
        window_bins = window_bins_auto


    row_trimmed = row[buffer: len(row) - buffer]
    n_trimmed = row_trimmed.shape[0]
    if n_trimmed <= 0:
        return (
            row_idx, (0,0), -np.inf, np.array([]), None, None, 0, 0,
            (0,0), -np.inf, 0.0, 0, None, None,
            (0,0), -np.inf, 0.0, 0, None, None, 0
        )
    
    if kernel_kind == KernelKind.GAUSSIAN:
        W_full, denom_cached_full = get_kernel_and_denom(n, w, KernelKind.GAUSSIAN)
        sra_full, _, _, numer_full, denom_full, ssr_ps_full = calculate_gaussian_sra_with_nd(row, W_full, denom_cached_full)
        W_trimmed, denom_cached_trimmed = get_kernel_and_denom(n_trimmed, w, KernelKind.GAUSSIAN)
        sra, _, pred_array, numer_trim, denom_trim, ssr_ps_trim = calculate_gaussian_sra_with_nd(row_trimmed, W_trimmed, denom_cached_trimmed)
        sigma = None
    else:
        W_full = None
        sigma = float(max(w, 1))
        sra_full, _, _, ssr_ps_full = calculate_laplace_sra_fast(row, sigma)
        W_trimmed = None
        sra, _, pred_array, ssr_ps_trim = calculate_laplace_sra_fast(row_trimmed, sigma)
    
    sra_full = max(sra_full, 1e-12)
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

    all_full = np.arange(n)
    all_trimmed = np.arange(n_trimmed)
    valid_idxs_masked = np.nonzero(mask)[0]

    inside_buf = np.empty(range_cap + 1, dtype=np.int64)

    def _varlen_search(valid: np.ndarray) -> Tuple[Tuple[int, int], float, np.ndarray | None, np.ndarray | None]:
        best_sc = -np.inf
        best_win = (0, 0)
        best_idx_full = None
        best_vals = None

        n_valid = valid.shape[0]
        for pos_i in range(n_valid):
            i = valid[pos_i]

            if pos_i + 1 < n_valid and valid[pos_i + 1] != i + 1:
                continue

            g_in_initialized = False

            inside_buf[0] = i
            m = 1

            # grow window forward in valid-index order
            max_k = min(pos_i + range_cap, n_valid - 1)

            for k in range(pos_i + 1, max_k + 1):
                # enforce contiguity in actual index space
                if valid[k] != valid[k - 1] + 1:
                    break

                j = valid[k]
                inside_buf[m] = j
                m += 1

                inside = inside_buf[:m]

                if kernel_kind == KernelKind.GAUSSIAN:
                    new_idx = j

                    if not g_in_initialized:
                        g_in_idxs, g_in_num, g_in_den, g_in_sse = gaussian_state_init(
                            row_trimmed, inside, W_trimmed
                        )
                        g_in_initialized = True

                        outside_init = np.setdiff1d(all_trimmed, inside, assume_unique=True)
                        g_out_idxs, g_out_num, g_out_den, g_out_sse = gaussian_state_init(
                            row_trimmed, outside_init, W_trimmed
                        )
                    else:
                        g_in_idxs, g_in_num, g_in_den, g_in_sse = gaussian_state_add_point(
                            row_trimmed,
                            g_in_idxs,
                            g_in_num,
                            g_in_den,
                            g_in_sse,
                            new_idx,
                            W_trimmed,
                        )
                        g_out_idxs, g_out_num, g_out_den, g_out_sse = gaussian_state_remove_point(
                            row_trimmed,
                            g_out_idxs,
                            g_out_num,
                            g_out_den,
                            g_out_sse,
                            new_idx,
                            W_trimmed,
                        )

                    sri = g_in_sse
                    sro = g_out_sse
                    sc = -(sri + sro)

                else:
                    outside = np.setdiff1d(all_trimmed, inside, assume_unique=True)
                    sri = laplace_sri(row_trimmed, inside, int(i), int(j), sigma)
                    sro = laplace_sro_nearband(
                        row_trimmed,
                        outside,
                        int(i),
                        int(j),
                        range_cap,
                        ssr_ps_trim,
                        sigma,
                    )
                    sc = -(sri + sro)

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

    def _fixedlen_sweep() -> Tuple[Tuple[int, int], float, np.ndarray | None, np.ndarray | None]:
        best_sc = -np.inf
        best_win = (0, 0)
        best_idx_full = None
        best_vals = None

        if window_bins <= 0 or window_bins > n:
            return best_win, best_sc, best_idx_full, best_vals

        max_start = n - window_bins

        g_in_initialized = False

        g_in_idxs = g_in_num = g_in_den = None
        g_out_idxs = g_out_num = g_out_den = None
        g_in_sse = 0.0
        g_out_sse = 0.0

        for i in range(max_start + 1):
            j = i + window_bins - 1
            inside = np.arange(i, i + window_bins, dtype=np.int64)

            if kernel_kind == KernelKind.GAUSSIAN:
                if not g_in_initialized:
                    g_in_idxs, g_in_num, g_in_den, g_in_sse = gaussian_state_init(
                        row,
                        inside,
                        W_full,
                    )
                    g_in_initialized = True

                    outside_init = np.setdiff1d(all_full, inside, assume_unique=True)
                    g_out_idxs, g_out_num, g_out_den, g_out_sse = gaussian_state_init(
                        row,
                        outside_init,
                        W_full,
                    )

                else:
                    rem_idx_inside = i - 1
                    add_idx_inside = j

                    if rem_idx_inside >= 0:
                        g_in_idxs, g_in_num, g_in_den, g_in_sse = gaussian_state_remove_point(
                            row,
                            g_in_idxs,
                            g_in_num,
                            g_in_den,
                            g_in_sse,
                            rem_idx_inside,
                            W_full,
                        )
                    g_in_idxs, g_in_num, g_in_den, g_in_sse = gaussian_state_add_point(
                        row,
                        g_in_idxs,
                        g_in_num,
                        g_in_den,
                        g_in_sse,
                        add_idx_inside,
                        W_full,
                    )

                    g_out_idxs, g_out_num, g_out_den, g_out_sse = gaussian_state_remove_point(
                        row,
                        g_out_idxs,
                        g_out_num,
                        g_out_den,
                        g_out_sse,
                        add_idx_inside,
                        W_full,
                    )
                    if rem_idx_inside >= 0:
                        g_out_idxs, g_out_num, g_out_den, g_out_sse = gaussian_state_add_point(
                            row,
                            g_out_idxs,
                            g_out_num,
                            g_out_den,
                            g_out_sse,
                            rem_idx_inside,
                            W_full,
                        )

                sri = g_in_sse
                sro = g_out_sse
                sc = -(sri + sro)

            else:
                outside = np.setdiff1d(all_full, inside, assume_unique=True)
                sri = laplace_sri(row, inside, int(i), int(j), sigma)
                sro = laplace_sro_nearband(
                    row,
                    outside,
                    int(i),
                    int(j),
                    range_cap,
                    ssr_ps_full,
                    sigma,
                )
                sc = -(sri + sro)

            sc = sc / sra_full + 1.0

            if sc > best_sc:
                best_sc = sc
                best_win = (i, j)
                best_idx_full = inside
                best_vals = predict_on_idxs(
                    row,
                    inside,
                    W_full,
                    kernel_kind,
                    sigma,
                )

        return best_win, best_sc, best_idx_full, best_vals




    best_win_masked, best_sc_masked, sri_idx_masked, sri_vals_masked = _varlen_search(valid_idxs_masked)

    best_win_unmasked, best_sc_unmasked, sri_idx_unmasked, sri_vals_unmasked = _varlen_search(all_trimmed)
    overlap_pct_unmasked = _overlap_stats(best_win_unmasked[0], best_win_unmasked[1], ignore)

    best_win_fixed, best_sc_fixed, sri_idx_fixed, sri_vals_fixed, overlap_pct_fixed = (-1, -1), -1, None, None, -1
    
    if fixed_len_flag == True:
        best_win_fixed, best_sc_fixed, sri_idx_fixed, sri_vals_fixed = _fixedlen_sweep()
        overlap_pct_fixed = _overlap_stats(best_win_fixed[0], best_win_fixed[1], ignore)

    return (
        row_idx, best_win_masked, best_sc_masked, pred_array, sri_idx_masked, sri_vals_masked, w * sr_factor, range_cap * sr_factor,
        best_win_unmasked, best_sc_unmasked, overlap_pct_unmasked, sri_idx_unmasked, sri_vals_unmasked,
        best_win_fixed, best_sc_fixed, overlap_pct_fixed, sri_idx_fixed, sri_vals_fixed, window_bins * sr_factor
    )
