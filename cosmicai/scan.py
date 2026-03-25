from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple
from .config import ref_freq, get_kernel_kind, KernelKind
from .kernels import _kernel_matrix, truncated_kernel_vector, get_kernel as get_kernel
from .kernel_optimized_state import (
    kernel_state_add_point_trunc,
    kernel_state_init_trunc,
    kernel_state_remove_point_trunc,
)
from .scoring import (
    _nwkr_predict_subset,
    _nwkr_sse_on_subset,
    calculate_gaussian_sra_trunc,
    calculate_laplace_sra_fast_trunc
)
from .regressors import (
    build_regressor_sra,
    predict_on_idxs_mean,
    predict_on_idxs_poly,
    predict_on_idxs_krr,
)
from .predictors import predict_on_idxs_trunc

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

    sra_all, _, _, _, _, _, _ = build_regressor_sra(x, family, **kwargs)

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

def scan_row_with_nwkr_naive(params: Tuple):
    """
    Naive NWKR scan: no incremental updates, no nearband, no recurrence.
    Extremely slow but simple and correct.

    Expected params (same leading fields as yours):
      (row_idx, row, ignore, flags, freqs, buffer, sr_factor, [w_override], [range_cap_override], [fixed_bins_override])
    """
    row_idx, row, ignore, flags, freqs, buffer, sr_factor = params[:7]

    w_override = params[7] if len(params) >= 8 else None
    range_cap_override = params[8] if len(params) >= 9 else None
    fixed_bins_override = params[9] if len(params) >= 10 else None

    row = np.asarray(row, dtype=np.float64)
    n = row.size

    # ---- infer w/range_cap/window_bins similarly to your code (kept minimal) ----
    if len(freqs) < 2 or not np.isfinite(freqs[:2]).all():
        return (
            row_idx, (0,0), -np.inf, np.array([]), None, None, 0, 0,
            (0,0), -np.inf, 0.0, 0, None, None,
            (0,0), -np.inf, 0.0, 0, None, None, 0
        )

    freq_step = abs(freqs[1] - freqs[0])
    L = len(freqs)
    R = ref_freq / (freq_step if freq_step > 0 else ref_freq)
    w_auto = int(round(max(3, min(R, L / 16))))
    range_cap_auto = 3 * w_auto
    window_bins_auto = int(math.floor(R)) + 1

    w = int(w_override) if (w_override is not None and int(w_override) > 0) else int(w_auto)
    range_cap = int(range_cap_override) if (range_cap_override is not None and int(range_cap_override) > 0) else int(range_cap_auto)
    window_bins = int(fixed_bins_override) if (fixed_bins_override is not None and int(fixed_bins_override) > 0) else int(window_bins_auto)

    # ---- trimming ----
    row_trimmed = row[buffer: n - buffer]
    n_trimmed = row_trimmed.size
    if n_trimmed <= 2:
        return (
            row_idx, (0,0), -np.inf, np.array([]), None, None, 0, 0,
            (0,0), -np.inf, 0.0, 0, None, None,
            (0,0), -np.inf, 0.0, 0, None, None, 0
        )

    # ---- kernel kind from your global setting ----
    kernel_kind = get_kernel_kind()
    kind = "gaussian" if kernel_kind == KernelKind.GAUSSIAN else "laplace"

    # ---- build FULL kernel matrices once (still not a “scan trick”; just not insane) ----
    # Full for fixed-length sweep on full row
    K_full = _kernel_matrix(n, float(w), kind)
    # Trimmed for varlen scans on trimmed row
    K_trim = _kernel_matrix(n_trimmed, float(w), kind)

    # ---- SRA (global SSE) ----
    all_full = np.arange(n, dtype=np.int64)
    all_trim = np.arange(n_trimmed, dtype=np.int64)

    sra_full = max(_nwkr_sse_on_subset(row, K_full, all_full), 1e-12)
    sra_trim = max(_nwkr_sse_on_subset(row_trimmed, K_trim, all_trim), 1e-12)

    # For plotting compatibility in your caller, provide a "global prediction array"
    # (NWKR trained on all indices, evaluated on all indices)
    # In naive form we can just compute it once:
    pred_array = _nwkr_predict_subset(row_trimmed, K_trim, all_trim)

    # ---- build ignore mask (trimmed) ----
    ignore_trimmed: List[Tuple[int, int]] = []
    for (start, end) in (ignore or []):
        s0 = max(start - buffer, 0)
        e0 = min(end - buffer, n_trimmed - 1)
        if s0 <= e0:
            ignore_trimmed.append((s0, e0))
    for (start, end) in (flags or []):
        s0 = max(start - buffer, 0)
        e0 = min(end - buffer, n_trimmed - 1)
        if s0 <= e0:
            ignore_trimmed.append((s0, e0))

    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s0, e0 in ignore_trimmed:
        mask[s0:e0 + 1] = False
    valid_idxs_masked = np.nonzero(mask)[0]

    def _overlap_stats(a_orig: int, b_orig: int, ignore_ranges: List[Tuple[int, int]]) -> float:
        if b_orig <= a_orig:
            return 0.0
        win_len = (b_orig - a_orig + 1)
        overlap = 0
        for s, e in ignore_ranges:
            lo = max(a_orig, s)
            hi = min(b_orig, e)
            if hi >= lo:
                overlap += (hi - lo + 1)
        return overlap / max(win_len, 1)

    def _score_window_on_trim(i: int, j: int) -> Tuple[float, np.ndarray, np.ndarray]:
        inside = np.arange(i, j + 1, dtype=np.int64)
        outside = np.setdiff1d(all_trim, inside, assume_unique=True)

        sri = _nwkr_sse_on_subset(row_trimmed, K_trim, inside)
        sro = _nwkr_sse_on_subset(row_trimmed, K_trim, outside)

        # your score definition
        sc = 1.0 - (sri + sro) / sra_trim
        # for visualization of inside fit
        sri_vals = _nwkr_predict_subset(row_trimmed, K_trim, inside)
        return sc, inside, sri_vals

    # ---- variable-length search: brute force over contiguous windows up to range_cap ----
    def _varlen_search(valid: np.ndarray):
        best_sc = -np.inf
        best_win = (0, 0)
        best_idx_full = None
        best_vals = None

        if valid.size < 2:
            return best_win, 0.0, None, None

        # brute force: try each start among valid, grow contiguously
        for pos_i in range(valid.size):
            i = int(valid[pos_i])
            # require that i begins a contiguous run (same idea as your code)
            if pos_i + 1 < valid.size and valid[pos_i + 1] != i + 1:
                continue

            max_k = min(pos_i + range_cap, valid.size - 1)
            for k in range(pos_i + 1, max_k + 1):
                if valid[k] != valid[k - 1] + 1:
                    break
                j = int(valid[k])

                sc, inside, sri_vals = _score_window_on_trim(i, j)
                if sc > best_sc:
                    best_sc = sc
                    best_win = (i, j)
                    best_idx_full = inside + buffer
                    best_vals = sri_vals

        (oi, oj) = best_win
        return (oi + buffer, oj + buffer), float(best_sc), best_idx_full, best_vals

    best_win_masked, best_sc_masked, sri_idx_masked, sri_vals_masked = _varlen_search(valid_idxs_masked)
    best_win_unmasked, best_sc_unmasked, sri_idx_unmasked, sri_vals_unmasked = _varlen_search(all_trim)
    overlap_pct_unmasked = _overlap_stats(best_win_unmasked[0], best_win_unmasked[1], ignore or [])

    # ---- fixed-length sweep on full row: brute force, no incremental updates ----
    def _fixedlen_sweep_full():
        best_sc = -np.inf
        best_win = (0, 0)
        best_idx = None
        best_vals = None

        if window_bins <= 1 or window_bins > n:
            return best_win, 0.0, None, None

        for i in range(0, n - window_bins + 1):
            j = i + window_bins - 1
            inside = np.arange(i, j + 1, dtype=np.int64)
            outside = np.setdiff1d(all_full, inside, assume_unique=True)

            sri = _nwkr_sse_on_subset(row, K_full, inside)
            sro = _nwkr_sse_on_subset(row, K_full, outside)
            sc = 1.0 - (sri + sro) / sra_full

            if sc > best_sc:
                best_sc = sc
                best_win = (i, j)
                best_idx = inside
                best_vals = _nwkr_predict_subset(row, K_full, inside)

        return best_win, float(best_sc), best_idx, best_vals

    best_win_fixed, best_sc_fixed, sri_idx_fixed, sri_vals_fixed = _fixedlen_sweep_full()
    overlap_pct_fixed = _overlap_stats(best_win_fixed[0], best_win_fixed[1], ignore or [])

    return (
        row_idx, best_win_masked, best_sc_masked, pred_array, sri_idx_masked, sri_vals_masked, w * sr_factor, range_cap * sr_factor,
        best_win_unmasked, best_sc_unmasked, overlap_pct_unmasked, sri_idx_unmasked, sri_vals_unmasked,
        best_win_fixed, best_sc_fixed, overlap_pct_fixed, sri_idx_fixed, sri_vals_fixed, window_bins * sr_factor
    )

def scan_row_with_nwkr(params: Tuple[int, np.ndarray, List[Tuple[int,int]], List[Tuple[int,int]], np.ndarray, int, int]):
    row_idx, row, ignore, flags, freqs, buffer, sr_factor = params[:7]

    w_override = None
    range_cap_override = None
    fixed_bins_override = None

    if len(params) >= 8:
        w_override = params[7]
    if len(params) >= 9:
        range_cap_override = params[8]
    if len(params) >= 10:
        fixed_bins_override = params[9]

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
    window_bins_auto = int(math.floor(R)) + 1
    

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
        k_vector = truncated_kernel_vector(w=float(w), r=range_cap, kind="gaussian")
        sra_full, _, _, _, _, _ = calculate_gaussian_sra_trunc(row, k_vector)
        sra, _, pred_array, _, _, _ = calculate_gaussian_sra_trunc(row_trimmed, k_vector)
    else:
        sigma = float(max(w, 1))
        k_vector = truncated_kernel_vector(w=float(w), r=range_cap, kind="laplace")
        sra_full, _, _, _ = calculate_laplace_sra_fast_trunc(row, sigma, range_cap)
        sra, _, pred_array, _ = calculate_laplace_sra_fast_trunc(row_trimmed, sigma, range_cap)
    
    sra_full = max(sra_full, 1e-12)
    sra = sra if sra > 1e-12 else 1e-12    

    ignore_trimmed: List[Tuple[int, int]] = []
    for (start, end) in ignore:
        s0 = max(start - buffer, 0)
        e0 = min(end - buffer, n_trimmed - 1)
        if s0 <= e0:
            ignore_trimmed.append((s0, e0))

    for (start, end) in flags:
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
        best_sc_var = 0
        best_win_var = (0, 0)
        best_idx_full_var = None
        best_vals_var = None

        if len(valid) < 2:
            return best_win_var, best_sc_var, best_idx_full_var, best_vals_var

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

                new_idx = j

                if not g_in_initialized:
                    g_in_idxs, g_in_num, g_in_den, g_in_sse = kernel_state_init_trunc(
                        row_trimmed, inside, k_vector
                    )

                    g_in_initialized = True

                    outside_init = np.setdiff1d(all_trimmed, inside, assume_unique=True)
                    g_out_idxs, g_out_num, g_out_den, g_out_sse = kernel_state_init_trunc(
                        row_trimmed, outside_init, k_vector
                    )

                else:
                    g_in_idxs, g_in_num, g_in_den, g_in_sse = kernel_state_add_point_trunc(
                        row_trimmed,
                        g_in_idxs,
                        g_in_num,
                        g_in_den,
                        g_in_sse,
                        new_idx,
                        k_vector,
                    )
                    g_out_idxs, g_out_num, g_out_den, g_out_sse = kernel_state_remove_point_trunc(
                        row_trimmed,
                        g_out_idxs,
                        g_out_num,
                        g_out_den,
                        g_out_sse,
                        new_idx,
                        k_vector,
                    )

                sri = g_in_sse
                sro = g_out_sse
                sc = -(sri + sro)
                sc = sc / sra + 1.0

                if sc > best_sc_var:
                    best_sc_var = sc
                    best_win_var = (i, j)
                    best_idx_full_var = inside + buffer
                    best_vals_var = predict_on_idxs_trunc(
                        row_trimmed,
                        inside,
                        k_vector
                    )

        oi, oj = best_win_var
        return (oi + buffer, oj + buffer), best_sc_var, best_idx_full_var, best_vals_var

    def _fixedlen_sweep() -> Tuple[Tuple[int, int], float, np.ndarray | None, np.ndarray | None]:
        best_sc_fix = 0
        best_win_fix = (0, 0)
        best_idx_full_fix = None
        best_vals_fix = None

        if window_bins <= 0 or window_bins > n:
            return best_win_fix, best_sc_fix, best_idx_full_fix, best_vals_fix

        max_start = n - window_bins

        g_in_initialized = False

        g_in_idxs = g_in_num = g_in_den = None
        g_out_idxs = g_out_num = g_out_den = None
        g_in_sse = 0.0
        g_out_sse = 0.0

        for i in range(max_start + 1):
            j = i + window_bins - 1
            inside = np.arange(i, i + window_bins, dtype=np.int64)

            if not g_in_initialized:
                g_in_idxs, g_in_num, g_in_den, g_in_sse = kernel_state_init_trunc(
                    row,
                    inside,
                    k_vector,
                )
                g_in_initialized = True

                outside_init = np.setdiff1d(all_full, inside, assume_unique=True)
                g_out_idxs, g_out_num, g_out_den, g_out_sse = kernel_state_init_trunc(
                    row,
                    outside_init,
                    k_vector,
                )

            else:
                rem_idx_inside = i - 1
                add_idx_inside = j

                if rem_idx_inside >= 0:
                    g_in_idxs, g_in_num, g_in_den, g_in_sse = kernel_state_remove_point_trunc(
                        row,
                        g_in_idxs,
                        g_in_num,
                        g_in_den,
                        g_in_sse,
                        rem_idx_inside,
                        k_vector,
                    )
                g_in_idxs, g_in_num, g_in_den, g_in_sse = kernel_state_add_point_trunc(
                    row,
                    g_in_idxs,
                    g_in_num,
                    g_in_den,
                    g_in_sse,
                    add_idx_inside,
                    k_vector,
                )

                g_out_idxs, g_out_num, g_out_den, g_out_sse = kernel_state_remove_point_trunc(
                    row,
                    g_out_idxs,
                    g_out_num,
                    g_out_den,
                    g_out_sse,
                    add_idx_inside,
                    k_vector,
                )
                if rem_idx_inside >= 0:
                    g_out_idxs, g_out_num, g_out_den, g_out_sse = kernel_state_add_point_trunc(
                        row,
                        g_out_idxs,
                        g_out_num,
                        g_out_den,
                        g_out_sse,
                        rem_idx_inside,
                        k_vector,
                    )

            sri = g_in_sse
            sro = g_out_sse
            sc = -(sri + sro)
            sc = sc / sra_full + 1.0

            if sc > best_sc_fix:
                best_sc_fix = sc
                best_win_fix = (i, j)
                best_idx_full_fix = inside
                best_vals_fix = predict_on_idxs_trunc(
                    row,
                    inside,
                    k_vector
                )

        return best_win_fix, best_sc_fix, best_idx_full_fix, best_vals_fix




    best_win_masked, best_sc_masked, sri_idx_masked, sri_vals_masked = _varlen_search(valid_idxs_masked)

    best_win_unmasked, best_sc_unmasked, sri_idx_unmasked, sri_vals_unmasked = _varlen_search(all_trimmed)
    overlap_pct_unmasked = _overlap_stats(best_win_unmasked[0], best_win_unmasked[1], ignore)

    best_win_fixed, best_sc_fixed, sri_idx_fixed, sri_vals_fixed, overlap_pct_fixed = (0, 0), 0, None, None, 0
    
    best_win_fixed, best_sc_fixed, sri_idx_fixed, sri_vals_fixed = _fixedlen_sweep()
    overlap_pct_fixed = _overlap_stats(best_win_fixed[0], best_win_fixed[1], ignore)

    return (
        row_idx, best_win_masked, best_sc_masked, pred_array, sri_idx_masked, sri_vals_masked, w * sr_factor, range_cap * sr_factor,
        best_win_unmasked, best_sc_unmasked, overlap_pct_unmasked, sri_idx_unmasked, sri_vals_unmasked,
        best_win_fixed, best_sc_fixed, overlap_pct_fixed, sri_idx_fixed, sri_vals_fixed, window_bins * sr_factor
    )
