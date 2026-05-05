from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple
from .config import ref_freq, get_kernel_kind, KernelKind
from .kernels import _kernel_matrix, truncated_kernel_vector, get_kernel as get_kernel
from .kernel_optimized_state import (
    buf_init, buf_add, buf_remove,
    nin_din_init_full, nin_din_add, nin_din_remove,
    sse_out_from_nin_din, sse_out_add, sse_out_remove,
)
from .scoring import (
    _nwkr_predict_subset,
    _nwkr_sse_on_subset,
    calculate_gaussian_sra_trunc,
    calculate_laplace_sra_fast_trunc,
)
from .regressors import (
    build_regressor_sra,
    predict_on_idxs_mean,
    predict_on_idxs_poly,
    predict_on_idxs_krr,
)
from .predictors import predict_on_idxs_trunc


def predict_subset_using_regressor(x, idxs, family, **kwargs):
    family = family.lower()
    if family == "mean":   return predict_on_idxs_mean(x, idxs)
    if family == "poly":   return predict_on_idxs_poly(x, idxs,
                               degree=int(kwargs.get("degree", 1)), reg=float(kwargs.get("reg", 1e-8)))
    if family == "krr":    return predict_on_idxs_krr(x, idxs,
                               kernel=kwargs.get("kernel", "rbf"),
                               kernel_param=float(kwargs.get("kernel_param", 1.0)),
                               reg=float(kwargs.get("reg", 1e-3)))
    raise ValueError(f"Unknown family {family!r}")


def scan_row_with_regressor(x, w, family="mean", **kwargs):
    x = x.astype(np.float64); n = x.shape[0]
    sra_all, _, _, _, _, _, _ = build_regressor_sra(x, family, **kwargs)
    best_score = -np.inf; best_a = best_b = -1
    for a in range(n):
        for b in range(a, min(n - 1, a + w - 1) + 1):
            inside = np.arange(a, b + 1, dtype=np.int64)
            if a == 0 and b == n - 1: continue
            outside = (np.arange(b + 1, n, dtype=np.int64) if a == 0 else
                       np.arange(0, a, dtype=np.int64) if b == n - 1 else
                       np.concatenate((np.arange(0, a, dtype=np.int64),
                                       np.arange(b + 1, n, dtype=np.int64))))
            ri = x[inside]  - predict_subset_using_regressor(x, inside,  family, **kwargs)
            ro = x[outside] - predict_subset_using_regressor(x, outside, family, **kwargs)
            sc = sra_all - float(ri @ ri) - float(ro @ ro)
            if sc > best_score: best_score = sc; best_a = a; best_b = b
    return best_score, best_a, best_b


def scan_row_with_nwkr_naive(params: Tuple):
    row_idx, row, ignore, flags, freqs, buffer, sr_factor = params[:7]
    w_override          = params[7]  if len(params) >= 8  else None
    range_cap_override  = params[8]  if len(params) >= 9  else None
    fixed_bins_override = params[9]  if len(params) >= 10 else None
    row = np.asarray(row, dtype=np.float64); n = row.size
    _bad = (row_idx,(0,-1),0.,np.array([]),None,None,0,0,
            (0,-1),0.,0.,None,None,(0,-1),0.,0.,None,None,0)
    if len(freqs) < 2 or not np.isfinite(freqs[:2]).all(): return _bad
    freq_step = abs(freqs[1] - freqs[0]); L = len(freqs)
    R = ref_freq / (freq_step if freq_step > 0 else ref_freq)
    w_auto = int(round(max(3, min(R, L / 16)))); range_cap_auto = 3 * w_auto
    window_bins_auto = int(math.floor(R)) + 1
    w           = int(w_override)           if (w_override           is not None and int(w_override)           > 0) else w_auto
    range_cap   = int(range_cap_override)   if (range_cap_override   is not None and int(range_cap_override)   > 0) else range_cap_auto
    window_bins = int(fixed_bins_override)  if (fixed_bins_override  is not None and int(fixed_bins_override)  > 0) else window_bins_auto
    row_trimmed = row[buffer: n - buffer]; n_trimmed = row_trimmed.size
    if n_trimmed <= 2: return _bad
    kernel_kind = get_kernel_kind()
    kind = "gaussian" if kernel_kind == KernelKind.GAUSSIAN else "laplace"
    K_full = _kernel_matrix(n, float(w), kind); K_trim = _kernel_matrix(n_trimmed, float(w), kind)
    all_full = np.arange(n, dtype=np.int64); all_trim = np.arange(n_trimmed, dtype=np.int64)
    sra_full = max(_nwkr_sse_on_subset(row, K_full, all_full), 1e-12)
    sra_trim = max(_nwkr_sse_on_subset(row_trimmed, K_trim, all_trim), 1e-12)
    pred_array = _nwkr_predict_subset(row_trimmed, K_trim, all_trim)
    ignore_trimmed = []
    for s, e in (ignore or []):
        s0 = max(s - buffer, 0); e0 = min(e - buffer, n_trimmed - 1)
        if s0 <= e0: ignore_trimmed.append((s0, e0))
    for s, e in (flags or []):
        s0 = max(s - buffer, 0); e0 = min(e - buffer, n_trimmed - 1)
        if s0 <= e0: ignore_trimmed.append((s0, e0))
    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s0, e0 in ignore_trimmed: mask[s0:e0 + 1] = False
    valid_idxs_masked = np.nonzero(mask)[0]
    def _ov(a, b, rngs):
        if b <= a: return 0.
        return sum(max(0, min(b,e)-max(a,s)+1) for s,e in rngs)/(b-a+1)
    def _score(i, j):
        ins = np.arange(i, j+1, dtype=np.int64); out = np.setdiff1d(all_trim, ins, assume_unique=True)
        sc = 1. - ((_nwkr_sse_on_subset(row_trimmed,K_trim,ins)+_nwkr_sse_on_subset(row_trimmed,K_trim,out))/sra_trim)
        return sc, ins, _nwkr_predict_subset(row_trimmed, K_trim, ins)
    def _varlen(valid):
        best_sc=-np.inf; best_win=(0,0); best_idx=None; best_vals=None
        if valid.size<2: return best_win,0.,None,None
        for pi in range(valid.size):
            i=int(valid[pi])
            if pi+1<valid.size and valid[pi+1]!=i+1: continue
            for kk in range(pi+1,min(pi+range_cap,valid.size-1)+1):
                if valid[kk]!=valid[kk-1]+1: break
                j=int(valid[kk]); sc,ins,vals=_score(i,j)
                if sc>best_sc: best_sc=sc;best_win=(i,j);best_idx=ins+buffer;best_vals=vals
        oi,oj=best_win; return (oi+buffer,oj+buffer),float(best_sc),best_idx,best_vals
    def _fixed():
        best_sc=-np.inf; best_win=(0,0); best_idx=None; best_vals=None
        if window_bins<=1 or window_bins>n: return best_win,0.,None,None
        for i in range(n-window_bins+1):
            j=i+window_bins-1; ins=np.arange(i,j+1,dtype=np.int64); out=np.setdiff1d(all_full,ins,assume_unique=True)
            sc=1.-((_nwkr_sse_on_subset(row,K_full,ins)+_nwkr_sse_on_subset(row,K_full,out))/sra_full)
            if sc>best_sc: best_sc=sc;best_win=(i,j);best_idx=ins;best_vals=_nwkr_predict_subset(row,K_full,ins)
        return best_win,float(best_sc),best_idx,best_vals
    wm,sm,im,vm=_varlen(valid_idxs_masked)
    wu,su,iu,vu=_varlen(all_trim); ovl_u=_ov(wu[0],wu[1],ignore or [])
    wf,sf,if_,vf=_fixed(); ovl_f=_ov(wf[0],wf[1],ignore or [])
    return (row_idx,wm,sm,pred_array,im,vm,w*sr_factor,range_cap*sr_factor,
            wu,su,ovl_u,iu,vu,wf,sf,ovl_f,if_,vf,window_bins*sr_factor)


def scan_row_with_nwkr(params: Tuple):
    row_idx, row, ignore, flags, freqs, buffer, sr_factor = params[:7]
    w_override          = params[7]  if len(params) >= 8 and params[7] is not None  else None
    kernel_cap_override  = params[8]  if len(params) >= 9 and params[8] is not None else None
    fixed_bins_override = params[9]  if len(params) >= 10 and params[9] is not None else None
    masked_search_flag = params[10]  if len(params) >= 11 and params[10] is not None else True
    unmasked_search_flag = params[11]  if len(params) >= 12 and params[11] is not None else True
    fixed_search_flag = params[12]  if len(params) >= 13 and params[12] is not None else True

    n = row.shape[0]; kernel_kind = get_kernel_kind()
    _bad = (row_idx,(0,-1),0.,np.array([]),None,None,0,0,
            (0,-1),0.,0.,None,None,(0,-1),0.,0.,None,None,0)

    def _ov(a, b, rngs):
        if b <= a: return 0.
        wl = b-a+1; ov = 0
        for s,e in rngs:
            lo=max(a,s); hi=min(b,e)
            if hi>=lo: ov+=hi-lo+1
        return ov/wl

    if len(freqs) < 2 or not np.isfinite(freqs[:2]).all(): return _bad

    freq_step = abs(freqs[1] - freqs[0]); L = len(freqs)
    R = ref_freq / (freq_step if freq_step > 0 else 1.)
    w_auto = int(round(max(3, min(R, L / 16))))
    kernel_cap_auto = 2 * w_auto
    range_cap_auto = 3 * w_auto
    window_bins_auto = int(math.floor(R)) + 1

    w           = int(w_override)           if (w_override           is not None and int(w_override)           > 0) else w_auto
    range_cap = int(w_override * 3)   if (w_override   is not None and int(w_override)   > 0) else range_cap_auto
    kernel_cap   = int(kernel_cap_override)   if (kernel_cap_override   is not None and int(kernel_cap_override)   > 0) else kernel_cap_auto
    window_bins = int(fixed_bins_override)  if (fixed_bins_override  is not None and int(fixed_bins_override)  > 0) else window_bins_auto
    
    row_trimmed = row[buffer: len(row) - buffer]; n_trimmed = row_trimmed.shape[0]
    if n_trimmed <= 0: return _bad

    if kernel_kind == KernelKind.GAUSSIAN:
        k_vector = truncated_kernel_vector(w=float(w), r=kernel_cap, kind="gaussian")
        sra_full,_,_,numer_all_full,denom_all_full,_ = calculate_gaussian_sra_trunc(row, k_vector)
        sra,_,pred_array,numer_all,denom_all,_       = calculate_gaussian_sra_trunc(row_trimmed, k_vector)
    else:
        from .scoring import _laplace_accum_trunc_1d
        sigma = float(max(w, 1))
        k_vector = truncated_kernel_vector(w=float(w), r=kernel_cap, kind="laplace")
        sra_full,_,_,_    = calculate_laplace_sra_fast_trunc(row, sigma, kernel_cap)
        sra,_,pred_array,_ = calculate_laplace_sra_fast_trunc(row_trimmed, sigma, kernel_cap)
        numer_all_full = _laplace_accum_trunc_1d(row.astype(np.float64), sigma, kernel_cap)
        denom_all_full = _laplace_accum_trunc_1d(np.ones(n, dtype=np.float64), sigma, kernel_cap)
        numer_all      = _laplace_accum_trunc_1d(row_trimmed.astype(np.float64), sigma, kernel_cap)
        denom_all      = _laplace_accum_trunc_1d(np.ones(n_trimmed, dtype=np.float64), sigma, kernel_cap)

    sra_full = max(sra_full, 1e-12); sra = sra if sra > 1e-12 else 1e-12

    ignore_trimmed: List[Tuple[int,int]] = []
    for s,e in ignore:
        s0=max(s-buffer,0); e0=min(e-buffer,n_trimmed-1)
        if s0<=e0: ignore_trimmed.append((s0,e0))
    for s,e in flags:
        s0=max(s-buffer,0); e0=min(e-buffer,n_trimmed-1)
        if s0<=e0: ignore_trimmed.append((s0,e0))

    mask = np.ones(n_trimmed, dtype=np.bool_)
    for s0,e0 in ignore_trimmed: mask[s0:e0+1] = False
    all_full    = np.arange(n, dtype=np.int64)
    all_trimmed = np.arange(n_trimmed, dtype=np.int64)
    valid_idxs_masked = np.nonzero(mask)[0]

    REFRESH = max(1, range_cap)

    # ------------------------------------------------------------------
    # _varlen_search
    #
    # Cost analysis:
    #   nin/din carried across ALL outer steps via O(r) add/remove.
    #   sse_out carried and updated O(r) per inner step.
    #   buf/sse_in reinited per outer step: O(r) (only 2 points in [i,j]).
    #   sse_out_from_nin_din called once at first outer step: O(n).
    #   After that: all updates are O(r).
    #
    #   Total: O(n) one-time + O(n * r) outer-step nin/din transitions
    #          + O(n * w * r) inner grows = O(n * w * r).
    # ------------------------------------------------------------------
    def _varlen_search(valid: np.ndarray):
        best_sc_var       = 0.
        best_win_var      = (0, -1)
        best_idx_full_var = None
        best_vals_var     = None

        if len(valid) < 2:
            return best_win_var, best_sc_var, best_idx_full_var, best_vals_var

        n_valid = valid.shape[0]

        cap      = range_cap + 2
        buf_idxs = np.empty(cap, dtype=np.int64)
        buf_num  = np.empty(cap, dtype=np.float64)
        buf_den  = np.empty(cap, dtype=np.float64)

        # nin/din reflect the inside set of the CURRENT growing window.
        # They are updated incrementally at O(r) per add/remove.
        nin = np.zeros(n_trimmed, dtype=np.float64)
        din = np.zeros(n_trimmed, dtype=np.float64)

        # carry: what nin/din/sse_out reflect at end of each outer step.
        # At end of outer step i, nin/din reflect [i .. right_end].
        # At start of outer step i+1, we remove i from nin/din (O(r)),
        # then nin/din reflect [i+1 .. right_end].
        carry_valid     = False
        carry_left_idx  = -1
        carry_right_idx = -1
        carry_sse_out   = 0.

        steps_since_refresh = 0

        for pos_i in range(n_valid):
            i = int(valid[pos_i])

            if pos_i + 1 < n_valid and valid[pos_i + 1] != i + 1:
                carry_valid = False
                continue

            use_carry = carry_valid and carry_left_idx == i - 1

            # ---- Update nin/din for new left edge ----
            if use_carry:
                # Remove (i-1) from nin/din — O(r)
                nin_din_remove(row_trimmed, nin, din, i - 1, k_vector)
                # nin/din now reflect [i .. carry_right_idx]
            else:
                # Run boundary: rebuild nin/din from empty — O(n)
                # (will be populated by nin_din_add inside the inner loop)
                for ii in range(n_trimmed):
                    nin[ii] = 0.0; din[ii] = 0.0
                carry_sse_out = 0.
                steps_since_refresh = 0

            # ---- Inner loop: grow window [i..j] right ----
            g_in_initialized = False
            m      = 0
            sse_in = 0.
            # sse_out starts from carry_sse_out, but after nin/din update.
            # If use_carry: nin/din reflect [i..carry_right_idx], not [i..first_j].
            # We'll recompute sse_out exactly at first init below.
            sse_out = 0.

            max_k = min(pos_i + range_cap, n_valid - 1)

            for kk in range(pos_i + 1, max_k + 1):
                if valid[kk] != valid[kk - 1] + 1:
                    break
                j = int(valid[kk])

                if not g_in_initialized:
                    # First j of this outer step.
                    # nin/din currently reflect either empty (no carry) or
                    # [i..carry_right_idx] (carry). We need them to reflect [i,j].
                    # Easiest: add i and j to nin/din (they may or may not be in).
                    # Since we zeroed on no-carry, or removed (i-1) on carry,
                    # the current state could have stale entries if carry_right_idx > j.
                    # Safest: rebuild nin/din for [i,j] from scratch.
                    nin_din_init_full(
                        row_trimmed,
                        np.array([i, j], dtype=np.int64),
                        k_vector, nin, din)
                    m, sse_in = buf_init(
                        row_trimmed,
                        np.array([i, j], dtype=np.int64),
                        k_vector, buf_idxs, buf_num, buf_den)
                    sse_out = sse_out_from_nin_din(
                        row_trimmed, numer_all, denom_all, nin, din, buf_idxs, m)
                    steps_since_refresh = 0
                    g_in_initialized = True

                else:
                    # Grow right: O(r) each
                    nin_din_add(row_trimmed, nin, din, j, k_vector)
                    m, sse_in = buf_add(
                        row_trimmed, j, k_vector, buf_idxs, buf_num, buf_den, m, sse_in)
                    sse_out = sse_out_add(
                        row_trimmed, numer_all, denom_all, nin, din,
                        buf_idxs, m, j, k_vector, sse_out)
                    steps_since_refresh += 1
                    if steps_since_refresh >= REFRESH:
                        sse_out = sse_out_from_nin_din(
                            row_trimmed, numer_all, denom_all, nin, din, buf_idxs, m)
                        steps_since_refresh = 0

                sc = 1. - (sse_in + sse_out) / sra
                if sc > best_sc_var:
                    best_sc_var       = sc
                    best_win_var      = (i, j)
                    best_idx_full_var = buf_idxs[:m].copy() + buffer
                    best_vals_var     = predict_on_idxs_trunc(
                        row_trimmed, buf_idxs[:m].copy(), k_vector)

            if g_in_initialized:
                carry_valid     = True
                carry_left_idx  = int(i)
                carry_right_idx = int(buf_idxs[m - 1]) if m > 0 else i
                carry_sse_out   = sse_out
                # nin/din now reflect [i..carry_right_idx]
            else:
                carry_valid = False

        oi, oj = best_win_var
        if (oj == -1):
            print("Catching here.")
            print(sc)
        return (oi + buffer, oj + buffer), best_sc_var, best_idx_full_var, best_vals_var

    # ------------------------------------------------------------------
    # _fixedlen_sweep — O(n*r) total
    # ------------------------------------------------------------------
    def _fixedlen_sweep():
        best_sc_fix       = 0.
        best_win_fix      = (0, -1)
        best_idx_full_fix = None
        best_vals_fix     = None

        if window_bins <= 0 or window_bins > n:
            return best_win_fix, best_sc_fix, best_idx_full_fix, best_vals_fix

        cap      = window_bins + 1
        buf_idxs = np.empty(cap, dtype=np.int64)
        buf_num  = np.empty(cap, dtype=np.float64)
        buf_den  = np.empty(cap, dtype=np.float64)
        nin      = np.zeros(n, dtype=np.float64)
        din      = np.zeros(n, dtype=np.float64)

        m = 0; sse_in = 0.; sse_out = 0.
        g_in_initialized = False; steps = 0

        for i in range(n - window_bins + 1):
            j      = i + window_bins - 1
            inside = np.arange(i, i + window_bins, dtype=np.int64)

            if not g_in_initialized:
                nin_din_init_full(row, inside, k_vector, nin, din)
                m, sse_in = buf_init(row, inside, k_vector, buf_idxs, buf_num, buf_den)
                sse_out = sse_out_from_nin_din(
                    row, numer_all_full, denom_all_full, nin, din, buf_idxs, m)
                steps = 0; g_in_initialized = True
            else:
                rem = np.int64(i - 1); add = np.int64(j)
                # Remove left edge
                nin_din_remove(row, nin, din, rem, k_vector)
                m, sse_in = buf_remove(row, rem, k_vector, buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = sse_out_remove(
                    row, numer_all_full, denom_all_full, nin, din, buf_idxs, m, rem, k_vector, sse_out)
                # Add right edge
                nin_din_add(row, nin, din, add, k_vector)
                m, sse_in = buf_add(row, add, k_vector, buf_idxs, buf_num, buf_den, m, sse_in)
                sse_out = sse_out_add(
                    row, numer_all_full, denom_all_full, nin, din, buf_idxs, m, add, k_vector, sse_out)
                steps += 1
                if steps >= REFRESH:
                    sse_out = sse_out_from_nin_din(
                        row, numer_all_full, denom_all_full, nin, din, buf_idxs, m)
                    steps = 0

            sc = 1. - (sse_in + sse_out) / sra_full
            if sc > best_sc_fix:
                best_sc_fix       = sc
                best_win_fix      = (i, j)
                best_idx_full_fix = inside
                best_vals_fix     = predict_on_idxs_trunc(row, inside, k_vector)

        return best_win_fix, best_sc_fix, best_idx_full_fix, best_vals_fix

    if masked_search_flag:
        wm, sm, im, vm = _varlen_search(valid_idxs_masked)
    else:
        wm, sm, im, vm = (0, -1), 0., None, None
    if unmasked_search_flag:
        wu, su, iu, vu = _varlen_search(all_trimmed)
        ovl_u = _ov(wu[0], wu[1], ignore)
    else:
        wu, su, iu, vu = (0, -1), 0., None, None
        ovl_u = 0.
    if fixed_search_flag:
        wf, sf, if_, vf = _fixedlen_sweep()
        ovl_f = _ov(wf[0], wf[1], ignore)
    else:
        wf, sf, if_, vf = (0, -1), 0., None, None
        ovl_f = 0.
    

    return (row_idx, wm, sm, pred_array, im, vm, w * sr_factor, range_cap * sr_factor,
            wu, su, ovl_u, iu, vu, wf, sf, ovl_f, if_, vf, window_bins * sr_factor)