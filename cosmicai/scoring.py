from __future__ import annotations
import math
import numpy as np
from numba import njit
from .config import KernelKind
from typing import Optional
from .predictors import predict_on_idxs_laplace, predict_on_idxs_trunc

def _nwkr_predict_subset(y: np.ndarray, K: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """
    Predict y_hat on 'subset' using NWKR trained on 'subset':
      y_hat[i] = sum_{j in subset} K[i,j] y[j] / sum_{j in subset} K[i,j]
    evaluated only for i in subset (so output length = len(subset)).

    This is the most direct analogue of your inside/outside fits.
    """
    # weights submatrix: K[subset, subset]
    W = K[np.ix_(subset, subset)]  # shape (m,m)
    denom = W.sum(axis=1)
    denom = np.maximum(denom, 1e-12)
    numer = W @ y[subset]
    return numer / denom

def _nwkr_sse_on_subset(y: np.ndarray, K: np.ndarray, subset: np.ndarray) -> float:
    if subset.size == 0:
        return 0.0
    yhat = _nwkr_predict_subset(y, K, subset)
    resid = y[subset] - yhat
    return float(np.dot(resid, resid))

@njit(cache=True, fastmath=True)
def calculate_gaussian_sra(array: np.ndarray, W: np.ndarray):
    n = array.shape[0]
    numer = np.empty(n, dtype=array.dtype)
    denom = np.empty(n, dtype=array.dtype)
    for i in range(n):
        s_num = 0.0; s_den = 0.0
        for j in range(n):
            w_ij = W[i, j]
            s_num += w_ij * array[j]
            s_den += w_ij
        numer[i] = s_num; denom[i] = s_den
    ssr = 0.0
    ssr_array = np.empty(n, dtype=array.dtype)
    pred_array = np.empty(n, dtype=array.dtype)
    for i in range(n):
        pred = numer[i] / denom[i] if denom[i] > 0.0 else 0.0
        pred_array[i] = pred
        diff = array[i] - pred
        ssr_array[i] = diff * diff
        ssr += ssr_array[i]
    return ssr, ssr_array, pred_array

@njit(cache=True, fastmath=True)
def calculate_gaussian_sra_with_nd(array: np.ndarray, W: np.ndarray, denom_pre: Optional[np.ndarray] = None):
    n = array.shape[0]
    numer = W @ array
    denom = denom_pre[:n]
    eps = 1e-12
    pred = numer / np.maximum(denom, eps)
    ssr_array = (array - pred) ** 2
    sra = 0.0
    for i in range(n):
        sra += ssr_array[i]

    ssr_ps = np.empty(n + 1, dtype=np.float64)
    ssr_ps[0] = 0.0
    acc = 0.0
    for i in range(n):
        acc += ssr_array[i]
        ssr_ps[i + 1] = acc

    return sra, ssr_array, pred, numer, denom, ssr_ps

@njit(cache=True, fastmath=True)
def calculate_gaussian_sra_trunc(
    array: np.ndarray,
    k: np.ndarray,
):
    n = array.shape[0]
    r = k.shape[0] - 1
    eps = 1e-12

    numer = np.empty(n, dtype=np.float64)
    denom = np.empty(n, dtype=np.float64)
    pred  = np.empty(n, dtype=np.float64)
    ssr_array = np.empty(n, dtype=np.float64)

    for i in range(n):
        j0 = i - r
        if j0 < 0:
            j0 = 0
        j1 = i + r
        if j1 > n - 1:
            j1 = n - 1

        num = 0.0
        den = 0.0
        for j in range(j0, j1 + 1):
            d = i - j
            if d < 0:
                d = -d
            wgt = k[d]
            den += wgt
            num += wgt * array[j]

        numer[i] = num
        denom[i] = den
        pi = num / (den if den > eps else eps)
        pred[i] = pi
        di = array[i] - pi
        ssr_array[i] = di * di

    sra = 0.0
    for i in range(n):
        sra += ssr_array[i]

    ssr_ps = np.empty(n + 1, dtype=np.float64)
    ssr_ps[0] = 0.0
    acc = 0.0
    for i in range(n):
        acc += ssr_array[i]
        ssr_ps[i + 1] = acc

    return sra, ssr_array, pred, numer, denom, ssr_ps


@njit(cache=True, fastmath=True)
def gaussian_sro_nearband(
    x: np.ndarray,
    inside: np.ndarray,
    a: int,
    b: int,
    range_cap: int,
    W: np.ndarray,
    ssr_ps: np.ndarray,
    numer_all: np.ndarray,
    denom_all: np.ndarray,
    eps: float = 1e-12,
) -> float:
    n = x.shape[0]
    lo = max(0, a - (range_cap // 2))
    hi = min(n - 1, (b + range_cap //2))

    sro_far = 0.0
    if lo > 0:
        sro_far += float(ssr_ps[lo] - ssr_ps[0])
    if hi < n - 1:
        sro_far += float(ssr_ps[n] - ssr_ps[hi + 1])

    sro_near = 0.0
    for i0 in range(lo, hi + 1):
        if a <= i0 <= b:
            continue
        num_in = 0.0
        den_in = 0.0
        for t in range(inside.shape[0]):
            j = int(inside[t])
            wij = float(W[i0, j])
            num_in += wij * float(x[j])
            den_in += wij

        num_out = float(numer_all[i0]) - num_in
        den_out = float(denom_all[i0]) - den_in

        pred = num_out / den_out if den_out > eps else 0.0
        diff = x[i0] - pred
        sro_near += diff * diff

    return sro_far + sro_near

@njit(cache=True, fastmath=True)
def _ssr_region_gaussian(array: np.ndarray, idxs: np.ndarray, W: np.ndarray,
                      ssr_array: np.ndarray, a: int, b: int, range_cap: int) -> float:
    m = idxs.shape[0]
    sri = 0.0; sro_far = 0.0; sro_near = 0.0
    low_cut = a - range_cap // 2; high_cut = b + range_cap // 2
    for ii in range(m):
        i0 = idxs[ii]
        if a <= i0 <= b:
            num = 0.0; den = 0.0
            for jj in range(m):
                j0 = idxs[jj]
                w_ij = W[i0, j0]
                num += w_ij * array[j0]
                den += w_ij
            diff = array[i0] - (num / den if den > 1e-12 else 0.0)
            sri += diff * diff
        elif i0 < low_cut or i0 > high_cut:
            sro_far += ssr_array[i0]
        else:
            num = 0.0; den = 0.0
            for jj in range(m):
                j0 = idxs[jj]
                w_ij = W[i0, j0]
                num += w_ij * array[j0]
                den += w_ij
            diff = array[i0] - (num / den if den > 1e-12 else 0.0)
            sro_near += diff * diff
    return sri + sro_near + sro_far

@njit(cache=True, fastmath=True)
def _laplace_accum_1d(w: np.ndarray, sigma: float) -> np.ndarray:
    """
    Stable O(n) accumulator for Laplace smoothing:
      f[i] = sum_j exp(-|i-j|/sigma) * w[j]

    - FIX: rightward recurrence off-by-one corrected
    - Optional edge seeding (pad_k>=1) to reduce boundary sag
    """
    n = w.shape[0]
    gamma = math.exp(-1.0 / sigma)

    Lx = np.zeros(n)
    L1 = np.zeros(n)
    for i in range(1, n):
        Lx[i] = gamma * (Lx[i-1] + w[i-1])
        L1[i] = gamma * (L1[i-1] + 1.0)

    Rx = np.zeros(n)
    R1 = np.zeros(n)
    for i in range(n-2, -1, -1):
        Rx[i] = gamma * (Rx[i+1] + w[i+1])
        R1[i] = gamma * (R1[i+1] + 1.0)

    out = (Lx + w + Rx) / np.maximum(L1 + 1.0 + R1, 1e-12)
    return out

@njit(cache=True, fastmath=True)
def calculate_laplace_sra_fast(array: np.ndarray, sigma: float):
    n = array.shape[0]

    x = array.astype(np.float64)

    num = _laplace_accum_1d(x, sigma)
    den = _laplace_accum_1d(np.ones_like(x), sigma)

    pred = np.empty(n, dtype=np.float64)
    ssr_arr = np.empty(n, dtype=np.float64)
    ssr = 0.0

    for i in range(n):
        d = den[i]
        p = num[i] / d if d > 1e-12 else 0.0
        pred[i] = p
        r = x[i] - p
        ssr_arr[i] = r * r
        ssr += ssr_arr[i]

    ssr_ps = np.empty(n + 1, dtype=np.float64)
    ssr_ps[0] = 0.0
    for i in range(n):
        ssr_ps[i + 1] = ssr_ps[i] + ssr_arr[i]

    return ssr, ssr_arr, pred, ssr_ps

@njit(cache=True, fastmath=True)
def calculate_laplace_sra_trunc(array: np.ndarray, sigma: float, r: int):
    n = array.shape[0]
    x = array.astype(np.float64)
    r = int(r)
    if r < 0:
        r = 0

    k = np.empty(r + 1, dtype=np.float64)
    inv = 1.0 / (sigma if sigma > 1e-12 else 1e-12)
    for d in range(r + 1):
        k[d] = np.exp(-d * inv)

    pred = np.empty(n, dtype=np.float64)
    ssr_arr = np.empty(n, dtype=np.float64)
    ssr = 0.0
    eps = 1e-12

    for i in range(n):
        j0 = i - r
        if j0 < 0:
            j0 = 0
        j1 = i + r
        if j1 > n - 1:
            j1 = n - 1

        num = 0.0
        den = 0.0
        for j in range(j0, j1 + 1):
            d = i - j
            if d < 0:
                d = -d
            wgt = k[d]
            den += wgt
            num += wgt * x[j]

        p = num / (den if den > eps else eps)
        pred[i] = p
        r_i = x[i] - p
        ssr_arr[i] = r_i * r_i
        ssr += ssr_arr[i]

    ssr_ps = np.empty(n + 1, dtype=np.float64)
    ssr_ps[0] = 0.0
    for i in range(n):
        ssr_ps[i + 1] = ssr_ps[i] + ssr_arr[i]

    return ssr, ssr_arr, pred, ssr_ps

@njit(cache=True, fastmath=True)
def _laplace_accum_trunc_1d(x: np.ndarray, sigma: float, r: int) -> np.ndarray:
    """
    Exact truncated Laplace accumulator:
        acc[i] = sum_{|i-j| <= r} gamma^{|i-j|} x[j]
    with gamma = exp(-1/sigma)

    Complexity: O(n)
    """
    n = x.shape[0]
    acc = np.empty(n, dtype=np.float64)

    if n == 0:
        return acc

    if r < 0:
        r = 0

    gamma = np.exp(-1.0 / (sigma if sigma > 1e-12 else 1e-12))
    gamma_r1 = gamma ** (r + 1)

    # left[i] = sum_{k=1}^{min(r,i)} gamma^k x[i-k]
    left = np.empty(n, dtype=np.float64)
    left[0] = 0.0
    for i in range(1, n):
        val = gamma * (x[i - 1] + left[i - 1])
        tail_idx = i - r - 1
        if tail_idx >= 0:
            val -= gamma_r1 * x[tail_idx]
        left[i] = val

    # right[i] = sum_{k=1}^{min(r,n-1-i)} gamma^k x[i+k]
    right = np.empty(n, dtype=np.float64)
    right[n - 1] = 0.0
    for i in range(n - 2, -1, -1):
        val = gamma * (x[i + 1] + right[i + 1])
        tail_idx = i + r + 1
        if tail_idx < n:
            val -= gamma_r1 * x[tail_idx]
        right[i] = val

    for i in range(n):
        acc[i] = left[i] + x[i] + right[i]

    return acc

@njit(cache=True, fastmath=True)
def calculate_laplace_sra_fast_trunc(array: np.ndarray, sigma: float, r: int):
    """
    Exact truncated Laplace NWKR SRA using the truncated Laplace recurrence trick.
    Complexity: O(n)
    """
    n = array.shape[0]
    x = array.astype(np.float64)

    num = _laplace_accum_trunc_1d(x, sigma, r)
    den = _laplace_accum_trunc_1d(np.ones_like(x), sigma, r)

    pred = np.empty(n, dtype=np.float64)
    ssr_arr = np.empty(n, dtype=np.float64)
    ssr = 0.0
    eps = 1e-12

    for i in range(n):
        d = den[i]
        p = num[i] / d if d > eps else 0.0
        pred[i] = p
        diff = x[i] - p
        ssr_arr[i] = diff * diff
        ssr += ssr_arr[i]

    ssr_ps = np.empty(n + 1, dtype=np.float64)
    ssr_ps[0] = 0.0
    for i in range(n):
        ssr_ps[i + 1] = ssr_ps[i] + ssr_arr[i]

    return ssr, ssr_arr, pred, ssr_ps

@njit(cache=True, fastmath=True)
def laplace_sri(array: np.ndarray, idxs: np.ndarray, a: int, b: int, sigma: float) -> float:
    preds = predict_on_idxs_laplace(array, idxs, sigma)
    sri = 0.0
    for t in range(idxs.shape[0]):
        i0 = int(idxs[t])
        if a <= i0 <= b:
            diff = array[i0] - preds[t]
            sri += diff * diff
    return sri

@njit(cache=True, fastmath=True)
def laplace_sro_nearband(
    x: np.ndarray,
    outside: np.ndarray,
    a: int,
    b: int,
    range_cap: int,
    ssr_ps: np.ndarray,
    sigma: float,
) -> float:
    n = x.shape[0]
    lo = max(0, a - (range_cap // 2))
    hi = min(n - 1, b + (range_cap // 2))

    sro_far = 0.0
    if lo > 0:
        sro_far += ssr_ps[lo] - ssr_ps[0]
    if hi < n - 1:
        sro_far += ssr_ps[n] - ssr_ps[hi + 1]

    near_out = np.empty(outside.shape[0], dtype=outside.dtype)
    k = 0
    for t in range(outside.shape[0]):
        i0 = outside[t]
        if lo <= i0 <= hi:
            near_out[k] = i0
            k += 1

    if k == 0:
        return sro_far

    near_slice = near_out[:k]
    preds_near = predict_on_idxs_laplace(x, near_slice, sigma)

    sro_near = 0.0
    for t in range(k):
        i0 = int(near_slice[t])
        diff = x[i0] - preds_near[t]
        sro_near += diff * diff

    return sro_far + sro_near

@njit(cache=True, fastmath=True)
def laplace_sri_trunc(
    array: np.ndarray,
    idxs: np.ndarray,
    a: int,
    b: int,
    k: np.ndarray,
) -> float:
    preds = predict_on_idxs_trunc(array, idxs, k)

    sri = 0.0
    for t in range(idxs.shape[0]):
        i0 = int(idxs[t])
        if a <= i0 <= b:
            diff = array[i0] - preds[t]
            sri += diff * diff

    return sri

@njit(cache=True, fastmath=True)
def laplace_sro_trunc(
    x: np.ndarray,
    outside: np.ndarray,
    k: np.ndarray,
) -> float:
    if outside.shape[0] == 0:
        return 0.0

    preds = predict_on_idxs_trunc(x, outside, k)

    sro = 0.0
    for t in range(outside.shape[0]):
        i0 = int(outside[t])
        diff = x[i0] - preds[t]
        sro += diff * diff

    return sro

@njit(cache=True)
def laplace_ssr_window(array, ssr_ps, idxs, a, b, range_cap, sigma):
    n_trim = idxs.shape[0]
    if n_trim == 0:
        return 0.0

    low_cut = a - (range_cap // 2)
    high_cut = b + (range_cap // 2)

    recompute_start = -1
    recompute_end = -1
    for k in range(n_trim):
        pos = idxs[k]
        if pos >= low_cut and pos <= high_cut:
            if recompute_start == -1:
                recompute_start = k
            recompute_end = k

    if recompute_start == -1:
        return ssr_ps[n_trim]

    m = recompute_end - recompute_start + 1
    pos_buf = np.empty(m, dtype=np.int64)
    arr_vals = np.empty(m, dtype=np.float64)
    for ii in range(m):
        idx = recompute_start + ii
        pos_buf[ii] = idxs[idx]
        arr_vals[ii] = array[idx]

    Lx = np.empty(m, dtype=np.float64)
    L1 = np.empty(m, dtype=np.float64)
    Lx[0] = 0.0
    L1[0] = 0.0
    for ii in range(1, m):
        dprev = np.exp(-float(pos_buf[ii] - pos_buf[ii - 1]) / sigma)
        Lx[ii] = dprev * (Lx[ii - 1] + arr_vals[ii - 1])
        L1[ii] = dprev * (L1[ii - 1] + 1.0)

    Rx = np.empty(m, dtype=np.float64)
    R1 = np.empty(m, dtype=np.float64)
    Rx[m - 1] = 0.0
    R1[m - 1] = 0.0
    for ii in range(m - 2, -1, -1):
        dnext = np.exp(-float(pos_buf[ii + 1] - pos_buf[ii]) / sigma)
        Rx[ii] = dnext * (Rx[ii + 1] + arr_vals[ii + 1])
        R1[ii] = dnext * (R1[ii + 1] + 1.0)

    ssr_recomp = 0.0
    for ii in range(m):
        num = Lx[ii] + arr_vals[ii] + Rx[ii]
        den = L1[ii] + 1.0 + R1[ii]
        pred = num / den
        diff = arr_vals[ii] - pred
        ssr_recomp += diff * diff

    ssr_far = ssr_ps[recompute_start] + (ssr_ps[n_trim] - ssr_ps[recompute_end + 1])

    return ssr_recomp + ssr_far

@njit(cache=True, fastmath=True)
def _ssr_region_laplace(array: np.ndarray, idxs: np.ndarray, ssr_array: np.ndarray,
                        a: int, b: int, range_cap: int, sigma: float) -> float:
    m = idxs.shape[0]
    if m == 0: return 0.0
    preds = predict_on_idxs_laplace(array, idxs, sigma)
    sri = 0.0; sro_far = 0.0; sro_near = 0.0
    low_cut = a - range_cap // 2; high_cut = b + range_cap // 2
    for ii in range(m):
        i0 = idxs[ii]
        if a <= i0 <= b:
            diff = array[i0] - preds[ii]; sri += diff * diff
        elif i0 < low_cut or i0 > high_cut:
            sro_far += ssr_array[i0]
        else:
            diff = array[i0] - preds[ii]; sro_near += diff * diff
    return sri + sro_near + sro_far

def ssr_region_dispatch(array: np.ndarray, idxs: np.ndarray, W: np.ndarray, ssr_array: np.ndarray, a: int, b: int, range_cap: int, kind: str, sigma: float | None = None) -> float:
    if kind == KernelKind.LAPLACE:
        if sigma is None or sigma <= 0.0:
            raise ValueError("Laplace predictor requires positive sigma.")
        return _ssr_region_laplace(array.astype(np.float64), idxs.astype(np.int64),
                                   ssr_array.astype(np.float64), a, b, range_cap, float(sigma))
    elif kind == KernelKind.GAUSSIAN:
        if W is None:
            raise ValueError("Dense predictor requires W (got None).")
        return _ssr_region_gaussian(array, idxs.astype(np.int64), W, ssr_array, a, b, range_cap)
    else:
        raise ValueError(f"Unknown kernel kind: {kind!r}")
 
def score_variance_nwkr(array: np.ndarray, inside: np.ndarray, outside: np.ndarray, a: int, b: int, range_cap: int, W: np.ndarray, ssr_array: np.ndarray, kind: str, sigma: float | None = None) -> float:
    if kind == KernelKind.LAPLACE:
        if sigma is None or sigma <= 0:
            raise ValueError(...)
    elif kind == KernelKind.GAUSSIAN:
        sigma = 0.0
    sri = ssr_region_dispatch(array, inside,  W, ssr_array, a, b, range_cap, kind, sigma)
    sro = ssr_region_dispatch(array, outside, W, ssr_array, a, b, range_cap, kind, sigma)
    return -(sri + sro)
