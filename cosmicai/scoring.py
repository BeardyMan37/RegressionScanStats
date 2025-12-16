from __future__ import annotations
import numpy as np
from numba import njit
from .config import KernelKind
from .predictors import predict_on_idxs_laplace

@njit(cache=True, fastmath=True)
def calculate_nwkr_sra(array: np.ndarray, W: np.ndarray):
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
def calculate_nwkr_sra_with_nd(array: np.ndarray, W: np.ndarray):
    n = array.shape[0]
    numer = np.empty(n, dtype=array.dtype)
    denom = np.empty(n, dtype=array.dtype)

    for i in range(n):
        s_num = 0.0
        s_den = 0.0
        for j in range(n):
            w_ij = W[i, j]
            s_num += w_ij * array[j]
            s_den += w_ij
        numer[i] = s_num
        denom[i] = s_den

    ssr = 0.0
    ssr_array = np.empty(n, dtype=array.dtype)
    pred_array = np.empty(n, dtype=array.dtype)

    for i in range(n):
        d = denom[i]
        pred = numer[i] / d if d > 0.0 else 0.0
        pred_array[i] = pred
        diff = array[i] - pred
        ssr_array[i] = diff * diff
        ssr += ssr_array[i]

    ssr_ps = np.empty(n + 1, dtype=array.dtype)
    ssr_ps[0] = 0.0
    for k in range(n):
        ssr_ps[k + 1] = ssr_ps[k] + ssr_array[k]

    return ssr, ssr_array, pred_array, numer, denom, ssr_ps

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

    cnt = 0
    for t in range(outside.shape[0]):
        i0 = outside[t]
        if lo <= i0 <= hi:
            cnt += 1

    near_out = np.empty(cnt, dtype=outside.dtype)
    k = 0
    for t in range(outside.shape[0]):
        i0 = outside[t]
        if lo <= i0 <= hi:
            near_out[k] = i0
            k += 1

    preds_near = predict_on_idxs_laplace(x, near_out, sigma)

    sro_near = 0.0
    for k in range(near_out.shape[0]):
        i0 = int(near_out[k])
        diff = x[i0] - preds_near[k]
        sro_near += diff * diff

    return sro_far + sro_near

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
