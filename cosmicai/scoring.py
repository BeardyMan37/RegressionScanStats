from __future__ import annotations
import numpy as np
from numba import njit
from .config import KernelKind
from typing import Optional
from .predictors import predict_on_idxs_laplace

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
