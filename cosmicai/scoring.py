from __future__ import annotations
import numpy as np
from numba import njit
from .laplace_fast import calculate_laplace_sra_fast
from .predictors import predict_on_idxs_denseW, predict_on_idxs_laplace

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
def _ssr_region_dense(array: np.ndarray, idxs: np.ndarray, W: np.ndarray,
                      ssr_array: np.ndarray, a: int, b: int, range_cap: int) -> float:
    m = idxs.shape[0]
    sri = 0.0; sro_far = 0.0; sro_near = 0.0
    low_cut = a - 2 * range_cap; high_cut = b + 2 * range_cap
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
def _ssr_region_laplace(array: np.ndarray, idxs: np.ndarray, ssr_array: np.ndarray,
                        a: int, b: int, range_cap: int, sigma: float) -> float:
    m = idxs.shape[0]
    if m == 0: return 0.0
    preds = predict_on_idxs_laplace(array, idxs, sigma)
    sri = 0.0; sro_far = 0.0; sro_near = 0.0
    low_cut = a - 2 * range_cap; high_cut = b + 2 * range_cap
    for ii in range(m):
        i0 = idxs[ii]
        if a <= i0 <= b:
            diff = array[i0] - preds[ii]; sri += diff * diff
        elif i0 < low_cut or i0 > high_cut:
            sro_far += ssr_array[i0]
        else:
            diff = array[i0] - preds[ii]; sro_near += diff * diff
    return sri + sro_near + sro_far

def ssr_region_dispatch(array, idxs, W, ssr_array, a, b, range_cap, kind: str, sigma: float | None = None) -> float:
    if kind == "laplace":
        if sigma is None or sigma <= 0.0:
            raise ValueError("Laplace predictor requires positive sigma.")
        return _ssr_region_laplace(array.astype(np.float64), idxs.astype(np.int64),
                                   ssr_array.astype(np.float64), a, b, range_cap, float(sigma))
    elif kind == "gaussian":
        if W is None:
            raise ValueError("Dense predictor requires W (got None).")
        return _ssr_region_dense(array, idxs.astype(np.int64), W, ssr_array, a, b, range_cap)
 
def score_variance_nwkr(array, inside, outside, a, b, range_cap, W, ssr_array, kind: str) -> float:
    sigma = float(max(range_cap // 3, 1))
    sri = ssr_region_dispatch(array, inside,  W, ssr_array, a, b, range_cap, kind, sigma)
    sro = ssr_region_dispatch(array, outside, W, ssr_array, a, b, range_cap, kind, sigma)
    return -(sri + sro)
