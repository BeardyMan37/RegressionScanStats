from __future__ import annotations
import numpy as np
import math
from numba import njit
from .config import KernelKind

@njit(cache=True, fastmath=True)
def predict_on_idxs_denseW(array: np.ndarray, idxs: np.ndarray, W: np.ndarray) -> np.ndarray:
    m = idxs.shape[0]
    out = np.empty(m, dtype=array.dtype)
    for ii in range(m):
        i0 = idxs[ii]
        num = 0.0; den = 0.0
        for jj in range(m):
            j0 = idxs[jj]
            w_ij = W[i0, j0]
            num += w_ij * array[j0]
            den += w_ij
        out[ii] = num / den if den > 1e-12 else 0.0
    return out

@njit(cache=True, fastmath=True)
def predict_on_idxs_laplace(array: np.ndarray, idxs: np.ndarray, sigma: float) -> np.ndarray:
    m = idxs.shape[0]
    out = np.empty(m, dtype=np.float64)
    if m == 0: return out
    if m == 1:
        out[0] = array[idxs[0]]
        return out
    order = np.argsort(idxs)
    rev = np.empty_like(order)
    for k in range(m): rev[order[k]] = k
    sidx = idxs[order]
    inv_sigma = 1.0 / sigma if sigma > 0.0 else 0.0
    decay_lt = np.empty(m); decay_rt = np.empty(m)
    decay_lt[0] = 0.0; decay_rt[m-1] = 0.0
    for t in range(1, m):     decay_lt[t] = math.exp(-(sidx[t] - sidx[t-1]) * inv_sigma)
    for t in range(0, m - 1): decay_rt[t] = math.exp(-(sidx[t+1] - sidx[t]) * inv_sigma)
    left_num = np.zeros(m); left_den = np.zeros(m)
    for t in range(1, m):
        d = decay_lt[t]
        left_num[t] = d * (left_num[t-1] + array[sidx[t-1]])
        left_den[t] = d * (left_den[t-1] + 1.0)
    right_num = np.zeros(m); right_den = np.zeros(m)
    for t in range(m-2, -1, -1):
        d = decay_rt[t]
        right_num[t] = d * (right_num[t+1] + array[sidx[t+1]])
        right_den[t] = d * (right_den[t+1] + 1.0)
    preds_sorted = np.empty(m)
    for t in range(m):
        num = left_num[t] + right_num[t] + array[sidx[t]]
        den = left_den[t] + right_den[t] + 1.0
        preds_sorted[t] = num / den if den > 1e-12 else 0.0
    for k in range(m): out[k] = preds_sorted[rev[k]]
    return out

def predict_on_idxs(array, idxs, W, kind_str, sigma):
    kind = KernelKind(kind_str)
    if kind == KernelKind.LAPLACE and sigma is not None and sigma > 0.0:
        return predict_on_idxs_laplace(array.astype(np.float64),
                                       idxs.astype(np.int64),
                                       float(sigma))
    if W is None:
        raise ValueError("Dense predictor requires W (got None).")
    return predict_on_idxs_denseW(array, idxs.astype(np.int64), W)