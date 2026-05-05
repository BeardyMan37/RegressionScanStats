from __future__ import annotations
import numpy as np
import math
from numba import njit
from .config import KernelKind

@njit(cache=True, fastmath=True)
def predict_on_idxs_gaussian(array: np.ndarray, idxs: np.ndarray, W: np.ndarray) -> np.ndarray:
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
    out = np.empty(m, dtype=array.dtype)
    if m == 0:
        return out
    if m == 1:
        out[0] = array[idxs[0]]
        return out
    order = np.argsort(idxs)
    rev = np.empty_like(order)
    for k in range(m):
        rev[order[k]] = k
    sidx = idxs[order]
    inv_sigma = 1.0 / sigma if sigma > 0.0 else 0.0

    decay_lt = np.empty(m, dtype=np.float64)
    decay_rt = np.empty(m, dtype=np.float64)
    decay_lt[0] = 0.0
    for t in range(1, m):
        gap = sidx[t] - sidx[t-1]
        decay_lt[t] = math.exp(-gap * inv_sigma)
    for t in range(0, m-1):
        gap = sidx[t+1] - sidx[t]
        decay_rt[t] = math.exp(-gap * inv_sigma)
    decay_rt[m-1] = 0.0
    pad_k = 2
    gamma = math.exp(-inv_sigma) if sigma > 0.0 else 0.0
    one_minus_gamma = -math.expm1(-inv_sigma) if sigma > 0.0 else 1.0
    geom_tail = (-math.expm1(-pad_k * inv_sigma) / one_minus_gamma) if sigma > 0.0 else 0.0
    seed_coef = gamma * geom_tail if sigma > 0.0 else 0.0
    Lx = np.zeros(m, dtype=np.float64)
    L1 = np.zeros(m, dtype=np.float64)
    Lx[0] = seed_coef * float(array[sidx[0]])
    L1[0] = seed_coef
    for t in range(1, m):
        d = decay_lt[t]
        Lx[t] = d * (Lx[t-1] + float(array[sidx[t-1]]))
        L1[t] = d * (L1[t-1] + 1.0)
    Rx = np.zeros(m, dtype=np.float64)
    R1 = np.zeros(m, dtype=np.float64)
    Rx[m-1] = seed_coef * float(array[sidx[m-1]])
    R1[m-1] = seed_coef
    for t in range(m-2, -1, -1):
        d = decay_rt[t]
        Rx[t] = d * (Rx[t+1] + float(array[sidx[t+1]]))
        R1[t] = d * (R1[t+1] + 1.0)
    eps = 1e-12
    preds_sorted = np.empty(m, dtype=np.float64)
    for t in range(m):
        num = Lx[t] + float(array[sidx[t]]) + Rx[t]
        den = L1[t] + 1.0 + R1[t]
        preds_sorted[t] = num / (den if den > eps else eps)
    for k in range(m):
        out[k] = preds_sorted[rev[k]]
    return out

def predict_on_idxs(array, idxs, W, kind_str, sigma):
    kind = KernelKind(kind_str)
    if kind == KernelKind.LAPLACE and sigma is not None and sigma > 0.0:
        return predict_on_idxs_laplace(array.astype(np.float64),
                                       idxs.astype(np.int64),
                                       float(sigma))
    if W is None:
        raise ValueError("Dense predictor requires W (got None).")
    return predict_on_idxs_gaussian(array, idxs.astype(np.int64), W)

@njit(cache=True, fastmath=True)
def predict_on_idxs_trunc(array: np.ndarray, idxs: np.ndarray, k: np.ndarray) -> np.ndarray:
    m = idxs.shape[0]
    r = k.shape[0] - 1
    out = np.empty(m, dtype=np.float64)

    for ii in range(m):
        i0 = idxs[ii]
        num = 0.0
        den = 0.0

        jj = ii
        while jj >= 0:
            d = i0 - idxs[jj]
            if d > r:
                break
            w = k[d]
            num += w * array[idxs[jj]]
            den += w
            jj -= 1

        jj = ii + 1
        while jj < m:
            d = idxs[jj] - i0
            if d > r:
                break
            w = k[d]
            num += w * array[idxs[jj]]
            den += w
            jj += 1

        out[ii] = num / den if den > 1e-12 else 0.0

    return out