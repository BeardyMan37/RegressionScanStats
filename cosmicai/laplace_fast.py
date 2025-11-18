from __future__ import annotations
import math, numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _laplace_accum_1d(w: np.ndarray, sigma: float) -> np.ndarray:
    n = w.shape[0]
    out = np.empty(n, dtype=w.dtype)
    gamma = math.exp(-1.0 / sigma) if sigma > 0.0 else 0.0
    inv_n = 1.0 / n
    R = 0.0
    gpow = gamma
    for j in range(1, n):
        R += gpow * w[j]
        gpow *= gamma
    L = 0.0
    out[0] = inv_n * (w[0] + R)
    for i in range(1, n):
        L = gamma * (L + w[i - 1])
        R = (R - gamma * w[i]) / gamma
        out[i] = inv_n * (L + R + w[i])
    return out

@njit(cache=True, fastmath=True)
def calculate_laplace_sra_fast(array: np.ndarray, sigma: float):
    n = array.shape[0]
    num = _laplace_accum_1d(array, sigma)
    den = _laplace_accum_1d(np.ones_like(array), sigma)
    pred = np.empty(n, dtype=array.dtype)
    ssr_arr = np.empty(n, dtype=array.dtype)
    ssr = 0.0
    for i in range(n):
        d = den[i]
        p = num[i] / d if d > 1e-12 else 0.0
        pred[i] = p
        r = array[i] - p
        ssr_arr[i] = r * r
        ssr += ssr_arr[i]
    return ssr, ssr_arr, pred

