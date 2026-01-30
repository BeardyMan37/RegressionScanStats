from __future__ import annotations
import math, numpy as np
from numba import njit

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

