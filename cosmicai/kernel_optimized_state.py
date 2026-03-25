from __future__ import annotations
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def kernel_state_init_trunc(
    x: np.ndarray,
    idxs: np.ndarray,
    k: np.ndarray,   # kernel vector, length r+1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Initialize NWKR state on subset idxs using a truncated Gaussian kernel.

    Assumptions
    -----------
    - idxs is sorted increasing.
    - k[d] is the kernel weight for distance d, for d = 0..r.
    - weights are zero for |i-j| > r.

    Returns
    -------
    idxs_copy, num, den, sse
    """
    m = idxs.shape[0]
    r = k.shape[0] - 1
    eps = 1e-12

    idxs_out = np.empty(m, dtype=np.int64)
    idxs_out[:] = idxs

    num = np.zeros(m, dtype=np.float64)
    den = np.zeros(m, dtype=np.float64)

    # For each center u, only scan nearby subset entries whose index-distance <= r.
    for u in range(m):
        i = idxs[u]

        s_num = 0.0
        s_den = 0.0

        # scan left from u until distance > r
        v = u
        while v >= 0:
            d = i - idxs[v]   # idxs sorted, so this is nonnegative
            if d > r:
                break
            w = k[d]
            s_num += w * x[idxs[v]]
            s_den += w
            v -= 1

        # scan right from u+1 until distance > r
        v = u + 1
        while v < m:
            d = idxs[v] - i
            if d > r:
                break
            w = k[d]
            s_num += w * x[idxs[v]]
            s_den += w
            v += 1

        num[u] = s_num
        den[u] = s_den

    sse = 0.0
    for u in range(m):
        d = den[u]
        pred = num[u] / d if d > eps else 0.0
        diff = x[idxs[u]] - pred
        sse += diff * diff

    return idxs_out, num, den, float(sse)

@njit(cache=True, fastmath=True)
def kernel_state_add_point_trunc(
    x: np.ndarray,
    idxs: np.ndarray,
    num: np.ndarray,
    den: np.ndarray,
    sse: float,
    new_idx: int,
    k: np.ndarray   # kernel vector (length r+1)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    r = k.shape[0] - 1

    m_prev = idxs.shape[0]
    m_new = m_prev + 1

    idxs_new = np.empty(m_new, dtype=np.int64)
    idxs_new[:m_prev] = idxs
    idxs_new[m_prev] = new_idx

    x_prev = np.empty(m_prev, dtype=np.float64)
    for t in range(m_prev):
        x_prev[t] = x[idxs[t]]
    x_new = x[new_idx]

    num_new = np.empty(m_new, dtype=np.float64)
    den_new = np.empty(m_new, dtype=np.float64)

    # ---- update existing points ----
    for t in range(m_prev):
        i = idxs[t]
        d = i - new_idx
        if d < 0:
            d = -d

        if d <= r:
            w = k[d]
        else:
            w = 0.0

        num_new[t] = num[t] + w * x_new
        den_new[t] = den[t] + w

    # ---- new point ----
    s_num = 0.0
    s_den = 0.0

    for t in range(m_prev):
        j = idxs[t]
        d = new_idx - j
        if d < 0:
            d = -d

        if d <= r:
            w = k[d]
            s_num += w * x_prev[t]
            s_den += w

    # self-weight
    w_self = k[0]
    s_num += w_self * x_new
    s_den += w_self

    num_new[m_prev] = s_num
    den_new[m_prev] = s_den

    # ---- recompute SSE ----
    sse_new = 0.0

    for t in range(m_prev):
        d = den_new[t]
        pred = num_new[t] / d if d > 1e-12 else 0.0
        diff = x_prev[t] - pred
        sse_new += diff * diff

    pred_new = num_new[m_prev] / den_new[m_prev] if den_new[m_prev] > 1e-12 else 0.0
    diff_new = x_new - pred_new
    sse_new += diff_new * diff_new

    return idxs_new, num_new, den_new, float(sse_new)

@njit(cache=True, fastmath=True)
def kernel_state_remove_point_trunc(
    x: np.ndarray,
    idxs: np.ndarray,
    num: np.ndarray,
    den: np.ndarray,
    sse: float,
    rem_idx: int,
    k: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    r = k.shape[0] - 1

    m_prev = idxs.shape[0]

    pos = -1
    for t in range(m_prev):
        if idxs[t] == rem_idx:
            pos = t
            break

    if pos == -1:
        return idxs, num, den, sse

    m_new = m_prev - 1

    idxs_new = np.empty(m_new, dtype=np.int64)
    num_new = np.empty(m_new, dtype=np.float64)
    den_new = np.empty(m_new, dtype=np.float64)

    cur = 0
    for t in range(m_prev):
        if t == pos:
            continue

        i = idxs[t]
        idxs_new[cur] = i

        d = i - rem_idx
        if d < 0:
            d = -d

        if d <= r:
            w = k[d]
        else:
            w = 0.0

        num_new[cur] = num[t] - w * x[rem_idx]
        den_new[cur] = den[t] - w

        cur += 1

    # ---- recompute SSE ----
    sse_new = 0.0
    for t in range(m_new):
        i = idxs_new[t]
        d = den_new[t]
        pred = num_new[t] / d if d > 1e-12 else 0.0
        diff = x[i] - pred
        sse_new += diff * diff

    return idxs_new, num_new, den_new, float(sse_new)