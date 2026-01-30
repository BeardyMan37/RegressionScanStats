# from __future__ import annotations
# import numpy as np

# class GaussianInsideState:
#     __slots__ = ("idxs", "num", "den", "sse")
#     def __init__(self, idxs, num, den, sse):
#         self.idxs = idxs
#         self.num  = num
#         self.den  = den
#         self.sse  = float(sse)

# def gaussian_inside_init(x: np.ndarray, idxs: np.ndarray, W: np.ndarray) -> GaussianInsideState:
#     m = idxs.shape[0]
#     num = np.zeros(m, dtype=np.float64)
#     den = np.zeros(m, dtype=np.float64)
#     for k in range(m):
#         i = idxs[k]
#         num[k] = (W[i, idxs] * x[idxs]).sum()
#         den[k] = W[i, idxs].sum()
#     preds = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
#     sse = float(((x[idxs] - preds) ** 2).sum())
#     return GaussianInsideState(idxs.copy(), num, den, sse)

# def gaussian_inside_add_point(x: np.ndarray, state: GaussianInsideState, new_idx: int, W: np.ndarray) -> GaussianInsideState:
#     old_idxs = state.idxs
#     m_prev = old_idxs.shape[0]
#     idxs_new = np.empty(m_prev + 1, dtype=np.int64)
#     idxs_new[:m_prev] = old_idxs
#     idxs_new[m_prev]  = new_idx
#     x_prev = x[old_idxs]; x_new = x[new_idx]
#     w_existing = W[old_idxs, new_idx]
#     num_new = np.empty(m_prev + 1, dtype=np.float64)
#     den_new = np.empty(m_prev + 1, dtype=np.float64)
#     num_new[:m_prev] = state.num + w_existing * x_new
#     den_new[:m_prev] = state.den + w_existing
#     w_new_all = W[new_idx, idxs_new]
#     num_new[m_prev] = (w_new_all[:m_prev] * x_prev).sum() + w_new_all[m_prev] * x_new
#     den_new[m_prev] = w_new_all.sum()
#     preds_prev = np.divide(num_new[:m_prev], den_new[:m_prev], out=np.zeros_like(num_new[:m_prev]), where=den_new[:m_prev] > 1e-12)
#     pred_new   = num_new[m_prev] / den_new[m_prev] if den_new[m_prev] > 1e-12 else 0.0
#     residuals_prev = x_prev - preds_prev
#     sse_new = float((residuals_prev ** 2).sum() + (x_new - pred_new) ** 2)
#     return GaussianInsideState(idxs_new, num_new, den_new, sse_new)

# def gaussian_inside_remove_point(x: np.ndarray, state: GaussianInsideState, rem_idx: int, W: np.ndarray) -> GaussianInsideState:
#     idxs_old = state.idxs
#     m_prev = idxs_old.shape[0]
#     pos = -1
#     for t in range(m_prev):
#         if idxs_old[t] == rem_idx:
#             pos = t; break
#     if pos == -1:
#         return state
#     m_new = m_prev - 1
#     idxs_new = np.empty(m_new, dtype=np.int64)
#     num_new  = np.empty(m_new, dtype=np.float64)
#     den_new  = np.empty(m_new, dtype=np.float64)
#     cur = 0
#     for t in range(m_prev):
#         if t == pos: continue
#         i = idxs_old[t]
#         idxs_new[cur] = i
#         w_ir = W[i, rem_idx]
#         num_new[cur] = state.num[t] - w_ir * x[rem_idx]
#         den_new[cur] = state.den[t] - w_ir
#         cur += 1
#     sse_new = 0.0
#     for u in range(m_new):
#         i = idxs_new[u]
#         d = den_new[u]
#         pred = num_new[u] / d if d > 1e-12 else 0.0
#         diff = x[i] - pred
#         sse_new += diff * diff
#     return GaussianInsideState(idxs_new, num_new, den_new, sse_new)

# class GaussianOutsideState:
#     __slots__ = ("idxs", "delta_num", "delta_den", "sse")

#     def __init__(self, idxs, delta_num, delta_den, sse):
#         self.idxs      = idxs
#         self.delta_num = delta_num
#         self.delta_den = delta_den
#         self.sse       = float(sse)

# def gaussian_outside_init(
#     x: np.ndarray,
#     inside_idxs: np.ndarray,
#     near_idxs: np.ndarray,
#     W: np.ndarray,
#     numer_all: np.ndarray,
#     denom_all: np.ndarray,
# ) -> GaussianOutsideState:
#     m_near = near_idxs.shape[0]
#     delta_num = np.zeros(m_near, dtype=np.float64)
#     delta_den = np.zeros(m_near, dtype=np.float64)

#     for j in inside_idxs:
#         w_vec = W[near_idxs, j]
#         delta_num += w_vec * x[j]
#         delta_den += w_vec

#     eps = 1e-12
#     sse_near = 0.0
#     for t in range(m_near):
#         i = near_idxs[t]
#         N_out = numer_all[i] - delta_num[t]
#         D_out = denom_all[i] - delta_den[t]
#         if D_out > eps:
#             pred = N_out / D_out
#         else:
#             pred = 0.0
#         diff = x[i] - pred
#         sse_near += diff * diff

#     return GaussianOutsideState(near_idxs.copy(), delta_num, delta_den, sse_near)

# def gaussian_outside_add_inside(
#     x: np.ndarray,
#     state: GaussianOutsideState,
#     new_idx: int,
#     W: np.ndarray,
#     numer_all: np.ndarray,
#     denom_all: np.ndarray,
# ) -> GaussianOutsideState:
#     idxs_near  = state.idxs
#     delta_num0 = state.delta_num
#     delta_den0 = state.delta_den
#     m_near = idxs_near.shape[0]

#     w_vec = W[idxs_near, new_idx]

#     delta_num = delta_num0 + w_vec * x[new_idx]
#     delta_den = delta_den0 + w_vec

#     eps = 1e-12
#     sse_near = 0.0
#     for t in range(m_near):
#         i = idxs_near[t]
#         N_out = numer_all[i] - delta_num[t]
#         D_out = denom_all[i] - delta_den[t]
#         if D_out > eps:
#             pred = N_out / D_out
#         else:
#             pred = 0.0
#         diff = x[i] - pred
#         sse_near += diff * diff

#     return GaussianOutsideState(idxs_near, delta_num, delta_den, sse_near)

# def gaussian_outside_remove_inside(
#     x: np.ndarray,
#     state: GaussianOutsideState,
#     rem_idx: int,
#     W: np.ndarray,
#     numer_all: np.ndarray,
#     denom_all: np.ndarray,
# ) -> GaussianOutsideState:
#     idxs_near  = state.idxs
#     delta_num0 = state.delta_num
#     delta_den0 = state.delta_den
#     m_near = idxs_near.shape[0]

#     w_vec = W[idxs_near, rem_idx]        # shape (m_near,)

#     delta_num = delta_num0 - w_vec * x[rem_idx]
#     delta_den = delta_den0 - w_vec

#     eps = 1e-12
#     sse_near = 0.0
#     for t in range(m_near):
#         i = idxs_near[t]
#         N_out = numer_all[i] - delta_num[t]
#         D_out = denom_all[i] - delta_den[t]
#         if D_out > eps:
#             pred = N_out / D_out
#         else:
#             pred = 0.0
#         diff = x[i] - pred
#         sse_near += diff * diff

#     return GaussianOutsideState(idxs_near, delta_num, delta_den, sse_near)

# def gaussian_sro_naive(row_trimmed: np.ndarray,
#                        inside: np.ndarray,
#                        W_trimmed: np.ndarray) -> float:
#     """
#     Slow but exact SRO: fit Gaussian NWKR on the complement of `inside`
#     and return its SSE. Uses gaussian_inside_init on the outside set.
#     """
#     n_trim = row_trimmed.shape[0]
#     all_trimmed = np.arange(n_trim, dtype=np.int64)
#     outside = np.setdiff1d(all_trimmed, inside, assume_unique=True)
#     if outside.size == 0:
#         return 0.0
#     gstate_out_exact = gaussian_inside_init(row_trimmed, outside, W_trimmed)
#     return gstate_out_exact.sse

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def gaussian_state_init(x: np.ndarray,
                        idxs: np.ndarray,
                        W: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    m = idxs.shape[0]
    num = np.zeros(m, dtype=np.float64)
    den = np.zeros(m, dtype=np.float64)

    # Compute num[k], den[k] for each index i = idxs[k]
    for k in range(m):
        i = idxs[k]
        s_num = 0.0
        s_den = 0.0
        for t in range(m):
            j = idxs[t]
            w = W[i, j]
            s_num += w * x[j]
            s_den += w
        num[k] = s_num
        den[k] = s_den

    # preds = num / den with safe guard
    preds = np.empty(m, dtype=np.float64)
    for k in range(m):
        if den[k] > 1e-12:
            preds[k] = num[k] / den[k]
        else:
            preds[k] = 0.0

    # SSE over the indices in idxs
    sse = 0.0
    for k in range(m):
        i = idxs[k]
        diff = x[i] - preds[k]
        sse += diff * diff

    return idxs.copy(), num, den, float(sse)

@njit(cache=True, fastmath=True)
def gaussian_state_add_point(x: np.ndarray,
                             idxs: np.ndarray,
                             num: np.ndarray,
                             den: np.ndarray,
                             sse: float,
                             new_idx: int,
                             W: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    m_prev = idxs.shape[0]
    m_new = m_prev + 1

    idxs_new = np.empty(m_new, dtype=np.int64)
    idxs_new[:m_prev] = idxs
    idxs_new[m_prev] = new_idx

    x_prev = np.empty(m_prev, dtype=np.float64)
    for k in range(m_prev):
        x_prev[k] = x[idxs[k]]
    x_new = x[new_idx]

    num_new = np.empty(m_new, dtype=np.float64)
    den_new = np.empty(m_new, dtype=np.float64)

    # Update existing points' num/den
    for k in range(m_prev):
        i = idxs[k]
        w = W[i, new_idx]
        num_new[k] = num[k] + w * x_new
        den_new[k] = den[k] + w

    # New point's num/den
    s_num = 0.0
    s_den = 0.0
    for k in range(m_prev):
        j = idxs[k]
        w = W[new_idx, j]
        s_num += w * x_prev[k]
        s_den += w
    # self-weight (optional; keep if consistent with your W definition)
    w_self = W[new_idx, new_idx]
    s_num += w_self * x_new
    s_den += w_self

    num_new[m_prev] = s_num
    den_new[m_prev] = s_den

    # Recompute SSE from updated num/den
    preds_prev = np.empty(m_prev, dtype=np.float64)
    for k in range(m_prev):
        if den_new[k] > 1e-12:
            preds_prev[k] = num_new[k] / den_new[k]
        else:
            preds_prev[k] = 0.0

    if den_new[m_prev] > 1e-12:
        pred_new = num_new[m_prev] / den_new[m_prev]
    else:
        pred_new = 0.0

    sse_new = 0.0
    for k in range(m_prev):
        diff = x_prev[k] - preds_prev[k]
        sse_new += diff * diff
    diff_new = x_new - pred_new
    sse_new += diff_new * diff_new

    return idxs_new, num_new, den_new, float(sse_new)

@njit(cache=True, fastmath=True)
def gaussian_state_remove_point(x: np.ndarray,
                                idxs: np.ndarray,
                                num: np.ndarray,
                                den: np.ndarray,
                                sse: float,
                                rem_idx: int,
                                W: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    m_prev = idxs.shape[0]
    pos = -1
    for t in range(m_prev):
        if idxs[t] == rem_idx:
            pos = t
            break

    if pos == -1:
        # rem_idx not in window; return unchanged
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
        w_ir = W[i, rem_idx]
        num_new[cur] = num[t] - w_ir * x[rem_idx]
        den_new[cur] = den[t] - w_ir
        cur += 1

    # Recompute SSE from updated num/den
    sse_new = 0.0
    for u in range(m_new):
        i = idxs_new[u]
        d = den_new[u]
        if d > 1e-12:
            pred = num_new[u] / d
        else:
            pred = 0.0
        diff = x[i] - pred
        sse_new += diff * diff

    return idxs_new, num_new, den_new, float(sse_new)

