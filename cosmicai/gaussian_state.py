from __future__ import annotations
import numpy as np

class GaussianWindowState:
    __slots__ = ("idxs", "num", "den", "sse")
    def __init__(self, idxs, num, den, sse):
        self.idxs = idxs
        self.num  = num
        self.den  = den
        self.sse  = float(sse)

def gaussian_state_init(x: np.ndarray, idxs: np.ndarray, W: np.ndarray) -> GaussianWindowState:
    m = idxs.shape[0]
    num = np.zeros(m, dtype=np.float64)
    den = np.zeros(m, dtype=np.float64)
    for k in range(m):
        i = idxs[k]
        num[k] = (W[i, idxs] * x[idxs]).sum()
        den[k] = W[i, idxs].sum()
    preds = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    sse = float(((x[idxs] - preds) ** 2).sum())
    return GaussianWindowState(idxs.copy(), num, den, sse)

def gaussian_state_add_point(x: np.ndarray, state: GaussianWindowState, new_idx: int, W: np.ndarray) -> GaussianWindowState:
    old_idxs = state.idxs
    m_prev = old_idxs.shape[0]
    idxs_new = np.empty(m_prev + 1, dtype=np.int64)
    idxs_new[:m_prev] = old_idxs
    idxs_new[m_prev]  = new_idx
    x_prev = x[old_idxs]; x_new = x[new_idx]
    w_existing = W[old_idxs, new_idx]
    num_new = np.empty(m_prev + 1, dtype=np.float64)
    den_new = np.empty(m_prev + 1, dtype=np.float64)
    num_new[:m_prev] = state.num + w_existing * x_new
    den_new[:m_prev] = state.den + w_existing
    w_new_all = W[new_idx, idxs_new]
    num_new[m_prev] = (w_new_all[:m_prev] * x_prev).sum() + w_new_all[m_prev] * x_new
    den_new[m_prev] = w_new_all.sum()
    preds_prev = np.divide(num_new[:m_prev], den_new[:m_prev], out=np.zeros_like(num_new[:m_prev]), where=den_new[:m_prev] > 1e-12)
    pred_new   = num_new[m_prev] / den_new[m_prev] if den_new[m_prev] > 1e-12 else 0.0
    residuals_prev = x_prev - preds_prev
    sse_new = float((residuals_prev ** 2).sum() + (x_new - pred_new) ** 2)
    return GaussianWindowState(idxs_new, num_new, den_new, sse_new)

def gaussian_state_remove_point(x: np.ndarray, state: GaussianWindowState, rem_idx: int, W: np.ndarray) -> GaussianWindowState:
    idxs_old = state.idxs
    m_prev = idxs_old.shape[0]
    pos = -1
    for t in range(m_prev):
        if idxs_old[t] == rem_idx:
            pos = t; break
    if pos == -1:
        return state
    m_new = m_prev - 1
    idxs_new = np.empty(m_new, dtype=np.int64)
    num_new  = np.empty(m_new, dtype=np.float64)
    den_new  = np.empty(m_new, dtype=np.float64)
    cur = 0
    for t in range(m_prev):
        if t == pos: continue
        i = idxs_old[t]
        idxs_new[cur] = i
        w_ir = W[i, rem_idx]
        num_new[cur] = state.num[t] - w_ir * x[rem_idx]
        den_new[cur] = state.den[t] - w_ir
        cur += 1
    sse_new = 0.0
    for u in range(m_new):
        i = idxs_new[u]
        d = den_new[u]
        pred = num_new[u] / d if d > 1e-12 else 0.0
        diff = x[i] - pred
        sse_new += diff * diff
    return GaussianWindowState(idxs_new, num_new, den_new, sse_new)
