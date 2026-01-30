from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

def _prefix_sum(arr: np.ndarray) -> np.ndarray:
    ps = np.empty(arr.size + 1, dtype=arr.dtype)
    ps[0] = 0.0
    ps[1:] = np.cumsum(arr)
    return ps

# -------------------------
# Mean / constant
# -------------------------
def calculate_mean_sra(array: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    x = array.astype(np.float64)
    mu = x.mean() if x.size else 0.0
    pred = np.full_like(x, fill_value=mu, dtype=np.float64)
    resid = x - pred
    ssr_array = resid * resid
    sra = float(ssr_array.sum())
    ssr_ps = _prefix_sum(ssr_array)
    return sra, ssr_array, pred, ssr_ps

def predict_on_idxs_mean(array: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    if idxs.size == 0:
        return np.array([], dtype=np.float64)
    vals = array[idxs]
    mu = vals.mean()
    return np.full(idxs.shape, fill_value=mu, dtype=np.float64)

# -------------------------
# Polynomial regression (degree d)
# -------------------------
def _vandermonde_positions(n: int, degree: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    t = i / float(n) if n > 0 else np.array([], dtype=np.float64)
    V = np.vander(t, N=degree + 1, increasing=True)  # (n, d+1)
    return V

def calculate_poly_sra(array: np.ndarray, degree: int = 1, reg: float = 1e-8) \
        -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    x = array.astype(np.float64)
    n = x.size
    if n == 0:
        return 0.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), _prefix_sum(np.array([], dtype=np.float64))
    V = _vandermonde_positions(n, degree)
    A = V.T @ V
    if reg > 0:
        A.flat[:: A.shape[0] + 1] += reg
    b = V.T @ x
    beta = np.linalg.solve(A, b)
    pred = V @ beta
    resid = x - pred
    ssr_array = resid * resid
    sra = float(ssr_array.sum())
    ssr_ps = _prefix_sum(ssr_array)
    return sra, ssr_array, pred, ssr_ps

def predict_on_idxs_poly(array: np.ndarray, idxs: np.ndarray, degree: int = 1, reg: float = 1e-8) -> np.ndarray:
    if idxs.size == 0:
        return np.array([], dtype=np.float64)
    n = array.size
    t = idxs.astype(np.float64) / float(n) if n > 0 else idxs.astype(np.float64)
    V = np.vander(t, N=degree + 1, increasing=True)
    y = array[idxs].astype(np.float64)
    A = V.T @ V
    if reg > 0:
        A.flat[:: A.shape[0] + 1] += reg
    b = V.T @ y
    beta = np.linalg.solve(A, b)
    pred = V @ beta
    return pred

# -------------------------
# Kernel ridge regression (dense KRR)
# -------------------------
def _gaussian_kernel_matrix(n: int, length_scale: float) -> np.ndarray:
    ii = np.arange(n, dtype=np.float64).reshape(-1, 1)
    jj = np.arange(n, dtype=np.float64).reshape(1, -1)
    d2 = (ii - jj) ** 2
    denom = 2.0 * (length_scale ** 2)
    K = np.exp(-d2 / denom)
    return K

def _laplace_kernel_matrix(n: int, scale: float) -> np.ndarray:
    ii = np.arange(n, dtype=np.float64).reshape(-1, 1)
    jj = np.arange(n, dtype=np.float64).reshape(1, -1)
    d = np.abs(ii - jj)
    K = np.exp(-d / float(scale))
    return K

def calculate_krr_sra(array: np.ndarray, kernel: str = "rbf", kernel_param: float = 1.0, reg: float = 1e-3) \
        -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = array.astype(np.float64)
    n = x.size
    if n == 0:
        return 0.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([[]], dtype=np.float64), np.array([], dtype=np.float64), _prefix_sum(np.array([], dtype=np.float64))
    if kernel.lower() in ("rbf", "gaussian"):
        K = _gaussian_kernel_matrix(n, length_scale=float(kernel_param))
    elif kernel.lower() in ("laplace",):
        K = _laplace_kernel_matrix(n, scale=float(kernel_param))
    else:
        raise ValueError(f"Unknown kernel {kernel!r}")
    A = K.copy()
    A.flat[:: A.shape[0] + 1] += float(reg)
    alpha = np.linalg.solve(A, x)
    pred = K @ alpha
    resid = x - pred
    ssr_array = resid * resid
    sra = float(ssr_array.sum())
    ssr_ps = _prefix_sum(ssr_array)
    return sra, ssr_array, pred, K, alpha, ssr_ps

def predict_on_idxs_krr(array: np.ndarray, idxs: np.ndarray, kernel: str = "rbf", kernel_param: float = 1.0, reg: float = 1e-3, K_full: Optional[np.ndarray] = None, alpha_full: Optional[np.ndarray] = None) -> np.ndarray:
    if idxs.size == 0:
        return np.array([], dtype=np.float64)
    if (K_full is not None) and (alpha_full is not None):
        return (K_full[idxs, :] @ alpha_full).astype(np.float64)
    m = idxs.size
    ii = idxs.reshape(-1, 1).astype(np.float64)
    jj = idxs.reshape(1, -1).astype(np.float64)
    if kernel.lower() in ("rbf", "gaussian"):
        d2 = (ii - jj) ** 2
        Ksub = np.exp(-d2 / (2.0 * (float(kernel_param) ** 2)))
    elif kernel.lower() in ("laplace",):
        d = np.abs(ii - jj)
        Ksub = np.exp(-d / float(kernel_param))
    else:
        raise ValueError(f"Unknown kernel {kernel!r}")
    ysub = array[idxs].astype(np.float64)
    A = Ksub.copy()
    A.flat[:: A.shape[0] + 1] += float(reg)
    alpha_sub = np.linalg.solve(A, ysub)
    preds = Ksub @ alpha_sub
    return preds.astype(np.float64)

# -------------------------
# Unified factory
# -------------------------
def build_regressor_sra(array: np.ndarray, family: str, **kwargs):
    fam = family.lower()
    if fam == "mean":
        sra, ssr_array, pred, ssr_ps = calculate_mean_sra(array)
        return sra, ssr_array, pred, None, None, ssr_ps
    elif fam == "poly":
        degree = int(kwargs.get("degree", 1))
        reg = float(kwargs.get("reg", 1e-8))
        sra, ssr_array, pred, ssr_ps = calculate_poly_sra(array, degree=degree, reg=reg)
        return sra, ssr_array, pred, None, None, ssr_ps
    elif fam == "krr":
        kernel = kwargs.get("kernel", "gaussian")
        kernel_param = float(kwargs.get("kernel_param", 1.0))
        reg = float(kwargs.get("reg", 1e-3))
        sra, ssr_array, pred, K, alpha, ssr_ps = calculate_krr_sra(array, kernel=kernel, kernel_param=kernel_param, reg=reg)
        return sra, ssr_array, pred, K, alpha, ssr_ps
    else:
        raise ValueError(f"Unknown family {family!r}")
