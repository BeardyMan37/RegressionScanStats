from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

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

def _range_sum(ps: np.ndarray, a: int, b: int) -> float:
    # inclusive [a,b]
    return float(ps[b + 1] - ps[a])

@dataclass
class MeanPrecomp:
    """Sufficient stats for mean model on a fixed row."""
    x: np.ndarray
    ps_x: np.ndarray
    ps_x2: np.ndarray
    sum_x: float
    sum_x2: float
    n: int

def mean_precompute(x: np.ndarray) -> MeanPrecomp:
    x = np.asarray(x, dtype=np.float64)
    ps_x = _prefix_sum(x)
    ps_x2 = _prefix_sum(x * x)
    return MeanPrecomp(
        x=x,
        ps_x=ps_x,
        ps_x2=ps_x2,
        sum_x=float(ps_x[-1]),
        sum_x2=float(ps_x2[-1]),
        n=int(x.size),
    )

def mean_window_sse(pre: MeanPrecomp, a: int, b: int) -> float:
    """SSE on window [a,b] for best constant fit (mean)."""
    m = b - a + 1
    if m <= 0:
        return 0.0
    s1 = _range_sum(pre.ps_x, a, b)
    s2 = _range_sum(pre.ps_x2, a, b)
    # SSE = sum(x^2) - (sum(x)^2)/m
    return float(max(s2 - (s1 * s1) / m, 0.0))

def mean_outside_sse(pre: MeanPrecomp, a: int, b: int) -> float:
    """SSE on complement of [a,b] for best constant fit."""
    m_in = b - a + 1
    m_out = pre.n - m_in
    if m_out <= 0:
        return 0.0
    s1_in = _range_sum(pre.ps_x, a, b)
    s2_in = _range_sum(pre.ps_x2, a, b)
    s1_out = pre.sum_x - s1_in
    s2_out = pre.sum_x2 - s2_in
    return float(max(s2_out - (s1_out * s1_out) / m_out, 0.0))

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

@dataclass
class PolyPrecomp:
    """
    Precompute sufficient statistics for polynomial regression on a fixed row.
    Uses t=i/n (consistent with your existing Vandermonde code).
    """
    x: np.ndarray
    t: np.ndarray
    degree: int
    reg: float
    ps_S: list[np.ndarray]  # prefix sums of t^m for m=0..2d
    ps_T: list[np.ndarray]  # prefix sums of t^k * x for k=0..d
    ps_Q: np.ndarray        # prefix sums of x^2
    n: int

def poly_precompute(x: np.ndarray, degree: int = 1, reg: float = 1e-8) -> PolyPrecomp:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n > 0:
        i = np.arange(n, dtype=np.float64)
        t = i / float(n)
    else:
        t = np.array([], dtype=np.float64)

    d = int(degree)

    ps_S = []
    for m in range(0, 2 * d + 1):
        ps_S.append(_prefix_sum(t ** m if n > 0 else np.array([], dtype=np.float64)))

    ps_T = []
    for k in range(0, d + 1):
        ps_T.append(_prefix_sum((t ** k) * x if n > 0 else np.array([], dtype=np.float64)))

    ps_Q = _prefix_sum(x * x)

    return PolyPrecomp(
        x=x,
        t=t,
        degree=d,
        reg=float(reg),
        ps_S=ps_S,
        ps_T=ps_T,
        ps_Q=ps_Q,
        n=n,
    )

def _poly_S(pre: PolyPrecomp, m: int, a: int, b: int) -> float:
    return _range_sum(pre.ps_S[m], a, b)

def _poly_T(pre: PolyPrecomp, k: int, a: int, b: int) -> float:
    return _range_sum(pre.ps_T[k], a, b)

def _poly_Q(pre: PolyPrecomp, a: int, b: int) -> float:
    return _range_sum(pre.ps_Q, a, b)

def poly_fit_window(pre: PolyPrecomp, a: int, b: int) -> np.ndarray:
    """Fit beta on window [a,b] (inclusive)."""
    d = pre.degree
    A = np.empty((d + 1, d + 1), dtype=np.float64)
    rhs = np.empty(d + 1, dtype=np.float64)

    for k in range(d + 1):
        rhs[k] = _poly_T(pre, k, a, b)
        for j in range(d + 1):
            A[k, j] = _poly_S(pre, k + j, a, b)

    if pre.reg > 0:
        A.flat[:: A.shape[0] + 1] += pre.reg

    beta = np.linalg.solve(A, rhs)
    return beta

def poly_window_sse(pre: PolyPrecomp, a: int, b: int) -> float:
    """SSE on window [a,b] for best degree-d polynomial fit."""
    if b < a:
        return 0.0
    d = pre.degree
    beta = poly_fit_window(pre, a, b)

    # Rebuild A and rhs cheaply (d is small; OK)
    A = np.empty((d + 1, d + 1), dtype=np.float64)
    rhs = np.empty(d + 1, dtype=np.float64)
    for k in range(d + 1):
        rhs[k] = _poly_T(pre, k, a, b)
        for j in range(d + 1):
            A[k, j] = _poly_S(pre, k + j, a, b)

    Q = _poly_Q(pre, a, b)
    sse = Q - 2.0 * float(beta @ rhs) + float(beta @ (A @ beta))
    return float(max(sse, 0.0))

def poly_predict_window(pre: PolyPrecomp, a: int, b: int) -> np.ndarray:
    """Predictions on indices [a,b] from the polynomial fit on that same window."""
    if b < a:
        return np.array([], dtype=np.float64)
    beta = poly_fit_window(pre, a, b)
    tseg = pre.t[a:b + 1]
    Vseg = np.vander(tseg, N=pre.degree + 1, increasing=True)
    return (Vseg @ beta).astype(np.float64)

def _poly_inside_stats(pre, a: int, b: int):
    """
    Return inside stats over [a,b]:
      S_in[m] = sum t^m, m=0..2d
      T_in[k] = sum t^k x, k=0..d
      Q_in    = sum x^2
    """
    d = int(pre.degree)
    S_in = np.empty(2 * d + 1, dtype=np.float64)
    T_in = np.empty(d + 1, dtype=np.float64)

    for m in range(2 * d + 1):
        S_in[m] = _range_sum(pre.ps_S[m], a, b)
    for k in range(d + 1):
        T_in[k] = _range_sum(pre.ps_T[k], a, b)

    Q_in = _range_sum(pre.ps_Q, a, b)
    return S_in, T_in, float(Q_in)

def _poly_global_stats(pre):
    """Return global stats over [0, n-1] (uses prefix sums)."""
    n = int(pre.n)
    if n <= 0:
        d = int(pre.degree)
        return np.zeros(2 * d + 1), np.zeros(d + 1), 0.0

    a, b = 0, n - 1
    return _poly_inside_stats(pre, a, b)

def _poly_sse_from_stats(S: np.ndarray, T: np.ndarray, Q: float, d: int, reg: float) -> float:
    """
    Compute SSE given sufficient stats:
      A_{k,j} = S[k+j]
      b_k = T[k]
      SSE = Q - 2 beta^T b + beta^T A beta
    """
    A = np.empty((d + 1, d + 1), dtype=np.float64)
    rhs = np.empty(d + 1, dtype=np.float64)

    for k in range(d + 1):
        rhs[k] = T[k]
        for j in range(d + 1):
            A[k, j] = S[k + j]

    if reg > 0:
        A.flat[:: A.shape[0] + 1] += float(reg)

    beta = np.linalg.solve(A, rhs)
    sse = float(Q - 2.0 * (beta @ rhs) + beta @ (A @ beta))
    return float(max(sse, 0.0))

def poly_outside_sse(pre, a: int, b: int) -> float:
    """
    CHANGE: SSE on the complement of [a,b] for best degree-d polynomial fit,
    using additive sufficient statistics (global - inside).

    This is writeup-correct for fitting on the complement in the ORIGINAL coordinate system,
    even though the complement is disjoint.
    """
    n = int(pre.n)
    if n <= 0:
        return 0.0

    # If outside is empty, SSE is 0
    m_in = b - a + 1
    if m_in >= n:
        return 0.0

    d = int(pre.degree)
    reg = float(pre.reg)

    S_all, T_all, Q_all = _poly_global_stats(pre)
    S_in,  T_in,  Q_in  = _poly_inside_stats(pre, a, b)

    S_out = S_all - S_in
    T_out = T_all - T_in
    Q_out = Q_all - Q_in

    return _poly_sse_from_stats(S_out, T_out, Q_out, d=d, reg=reg)

# -------------------------
# Kernel ridge regression (dense KRR)
# -------------------------
def _positions_t(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    return i / float(n)

def _rbf_kernel_from_t(t1: np.ndarray, t2: np.ndarray, length_scale: float) -> np.ndarray:
    t1 = t1.reshape(-1, 1)
    t2 = t2.reshape(1, -1)
    d2 = (t1 - t2) ** 2
    return np.exp(-d2 / (2.0 * length_scale * length_scale))

def _laplace_kernel_from_t(t1: np.ndarray, t2: np.ndarray, scale: float) -> np.ndarray:
    t1 = t1.reshape(-1, 1)
    t2 = t2.reshape(1, -1)
    d = np.abs(t1 - t2)
    return np.exp(-d / float(scale))

def calculate_krr_sra(array: np.ndarray, kernel: str = "rbf", kernel_param: float = 1.0, reg: float = 1e-3) \
        -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = array.astype(np.float64)
    n = x.size
    if n == 0:
        return 0.0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([[]], dtype=np.float64), np.array([], dtype=np.float64), _prefix_sum(np.array([], dtype=np.float64))

    t = _positions_t(n)

    if kernel.lower() in ("rbf", "gaussian"):
        K = _rbf_kernel_from_t(t, t, float(kernel_param))
    elif kernel.lower() in ("laplace",):
        K = _laplace_kernel_from_t(t, t, float(kernel_param))
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

def predict_on_idxs_krr(array: np.ndarray, idxs: np.ndarray, kernel: str = "rbf", kernel_param: float = 1.0, reg: float = 1e-3,
                        K_full: Optional[np.ndarray] = None, alpha_full: Optional[np.ndarray] = None) -> np.ndarray:
    if idxs.size == 0:
        return np.array([], dtype=np.float64)

    idxs = np.asarray(idxs, dtype=np.int64)

    # If full fit provided, just evaluate it at idxs
    if (K_full is not None) and (alpha_full is not None):
        return (K_full[idxs, :] @ alpha_full).astype(np.float64)

    # Otherwise: fit on the subset itself (in-sample), but using normalized coordinates
    x = np.asarray(array, dtype=np.float64)
    t = _positions_t(x.size)
    t_sub = t[idxs]
    ysub = x[idxs]

    if kernel.lower() in ("rbf", "gaussian"):
        Ksub = _rbf_kernel_from_t(t_sub, t_sub, float(kernel_param))
    elif kernel.lower() in ("laplace",):
        Ksub = _laplace_kernel_from_t(t_sub, t_sub, float(kernel_param))
    else:
        raise ValueError(f"Unknown kernel {kernel!r}")

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
        pre = mean_precompute(array)
        return sra, ssr_array, pred, None, None, ssr_ps, pre

    elif fam == "poly":
        degree = int(kwargs.get("degree", 1))
        reg = float(kwargs.get("reg", 1e-8))
        sra, ssr_array, pred, ssr_ps = calculate_poly_sra(array, degree=degree, reg=reg)
        pre = poly_precompute(array, degree=degree, reg=reg)
        return sra, ssr_array, pred, None, None, ssr_ps, pre

    elif fam == "krr":
        kernel = kwargs.get("kernel", "gaussian")
        reg = float(kwargs.get("reg", 1e-3))
        if "kernel_param" in kwargs:
            kernel_param = float(kwargs["kernel_param"])
        elif "kernel_param_scale" in kwargs:
            s = max(float(kwargs["kernel_param_scale"]), 1e-12)
            kernel_param = 1.0 / s
        else:
            kernel_param = 1.0
        sra, ssr_array, pred, K, alpha, ssr_ps = calculate_krr_sra(
            array, kernel=kernel, kernel_param=kernel_param, reg=reg
        )
        return sra, ssr_array, pred, K, alpha, ssr_ps, None

    else:
        raise ValueError(f"Unknown family {family!r}")
