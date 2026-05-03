from __future__ import annotations
import math, numpy as np
from typing import Dict, Tuple
from .config import _KERNEL_ALPHA

kernel_cache: Dict[Tuple[int, float, str], np.ndarray] = {}
kerne_denom_cache: dict = {}

def _kernel_matrix(n: int, w: float, kind: str) -> np.ndarray:
    """
    Full kernel matrix K[i,j] based on index distance |i-j|.
    Gaussian: exp(-(d^2)/(w^2))
    Laplace:  exp(-(d)/w)   (treat w as sigma/scale here)
    """
    idx = np.arange(n, dtype=np.float64)
    D = np.abs(idx[:, None] - idx[None, :])
    if kind == "gaussian":
        return np.exp(-(D * D) / (w * w))
    elif kind == "laplace":
        sigma = max(float(w), 1e-12)
        return np.exp(-D / sigma)
    else:
        raise ValueError(f"Unknown kernel kind: {kind!r}")
    
def truncated_kernel_vector(w: float, r: int, kind: str = "gaussian") -> np.ndarray:
    # w = max(float(w), 3)
    d = np.arange(r + 1, dtype=np.float64)
    if kind == "gaussian":
        try:
            return np.exp(-(d * d) / (w * w))
        except Exception as e:
            print("Printing in exception", w)
    elif kind == "laplace":
        return np.exp(-d / w)
    raise ValueError(f"Unknown kernel kind: {kind!r}")

def precompute_kernel(L: int, w: float, kind: str = "gaussian", alpha: float = _KERNEL_ALPHA) -> np.ndarray:
    if w <= 0:
        raise ValueError("Kernel width w must be positive.")
    idx = np.arange(L, dtype=np.float64)
    D = np.abs(np.subtract.outer(idx, idx))
    if kind == "laplace":
        return np.exp(-D / w)
    elif kind == "gaussian":
        return np.exp(-(D * D) / (w * w))
    elif kind == "laplace_rt":
        if not (1.0 < alpha < 2.0):
            raise ValueError("laplace_rt: alpha must be in (1, 2).")
        b = max(float(w) / (math.log(2.0) ** (1.0 / alpha)), 1e-12)
        return np.exp(-np.power(D / b, alpha))
    raise ValueError(f"Unknown kernel kind: {kind!r}")

def get_kernel(n: int, w: float, kind: str = "gaussian") -> np.ndarray:
    key = (n, float(w), kind)
    K = kernel_cache.get(key)
    if K is None:
        K = precompute_kernel(n, w, kind)
        kernel_cache[key] = K
    return K

def get_kernel_and_denom(n: int, w, kind) -> Tuple[np.ndarray, np.ndarray]:
    key = (int(n), float(w), str(kind))
    cached = kerne_denom_cache.get(key)
    if cached is not None:
        return cached
    W = get_kernel(n, w, kind)
    denom = W.sum(axis=1).astype(np.float64)
    kerne_denom_cache[key] = (W, denom)
    return W, denom
