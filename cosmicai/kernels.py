from __future__ import annotations
import math, numpy as np
from typing import Dict, Tuple
from .config import _KERNEL_ALPHA

_kernel_cache: Dict[Tuple[int, float, str], np.ndarray] = {}

def precompute_kernel(L: int, w: float, kind: str = "gaussian", alpha: float = _KERNEL_ALPHA) -> np.ndarray:
    if w <= 0:
        raise ValueError("Kernel width w must be positive.")
    idx = np.arange(L, dtype=np.float64)
    D = np.abs(np.subtract.outer(idx, idx))
    if kind == "laplace":
        b = max(float(w) / math.sqrt(2.0), 1e-12)
        return np.exp(-D / b)
    elif kind == "gaussian":
        return np.exp(-(D * D) / (w * w))
    elif kind == "laplace_rt":
        if not (1.0 < alpha < 2.0):
            raise ValueError("laplace_rt: alpha must be in (1, 2).")
        b = max(float(w) / (math.log(2.0) ** (1.0 / alpha)), 1e-12)
        return np.exp(-np.power(D / b, alpha))
    raise ValueError(f"Unknown kernel kind: {kind!r}")

def _get_kernel(n: int, w: float, kind: str = "gaussian") -> np.ndarray:
    key = (n, float(w), kind)
    K = _kernel_cache.get(key)
    if K is None:
        K = precompute_kernel(n, w, kind)
        _kernel_cache[key] = K
    return K
