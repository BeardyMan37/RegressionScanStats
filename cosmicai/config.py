# cosmicai/config.py
from enum import Enum

class KernelKind(str, Enum):
    GAUSSIAN = "gaussian"
    LAPLACE  = "laplace"

ref_freq: float = 0.0625

# private module state
_KERNEL_KIND: KernelKind = KernelKind.GAUSSIAN
_KERNEL_ALPHA: float = 1.5  # used by any RT variants if you add back
_SUPER_RESOLVE_BASE: int = 450

def set_kernel(kind: str):
    global _KERNEL_KIND
    k = kind.lower()
    if k not in ("gaussian", "laplace"):
        raise ValueError(f"Unknown kernel kind: {kind}")
    _KERNEL_KIND = KernelKind(k)

def get_kernel_kind() -> KernelKind:
    return _KERNEL_KIND

def get_super_resolve_base() -> int:
    return _SUPER_RESOLVE_BASE

def get_alpha() -> float:
    return _KERNEL_ALPHA
