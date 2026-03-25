from __future__ import annotations

import logging
logging.getLogger().setLevel(logging.ERROR)

import kats
import os
import time
import stumpy
from typing import List, Optional
import ruptures as rpt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector

from cosmicai.kernels import get_kernel_and_denom
from cosmicai.predictors import predict_on_idxs
from cosmicai.scoring import calculate_gaussian_sra_with_nd, calculate_laplace_sra_fast
from gen_synth_data import SpectrumSpec, generate_synthetic_dataset
import cosmicai.regressors as rg
from cosmicai.config import KernelKind, ref_freq, set_kernel_kind
from cosmicai.scan import scan_row_with_nwkr, scan_row_with_nwkr_naive

def warmup_stumpy(n: int, m: int, *, z_norm: bool = True, ignore_trivial: bool = True):
    x = np.random.default_rng(0).normal(size=n).astype(np.float64)
    _ = stumpy.stump(x, m=m, ignore_trivial=ignore_trivial, normalize=z_norm)

warmup_stumpy(n=100, m=10)

def interval_length(a: int, b: int) -> int:
    return max(0, b - a + 1)


def interval_intersection(ab, cd) -> int:
    a, b = ab
    c, d = cd
    lo = max(a, c)
    hi = min(b, d)
    return max(0, hi - lo + 1)

def localization_metrics(pred_ab, gt_ab):
    """
    pred_ab : (a_hat, b_hat)
    gt_ab   : (a_gt, b_gt)

    Returns:
        coverage, parsimony, localization_score
    """
    a_hat, b_hat = pred_ab
    a_gt,  b_gt  = gt_ab

    inter = interval_intersection(pred_ab, gt_ab)
    gt_len = interval_length(a_gt, b_gt)
    pred_len = interval_length(a_hat, b_hat)

    if gt_len == 0 or pred_len == 0:
        return 0.0, 0.0, 0.0

    recall = inter / gt_len
    precision = inter / pred_len

    loc_score = (recall * precision) ** 0.5

    return recall, precision, loc_score

def _resolve_krr_params(family_kwargs: dict) -> dict:
    kw = dict(family_kwargs)
    if "kernel_param" not in kw:
        if "kernel_param_scale" in kw:
            s = float(kw["kernel_param_scale"])
            s = max(s, 1e-12)
            kw["kernel_param"] = 1.0 / s   # FIX for normalized t
        else:
            kw["kernel_param"] = 1.0
    kw.pop("kernel_param_scale", None)
    return kw

def _sse_subset_krr_from_full_kernel(
    x: np.ndarray,
    idxs: np.ndarray,
    K_full: np.ndarray,
    reg: float,
) -> float:
    """
    CHANGE: Compute subset KRR in-sample SSE using K_full slicing.
    Fits KRR on subset S with K_SS, predicts on S, returns ||y - K_SS alpha||^2.
    """
    if idxs.size == 0:
        return 0.0
    idxs = np.asarray(idxs, dtype=np.int64)
    y = x[idxs].astype(np.float64, copy=False)

    Kss = K_full[np.ix_(idxs, idxs)].astype(np.float64, copy=False)
    A = Kss.copy()
    A.flat[:: A.shape[0] + 1] += float(reg)

    alpha = np.linalg.solve(A, y)
    pred = Kss @ alpha
    resid = y - pred
    return float(resid @ resid)

def sse_subset(x: np.ndarray, idxs: np.ndarray,
               family: str,
               family_kwargs: dict,
               K_full: Optional[np.ndarray] = None) -> float:
    """Compute SSE of best regressor in `family` on subset `idxs`."""
    if idxs.size == 0:
        return 0.0

    fam = family.lower()
    if fam == "mean":
        vals = x[idxs]
        mu = vals.mean()
        resid = vals - mu
        return float(np.sum(resid * resid))

    elif fam == "poly":
        degree = int(family_kwargs.get("degree", 1))
        reg = float(family_kwargs.get("reg", 1e-8))
        preds = rg.predict_on_idxs_poly(x, idxs, degree=degree, reg=reg)
        resid = x[idxs] - preds
        return float(np.sum(resid * resid))

    elif fam == "krr":
        reg_val = float(family_kwargs.get("reg", 1e-1))
        if K_full is not None:
            return _sse_subset_krr_from_full_kernel(x, idxs, K_full, reg_val)

        kernel = family_kwargs.get("kernel", "rbf")
        kernel_param = float(family_kwargs.get("kernel_param", 1.0))
        preds = rg.predict_on_idxs_krr(
            x,
            idxs,
            kernel=kernel,
            kernel_param=kernel_param,
            reg=reg_val,
            K_full=None,
            alpha_full=None,
        )
        resid = x[idxs] - preds
        return float(np.sum(resid * resid))

    else:
        raise ValueError(f"Unknown family {family!r}")

# def scan_row_krr_two_model(
#     x: np.ndarray,
#     W: int,
#     h0_params: dict,
#     h1_family: str = "poly",
#     h1_params: dict | None = None,
# ) -> tuple[float, tuple[int, int]]:
#     """
#     Scan row using:
#       - H0: smooth global KRR (fixed hyperparameters, 'h0_params')
#       - H1: same as H0 outside the window, but *local flexible model* (h1_family)
#             inside the window.

#     Score:
#         score(I) = 1 - (SSE_out_H0 + SSE_in_local) / SSE_all_H0

#     Parameters
#     ----------
#     x : np.ndarray
#         1D row (length n).
#     W : int
#         Maximum window length.
#     h0_params : dict
#         Params for the global KRR under H0, e.g.
#         {"kernel": "rbf", "kernel_param": 50.0, "reg": 1e-1}
#     h1_family : {"mean","poly","krr"}
#         Family for the local model inside the window.
#     h1_params : dict
#         Params for the local model (degree, reg, kernel_param, etc.)

#     Returns
#     -------
#     best_score : float
#     best_ab    : (a, b)
#     """
#     if h1_params is None:
#         h1_params = {}

#     n = x.size
#     if n == 0:
#         return -np.inf, (0, 0)

#     # ---- 1) Global *smooth* KRR as H0 ----
#     # This gives:
#     #   sra_all = SSE_all under H0
#     #   pred_all = predictions under H0
#     sra_all, ssr_array, pred_all, K_full, alpha_full, ssr_ps = rg.build_regressor_sra(
#         x, family="krr", **h0_params
#     )
#     # Make sure sra_all is the SSE under H0:
#     resid_all = x - pred_all
#     se_all = resid_all * resid_all
#     sse_all = float(se_all.sum())

#     # ---- 2) Scan windows, replacing only INSIDE with flexible model ----
#     best_score = -np.inf
#     best_ab = (0, 0)

#     idx_all = np.arange(n, dtype=np.int32)
#     max_W = int(min(W, n))

#     for m in range(1, max_W + 1):
#         for a in range(0, n - m + 1):
#             b = a + m - 1
#             inside = idx_all[a:b + 1]

#             # SSE contribution of H0 inside the window (for reference)
#             sse_in_h0 = float(se_all[a:b + 1].sum())
#             # SSE outside DOES NOT change under H1: we keep H0 there
#             sse_out = sse_all - sse_in_h0

#             # SSE of local model inside the window (more flexible)
#             sse_in_local = sse_subset(x, inside, h1_family, h1_params)

#             sse_h1 = sse_out + sse_in_local

#             # LRT-like normalized score (higher is "more anomalous")
#             score = 1.0 - (sse_h1 / sse_all)

#             if score > best_score:
#                 best_score = score
#                 best_ab = (a, b)

#     return best_score, best_ab

# -------------------------------------------------
# Naive scan using a regression family
# -------------------------------------------------

def scan_row_with_family(x: np.ndarray,
                         W: int,
                         family: str,
                         family_kwargs: dict) -> tuple[float, tuple[int, int]]:
    
    n = x.size
    if n == 0:
        return -np.inf, (0, 0)

    fam = family.lower()
    max_W = int(min(W, n))
    if fam == "krr":
        family_kwargs = _resolve_krr_params(family_kwargs)

    # ---- Precompute per-row stats (writeup-style) ----
    mean_pre = None
    poly_pre = None

    if fam == "mean":
        mean_pre = rg.mean_precompute(x)
        # global SSE (SRA) from stats (O(1))
        sra_all = rg.mean_window_sse(mean_pre, 0, n - 1)

    elif fam == "poly":
        degree = int(family_kwargs.get("degree", 1))
        reg = float(family_kwargs.get("reg", 1e-8))
        poly_pre = rg.poly_precompute(x, degree=degree, reg=reg)    
        sra_all = rg.poly_window_sse(poly_pre, 0, n - 1)

    else:
        sra_all, *_ = rg.build_regressor_sra(x, family, **family_kwargs)

    if sra_all <= 1e-18:
        return 0.0, (0, 0)

    best_score = -np.inf
    best_ab = (0, 0)
    idx_all = np.arange(n, dtype=np.int32)

    # ---- Scan all windows up to W ----
    for m in range(1, max_W + 1):
        for a in range(0, n - m + 1):
            b = a + m - 1

            if fam == "mean":
                sse_in = rg.mean_window_sse(mean_pre, a, b)
                sse_out = rg.mean_outside_sse(mean_pre, a, b)

            elif fam == "poly":
                sse_in = rg.poly_window_sse(poly_pre, a, b)
                sse_out = rg.poly_outside_sse(poly_pre, a, b)

            else:
                inside = idx_all[a:b + 1]
                outside = np.concatenate([idx_all[:a], idx_all[b + 1:]])
                if fam == "krr":
                    # compute K once per row
                    if "K_full_row" not in locals():
                        _, _, _, K_full_row, _, _, _ = rg.build_regressor_sra(x, "krr", **family_kwargs)
                    sse_in = sse_subset(x, inside, fam, family_kwargs, K_full=K_full_row)
                    sse_out = sse_subset(x, outside, fam, family_kwargs, K_full=K_full_row)
                else:
                    sse_in = sse_subset(x, inside, fam, family_kwargs)
                    sse_out = sse_subset(x, outside, fam, family_kwargs)

            score = 1.0 - (sse_in + sse_out) / sra_all
            if score > best_score:
                best_score = score
                best_ab = (a, b)

    return float(best_score), best_ab

def scan_row_with_nwkr_naive_for_benchmark(
    x: np.ndarray,
    W: int,
    R: int,
    kernel_kind: str,
    which: str = "unmasked_varlen",  # {"masked_varlen","unmasked_varlen","fixed"}
    buffer: int = 0,
    sr_factor: int = 1,
) -> tuple[float, tuple[int, int]]:
    """
    Adapter: run CosmicAI NWKR scanner on a single row and return (score, (a,b)).

    Notes:
    - scan_row_with_nwkr computes its own internal w from (ref_freq, freq_step, L/16 cap).
    - We supply a synthetic frequency grid; we choose freq_step to make R ~= W, so that
      internal w = round(max(3, min(W, L/16))) in your current scan code.
    """
    x = x.astype(np.float64)
    n = x.size
    if n == 0:
        return -np.inf, (0, 0)

    # Choose a frequency step so that R = ref_freq / freq_step ≈ W.
    # This makes the NWKR internal w track your benchmark W whenever W <= L/16.
    freq_step = float(ref_freq) / float(max(1, W))
    freqs = (np.arange(n, dtype=np.float64) * freq_step)

    # No ignore ranges for synthetic benchmark
    ignore_ranges = []
    flags = []

    set_kernel_kind(kernel_kind)  # "gaussian" or "laplace" (matching your config)

    out = scan_row_with_nwkr_naive((0, x, ignore_ranges, flags, freqs, buffer, sr_factor, W, R, W))

    if which == "masked_varlen":
        best_win = out[1]
        best_sc = out[2]
    elif which == "unmasked_varlen":
        best_win = out[8]
        best_sc = out[9]
    elif which == "fixed":
        best_win = out[13]
        best_sc = out[14]
    else:
        raise ValueError(f"Unknown which={which!r}")

    a, b = best_win
    return float(best_sc), (int(a), int(b))

def scan_row_with_nwkr_for_benchmark(
    x: np.ndarray,
    W: int,
    R: int,
    kernel_kind: str,
    which: str = "unmasked_varlen",  # {"masked_varlen","unmasked_varlen","fixed"}
    buffer: int = 0,
    sr_factor: int = 1,
) -> tuple[float, tuple[int, int]]:
    """
    Adapter: run CosmicAI NWKR scanner on a single row and return (score, (a,b)).

    Notes:
    - scan_row_with_nwkr computes its own internal w from (ref_freq, freq_step, L/16 cap).
    - We supply a synthetic frequency grid; we choose freq_step to make R ~= W, so that
      internal w = round(max(3, min(W, L/16))) in your current scan code.
    """
    x = x.astype(np.float64)
    n = x.size
    if n == 0:
        return -np.inf, (0, 0)

    # Choose a frequency step so that R = ref_freq / freq_step ≈ W.
    # This makes the NWKR internal w track your benchmark W whenever W <= L/16.
    freq_step = float(ref_freq) / float(max(1, W))
    freqs = (np.arange(n, dtype=np.float64) * freq_step)

    # No ignore ranges for synthetic benchmark
    ignore_ranges = []
    flags = []

    set_kernel_kind(kernel_kind)  # "gaussian" or "laplace" (matching your config)

    out = scan_row_with_nwkr((0, x, ignore_ranges, flags, freqs, buffer, sr_factor, W, R, W))

    if which == "masked_varlen":
        best_win = out[1]
        best_sc = out[2]
    elif which == "unmasked_varlen":
        best_win = out[8]
        best_sc = out[9]
    elif which == "fixed":
        best_win = out[13]
        best_sc = out[14]
    else:
        raise ValueError(f"Unknown which={which!r}")

    a, b = best_win
    return float(best_sc), (int(a), int(b))

def scan_row_with_kernelcpd_rbf_c(
    x: np.ndarray,
    W: int,
    *,
    min_size: int = 2,
    gamma: float | None = None,     # None = median heuristic in ruptures
    standardize: bool = True,
):
    """
    KernelCPD (RBF, C impl) adapted to epidemic segment detection:
    - force n_bkps=2
    - take middle segment as anomaly
    Returns (score, (a,b)) with inclusive (a,b).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2 * min_size + 2:
        return 0.0, (0, -1)

    if standardize:
        mu = x.mean()
        sd = x.std()
        x0 = (x - mu) / (sd + 1e-12)
    else:
        x0 = x

    signal = x0.reshape(-1, 1)

    params = None if gamma is None else {"gamma": float(gamma)}
    algo = rpt.KernelCPD(kernel="rbf", params=params, min_size=int(min_size)).fit(signal)

    # Force exactly two CPs => 3 segments; middle is epidemic segment.
    bkps = algo.predict(n_bkps=2)  # [cp1, cp2, n]
    cp1, cp2 = int(bkps[0]), int(bkps[1])

    a, b = cp1, cp2 - 1

    # Enforce your width cap by shrinking around the center if needed.
    if b - a + 1 > W:
        mid = (a + b) // 2
        half = W // 2
        a = max(0, mid - half)
        b = min(n - 1, a + W - 1)

    if b <= a:
        return 0.0, (0, -1)

    # Step-like evidence score: standardized mean shift inside vs outside.
    inside = x0[a:b+1]
    outside = np.concatenate([x0[:a], x0[b+1:]])
    if outside.size == 0:
        return 0.0, (0, -1)
    score = float(abs(inside.mean() - outside.mean()))
    return score, (a, b)

def scan_row_with_stumpy_discord_for_benchmark(
    x: np.ndarray,
    W: int,
    *,
    m_list=None,
    detrend_deg: int | None = 2,
    z_norm: bool = False,          # key for step anomalies
    ignore_trivial: bool = True,
):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 20:
        return 0.0, (0, -1)

    # Detrend helps MP focus on the step rather than smooth baseline
    if detrend_deg is not None:
        t = np.arange(n, dtype=np.float64)
        coef = np.polyfit(t, x, detrend_deg)
        x0 = x - np.polyval(coef, t)
    else:
        x0 = x

    if m_list is None:
        # Include ~0.5W which often matches step width when W is a cap.
        base = [max(8, int(round(W * f))) for f in (0.2, 0.35, 0.5, 0.7, 1.0)]
        m_list = sorted({m for m in base if 4 <= m <= min(W, n - 2)})

    best_score = -np.inf
    best_ab = (0, -1)

    for m in m_list:
        mp = stumpy.stump(x0, m=int(m), ignore_trivial=ignore_trivial, normalize=z_norm)
        prof = np.asarray(mp[:, 0], dtype=np.float64)
        good = np.isfinite(prof)
        if not np.any(good):
            continue
        idx = int(np.argmax(np.where(good, prof, -np.inf)))
        score = float(prof[idx])
        a, b = idx, idx + int(m) - 1
        if b >= n:
            continue
        if score > best_score:
            best_score = score
            best_ab = (a, b)

    if not np.isfinite(best_score) or best_ab[1] <= best_ab[0]:
        return 0.0, (0, -1)
    return best_score, best_ab


def scan_row_with_stumpy_multilen_for_benchmark(
    x: np.ndarray,
    W: int,
    *,
    m_list: list[int] | None = None,
    z_norm: bool = True,
    ignore_trivial: bool = True,
):
    """
    Evaluate multiple subsequence lengths and return the best discord interval.

    score = max over m of discord profile distance
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0.0, (0, -1)

    if m_list is None:
        # Small default grid (tweak as you like); must be <= W and <= n-2
        candidates = sorted(set([max(4, W // 4), max(4, W // 2), max(4, W)]))
    else:
        candidates = sorted(set(int(mm) for mm in m_list))

    best_score = -np.inf
    best_ab = (0, -1)

    for m in candidates:
        m = int(min(m, W, n - 2))
        if m < 4 or m > n - 2:
            continue
        score, (a, b) = scan_row_with_stumpy_discord_for_benchmark(
            x, W=W, m=m, z_norm=z_norm, ignore_trivial=ignore_trivial
        )
        if score > best_score and b >= a:
            best_score = score
            best_ab = (a, b)

    if not np.isfinite(best_score) or best_ab[1] < best_ab[0]:
        return 0.0, (0, -1)

    return float(best_score), best_ab

# def _best_interval_from_cps(cps: list[int], n: int, W: int) -> tuple[int, int] | None:
#     """Pick the best (a,b) from candidate change points, enforcing max length W."""
#     cps = sorted({cp for cp in cps if 1 <= cp <= n - 1})
#     if len(cps) < 2:
#         return None

#     best = None
#     # try all pairs; for n=500 and cps small this is fine
#     for i in range(len(cps) - 1):
#         for j in range(i + 1, len(cps)):
#             a = cps[i]
#             b_excl = cps[j]
#             if b_excl <= a:
#                 continue
#             if b_excl - a > W:
#                 break  # cps sorted: larger j => longer interval
#             b = b_excl - 1  # inclusive
#             best = (a, b) if best is None else best
#     return best


# def _mean_shift_score(x: np.ndarray, a: int, b: int) -> float:
#     """Simple step-like evidence score (bigger => stronger mean shift)."""
#     n = x.size
#     inside = x[a:b+1]
#     if inside.size == 0:
#         return 0.0
#     outside = np.concatenate([x[:a], x[b+1:]]) if (a > 0 or b + 1 < n) else np.array([], dtype=x.dtype)
#     if outside.size == 0:
#         return 0.0
#     return float(abs(inside.mean() - outside.mean()))


def scan_row_with_kats_cusum_for_benchmark(
    x: np.ndarray,
    W: int,
    *,
    alpha: float = 0.01,
    standardize: bool = True,
    detrend_deg: int | None = 2,
):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 10 or W < 2:
        return 0.0, (0, -1)

    # detrend helps a lot on your synthetic backgrounds
    if detrend_deg is not None:
        t = np.arange(n, dtype=np.float64)
        coef = np.polyfit(t, x, detrend_deg)
        x0 = x - np.polyval(coef, t)
    else:
        x0 = x

    # Use datetime index to avoid version quirks
    tt = pd.date_range("2000-01-01", periods=n, freq="s")
    df = pd.DataFrame({"time": tt, "value": x0})
    ts = TimeSeriesData(df)

    det = CUSUMDetector(ts)
    res = det.detector(alpha=alpha)

    # normalize to list
    items = res if isinstance(res, (list, tuple)) else [res]

    # robust CP extraction
    cp_time = None
    cp_idx = None
    for r in items:
        if r is None:
            continue
        # attribute candidates
        for attr in ("cp", "changepoint", "cp_index", "change_point"):
            if hasattr(r, attr):
                v = getattr(r, attr)
                if isinstance(v, (int, np.integer)):
                    cp_idx = int(v)
                    break
                cp_time = v
        if cp_idx is not None:
            break
        # dict-like candidates
        if isinstance(r, dict):
            for key in ("cp", "changepoint", "cp_index", "change_point"):
                if key in r:
                    v = r[key]
                    if isinstance(v, (int, np.integer)):
                        cp_idx = int(v)
                        break
                    cp_time = v
        if cp_idx is not None:
            break

    # If we got a timestamp, map to index
    if cp_idx is None and cp_time is not None:
        try:
            cp_time = pd.to_datetime(cp_time)
            cp_idx = int(np.argmin(np.abs((tt - cp_time).to_numpy(dtype="timedelta64[ns]").astype(np.int64))))
        except Exception:
            cp_idx = None

    if cp_idx is None:
        return 0.0, (0, -1)

    tau = int(np.clip(cp_idx, 1, n - 2))

    # choose best of left/right windows by mean-shift score
    def mean_shift_score(a, b):
        inside = x0[a:b+1]
        outside = np.concatenate([x0[:a], x0[b+1:]])
        if outside.size == 0:
            return 0.0
        return float(abs(inside.mean() - outside.mean()))

    cand = []
    # right
    a1, b1 = tau, min(n - 1, tau + W - 1)
    if b1 > a1:
        cand.append((a1, b1))
    # left
    a2, b2 = max(0, tau - W + 1), tau
    if b2 > a2:
        cand.append((a2, b2))

    if not cand:
        return 0.0, (0, -1)

    best = max(cand, key=lambda ab: mean_shift_score(*ab))
    score = mean_shift_score(*best)
    return score, best

def plot_sra_sri(
    x: np.ndarray,
    freqs: np.ndarray,
    a: int,
    b: int,
    pred_sra: np.ndarray,
    pred_sri: np.ndarray,
    title: str,
    out_path: str,
):
    plt.figure(figsize=(10, 3))
    plt.plot(freqs, x, color="k", lw=1.0, label="data")
    plt.plot(freqs, pred_sra, color="C0", lw=1.5, label="SRA (global)")
    inside_idxs = np.arange(a, b+1)
    inside_idxs = inside_idxs[:len(pred_sri)]

    plt.plot(freqs[inside_idxs], pred_sri, color="C3", lw=2.0, label="SRI (inside)")


    plt.axvspan(freqs[a], freqs[b], color="C3", alpha=0.15)

    plt.xlabel("frequency")
    plt.ylabel("signal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------------------------------
# Benchmark on dataset
# -------------------------------------------------

def benchmark_dataset(dataset,
                      families: dict,
                      iou_threshold: float = 0.3,
                      max_rows_per_group: int | None = 10,
                      seed: int = 0):
    """
    Run scans for each family on the synthetic dataset.

    Parameters
    ----------
    dataset : dict
        Output of generate_synthetic_dataset().
    families : dict
        Mapping name -> {"family": str, "params": dict}.
    iou_threshold : float
        IoU threshold to count a detection as "hit".
    max_rows_per_group : int or None
        If not None, subsample at most this many rows per group.
    """
    rng = np.random.default_rng(seed)

    results = {
        name: {
            "scores_pos": [],
            "scores_neg": [],
            "coverage": [],
            "parsimony": [],
            "loc_score": [],
            "times": [],
        }
        for name in families.keys()
    }


    for g in dataset["groups"]:
        length = int(g["length"])
        W = int(g["W"])
        X = g["spectra"]                # (n_rows, length)
        has_strong = g["has_strong_anom"]
        strong_labels = g["strong_labels"]

        n_rows, _ = X.shape
        row_indices = np.arange(n_rows)
        # if max_rows_per_group is not None and n_rows > max_rows_per_group:
        #     row_indices = rng.choice(row_indices,
        #                              size=max_rows_per_group,
        #                              replace=False)

        print(f"Group L={length}, W={W}, rows used={len(row_indices)}")

        for row_idx in row_indices:
            x = X[row_idx]
            is_strong = bool(has_strong[row_idx])
            strong_ints = strong_labels[row_idx]

            for name, cfg in families.items():
                mode = cfg.get("mode", "simple")

                t0 = time.time()

                if mode == "simple":
                    fam = cfg["family"]
                    params = cfg.get("params", {})
                    if fam == "krr":
                        params = _resolve_krr_params(params)
                    score, (a, b) = scan_row_with_family(x, W, fam, params)

                    if is_strong and a < b:
                        freqs = np.linspace(0.0, 1.0, x.size)
                        inside = np.arange(a, b + 1, dtype=np.int64)

                        params_plot = params
                        if fam == "krr":
                            params_plot = _resolve_krr_params(params_plot)

                        _, _, pred_sra, K_full, alpha_full, _, _ = rg.build_regressor_sra(x, fam, **params_plot)

                        if fam == "mean":
                            pred_sri = rg.predict_on_idxs_mean(x, inside)
                            out_dir = f"images/SRA_SRI/{fam}"
                            os.makedirs(out_dir, exist_ok=True)

                            plot_sra_sri(
                                x=x,
                                freqs=freqs,
                                a=a,
                                b=b,
                                pred_sra=pred_sra,
                                pred_sri=pred_sri,
                                title=f"{fam} | row={row_idx}",
                                out_path=f"{out_dir}/row_{row_idx}.png",
                            )

                        elif fam == "poly":
                            degree = int(params_plot.get("degree", 1))
                            reg = float(params_plot.get("reg", 1e-8))
                            pre = rg.poly_precompute(x, degree=degree, reg=reg)
                            pred_sri = rg.poly_predict_window(pre, a, b)

                            out_dir = f"images/SRA_SRI/{fam}/{degree}"
                            os.makedirs(out_dir, exist_ok=True)

                            plot_sra_sri(
                                x=x,
                                freqs=freqs,
                                a=a,
                                b=b,
                                pred_sra=pred_sra,
                                pred_sri=pred_sri,
                                title=f"{fam} | row={row_idx}",
                                out_path=f"{out_dir}/row_{row_idx}.png",
                            )

                        elif fam == "krr":
                            pred_sri = rg.predict_on_idxs_krr(
                                x,
                                inside,
                                kernel=params_plot.get("kernel", "rbf"),
                                kernel_param=float(params_plot.get("kernel_param", 1.0)),
                                reg=float(params_plot.get("reg", 1e-3)),
                                K_full=K_full,
                                alpha_full=alpha_full,
                            )

                            kernel = params_plot.get("kernel", "rbf")
                            out_dir = f"images/SRA_SRI/{fam}/{kernel}"
                            os.makedirs(out_dir, exist_ok=True)

                            plot_sra_sri(
                                x=x,
                                freqs=freqs,
                                a=a,
                                b=b,
                                pred_sra=pred_sra,
                                pred_sri=pred_sri,
                                title=f"{fam} | row={row_idx}",
                                out_path=f"{out_dir}/row_{row_idx}.png",
                            )

                        else:
                            pred_sri = None

                elif mode == "nwkr":
                    score, (a, b) = scan_row_with_nwkr_for_benchmark(
                        x,
                        W=W,
                        kernel_kind=cfg.get("kernel_kind", "gaussian"),
                        which=cfg.get("which", "unmasked_varlen"),
                        buffer=int(cfg.get("buffer", 0)),
                        sr_factor=int(cfg.get("sr_factor", 1)),
                    )
                    if is_strong and a < b:
                        freqs = np.linspace(0.0, 1.0, x.size)
                        inside = np.arange(a, b + 1, dtype=np.int64)

                        if cfg["kernel_kind"] == "gaussian":
                            set_kernel_kind("gaussian")
                            Wk, denom = get_kernel_and_denom(x.size, W, KernelKind.GAUSSIAN)
                            _, _, pred_sra, _, _, _ = calculate_gaussian_sra_with_nd(x, Wk, denom)
                            pred_sri = predict_on_idxs(
                                x, inside, Wk, KernelKind.GAUSSIAN, float(W)
                            )
                            out_dir = f"images/SRA_SRI/{mode}/gaussian"
                            os.makedirs(out_dir, exist_ok=True)

                            plot_sra_sri(
                                x=x,
                                freqs=freqs,
                                a=a,
                                b=b,
                                pred_sra=pred_sra,
                                pred_sri=pred_sri,
                                title=f"{mode} | row={row_idx}",
                                out_path=f"{out_dir}/row_{row_idx}.png",
                            )

                        else:
                            set_kernel_kind("laplace")
                            sigma = float(max(W, 1))
                            _, _, pred_sra, _ = calculate_laplace_sra_fast(x, sigma)
                            pred_sri = predict_on_idxs(
                                x, inside, None, KernelKind.LAPLACE, sigma
                            )
                            out_dir = f"images/SRA_SRI/{mode}/laplace"
                            os.makedirs(out_dir, exist_ok=True)

                            plot_sra_sri(
                                x=x,
                                freqs=freqs,
                                a=a,
                                b=b,
                                pred_sra=pred_sra,
                                pred_sri=pred_sri,
                                title=f"{mode} | row={row_idx}",
                                out_path=f"{out_dir}/row_{row_idx}.png",
                            )
                elif mode == "kernelcpd_rbf_c":
                    score, (a, b) = scan_row_with_kernelcpd_rbf_c(
                        x,
                        W=W,
                        min_size=int(cfg.get("min_size", 2)),
                        gamma=cfg.get("gamma", 1e-2),
                    )
                elif mode == "stumpy":
                    score, (a, b) = scan_row_with_stumpy_discord_for_benchmark(
                        x,
                        W=W,
                        m_list=cfg.get("m", None),
                        ignore_trivial=bool(cfg.get("ignore_trivial", True)),
                    )

                    # score, (a, b) = scan_row_with_stumpy_multilen_for_benchmark(
                    #     x,
                    #     W=W,
                    #     m_list=cfg.get("m_list", None),
                    #     z_norm=bool(cfg.get("z_norm", True)),
                    #     ignore_trivial=bool(cfg.get("ignore_trivial", True)),
                    # )

                elif mode == "kats_cusum":
                    score, (a, b) = scan_row_with_kats_cusum_for_benchmark(
                        x,
                        W=W,
                        alpha=float(cfg.get("alpha", 0.01)),
                    )

                else:
                    raise ValueError(f"Unknown mode {mode!r} for family {name!r}")

                dt = time.time() - t0

                if is_strong and a < b:
                    cov_best = 0.0
                    pars_best = 0.0
                    loc_best = 0.0

                    for (s_gt, e_gt) in strong_ints:
                        cov, pars, loc = localization_metrics(
                            pred_ab=(a, b),
                            gt_ab=(s_gt, e_gt),
                        )
                        if loc > loc_best:
                            cov_best = cov
                            pars_best = pars
                            loc_best = loc

                    results[name]["coverage"].append(cov_best)
                    results[name]["parsimony"].append(pars_best)
                    results[name]["loc_score"].append(loc_best)

                    results[name]["scores_pos"].append(score)
                else:
                    results[name]["scores_neg"].append(score)

                results[name]["times"].append(dt)

    # Compute summary metrics
    summary = {}
    for name, r in results.items():
        cov = np.array(r["coverage"])
        pars = np.array(r["parsimony"])
        loc = np.array(r["loc_score"])
        times = np.array(r["times"])

        summary[name] = {
            "mean_coverage": float(cov.mean()) if cov.size else 0.0,
            "mean_parsimony": float(pars.mean()) if pars.size else 0.0,
            "mean_loc_score": float(loc.mean()) if loc.size else 0.0,
            "median_loc_score": float(np.median(loc)) if loc.size else 0.0,
            "mean_time": float(times.mean()) if times.size else 0.0,
            "n_pos": len(r["scores_pos"]),
            "n_neg": len(r["scores_neg"]),
        }


    return results, summary


# -------------------------------------------------
# Plotting: score distributions + simple ROC
# -------------------------------------------------

def plot_benchmark_results(results, summary, out_prefix="bench"):
    """
    Make three figures:
      1) For each family: score histograms (strong vs non-strong).
      2) Localization score distributions (strong rows only).
      3) Runtime comparison (median ± IQR) across families.
    """
    os.makedirs("Images/SyntheticPlots", exist_ok=True)
    families = list(results.keys())

    # -------------------------------------------------
    # 1) Score histograms + Localization Score histograms
    # -------------------------------------------------
    n_fam = len(families)
    fig, axes = plt.subplots(
        n_fam, 2,
        figsize=(12, 3 * n_fam),
        squeeze=False
    )

    for row, name in enumerate(families):
        r = results[name]
        scores_pos = np.array(r["scores_pos"])
        scores_neg = np.array(r["scores_neg"])
        loc_scores = np.array(r["loc_score"])

        # ---- Left: score distributions ----
        ax_scores = axes[row, 0]
        if scores_pos.size:
            ax_scores.hist(
                scores_pos,
                bins=30,
                alpha=0.6,
                label="strong rows",
                density=True,
            )
        if scores_neg.size:
            ax_scores.hist(
                scores_neg,
                bins=30,
                alpha=0.6,
                label="no-strong rows",
                density=True,
            )

        s = summary[name]
        ax_scores.set_title(
            f"{name} | median loc={s['median_loc_score']:.2f}, "
            f"time={s['mean_time']*1000:.1f} ms"
        )
        ax_scores.set_xlabel("Best window score")
        ax_scores.set_ylabel("Density")
        ax_scores.legend(loc="best")

        # ---- Right: localization score distribution ----
        ax_loc = axes[row, 1]
        if loc_scores.size:
            ax_loc.hist(
                loc_scores,
                bins=np.linspace(0.0, 1.0, 21),
                alpha=0.8,
                color="C3",
                edgecolor="k",
            )
            ax_loc.set_xlim(0.0, 1.0)

        ax_loc.set_xlabel("Localization score")
        ax_loc.set_ylabel("Count")
        ax_loc.set_title(f"{name} – localization (strong rows only)")

    fig.tight_layout()
    fig_path1 = f"Images/SyntheticPlots/{out_prefix}_scores_localization.png"
    plt.savefig(fig_path1, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path1}")

    # -------------------------------------------------
    # 2) Runtime comparison across families
    # -------------------------------------------------
    names   = []
    med_ms  = []
    q25_ms  = []
    q75_ms  = []

    for name in families:
        times = np.array(results[name]["times"])
        if times.size == 0:
            continue
        names.append(name)
        med_ms.append(np.median(times) * 1000.0)
        q25_ms.append(np.percentile(times, 25) * 1000.0)
        q75_ms.append(np.percentile(times, 75) * 1000.0)

    if names:
        x = np.arange(len(names))
        med_ms  = np.asarray(med_ms)
        q25_ms  = np.asarray(q25_ms)
        q75_ms  = np.asarray(q75_ms)
        err_low  = med_ms - q25_ms
        err_high = q75_ms - med_ms

        plt.figure(figsize=(8, 4))
        plt.errorbar(
            x,
            med_ms,
            yerr=[err_low, err_high],
            fmt="o-",
            capsize=4,
        )
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel("Runtime per row (ms)")
        plt.title("Runtime comparison across methods (median ± IQR)")
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig_path2 = f"Images/SyntheticPlots/{out_prefix}_runtime.png"
        plt.savefig(fig_path2, dpi=150)
        plt.close()
        print(f"Saved {fig_path2}")
    else:
        print("No runtime data to plot.")


def _warmup_methods(x: np.ndarray, W: int, R:int, families: dict):
    """Warm up compilation for Numba-heavy code paths by running each method once."""
    for name, cfg in families.items():
        mode = cfg.get("mode", "simple")
        try:
            if mode == "simple":
                fam = cfg["family"]
                params = cfg.get("params", {})
                _ = scan_row_with_family(x, W, fam, params)
            elif mode == "nwkr":
                _ = scan_row_with_nwkr_for_benchmark(
                    x,
                    W=W,
                    R=R,
                    kernel_kind=cfg.get("kernel_kind", "gaussian"),
                    which=cfg.get("which", "unmasked_varlen"),
                    buffer=int(cfg.get("buffer", 0)),
                    sr_factor=int(cfg.get("sr_factor", 1)),
                )
            elif mode == "kernelcpd_rbf_c":
                _ = scan_row_with_kernelcpd_rbf_c(
                    x,
                    W=W,
                    min_size=int(cfg.get("min_size", 2)),
                    gamma=cfg.get("gamma", None),
                )
            elif mode == "stumpy":
                _ = scan_row_with_stumpy_discord_for_benchmark(
                    x,
                    W=W,
                    m_list=cfg.get("m", None),
                    ignore_trivial=bool(cfg.get("ignore_trivial", True)),
                )
            elif mode == "kats_cusum":
                _ = scan_row_with_kats_cusum_for_benchmark(
                    x,
                    W=W,
                    alpha=float(cfg.get("alpha", 0.01)),
                )
        except Exception as e:
            print(f"[warmup] {name} failed: {e}")


def runtime_scaling_experiment_one_row(
    *,
    variable: str = "n",
    width_frac: float = 0.10,
    seed: int = 123,
    strong_kind: str = "rect_step",
    strong_rate: float = 1.0,      # for 1 row, make it always strong to avoid branching oddities
    exact_strong: bool = True,
    error_param=(2.5, 0.05),
    families: dict,
    reps: int = 10,                 # repeat each method per length and take median
    warmup: bool = True,
    out_dir: str = "runtime_scaling",
    csv_name: str = "runtimes.csv",
    fig_name: str = "runtimes.png",
):
    """
    Measures per-row runtime vs length for each method.
    Saves a CSV and a log-log plot.
    """
    os.makedirs(out_dir, exist_ok=True)

    if variable == "n":
        array = np.unique(np.round(np.logspace(np.log10(10), np.log10(10000), 10)).astype(int)).tolist()
    elif variable == "w":
        array = np.unique(np.round(np.logspace(np.log10(10), np.log10(500), 10)).astype(int)).tolist()
    elif variable == "r":
        array = np.unique(np.round(np.logspace(np.log10(10), np.log10(500), 10)).astype(int)).tolist()
    rows = []

    for i, elem in enumerate(array):
        if variable == "n":
            n = elem
            w = 10
            r = min(5, n - 1)
        elif variable == "w":
            n = 1000
            w = elem
            r = 500
        elif variable == "r":
            n = 1000
            w = 10
            r = elem

        group = [SpectrumSpec(int(n), 10, 1)]
        data = generate_synthetic_dataset(
            groups=group,
            seed=123 + i,
            strong_rate=strong_rate,
            strong_kind=strong_kind,
            exact_strong=exact_strong,
            error_params=error_param,
        )
        g = data["groups"][0]
        x = np.asarray(g["spectra"][0], dtype=np.float64)

        if warmup:
            _warmup_methods(x, w, r, families)

        for name, cfg in families.items():
            mode = cfg.get("mode", "simple")

            # timing repetitions (median is robust)
            times = []
            last_score = None
            last_ab = None

            for _ in range(reps):
                t0 = time.perf_counter()
                if mode == "simple":
                    fam = cfg["family"]
                    params = cfg.get("params", {})
                    last_score, last_ab = scan_row_with_family(x, w, fam, params)
                elif mode == "nwkr":
                    fam = cfg["family"]
                    if fam == "naive":
                        last_score, last_ab = scan_row_with_nwkr_naive_for_benchmark(
                        x,
                        W=w,
                        R=r,
                        kernel_kind=cfg.get("kernel_kind", "gaussian"),
                        which=cfg.get("which", "unmasked_varlen"),
                        buffer=int(cfg.get("buffer", 0)),
                        sr_factor=int(cfg.get("sr_factor", 1)),
                    )
                    else:
                        last_score, last_ab = scan_row_with_nwkr_for_benchmark(
                            x,
                            W=w,
                            R=r,
                            kernel_kind=cfg.get("kernel_kind", "gaussian"),
                            which=cfg.get("which", "unmasked_varlen"),
                            buffer=int(cfg.get("buffer", 0)),
                            sr_factor=int(cfg.get("sr_factor", 1)),
                        )
                elif mode == "kernelcpd_rbf_c":
                    last_score, last_ab = scan_row_with_kernelcpd_rbf_c(
                        x,
                        W=w,
                        min_size=int(cfg.get("min_size", 2)),
                        gamma=cfg.get("gamma", None),
                    )
                elif mode == "stumpy":
                    last_score, last_ab = scan_row_with_stumpy_discord_for_benchmark(
                        x,
                        W=w,
                        m_list=cfg.get("m", None),
                        ignore_trivial=bool(cfg.get("ignore_trivial", True)),
                    )
                elif mode == "kats_cusum":
                    last_score, last_ab = scan_row_with_kats_cusum_for_benchmark(
                        x,
                        W=w,
                        alpha=float(cfg.get("alpha", 0.01)),
                    )
                else:
                    raise ValueError(f"Unknown mode {mode!r}")

                t1 = time.perf_counter()
                times.append(t1 - t0)

            dt_med = float(np.median(times))
            rows.append(
                {
                    "n": int(n),
                    "w": int(w),
                    "r": int(r),
                    "method": name,
                    "time_s_median": dt_med,
                    "time_ms_median": dt_med * 1000.0,
                    "score": float(last_score) if last_score is not None else np.nan,
                    "a": int(last_ab[0]) if last_ab is not None else -1,
                    "b": int(last_ab[1]) if last_ab is not None else -1,
                }
            )

        print(f"[n={elem:5d}, w={w:4d}, r={r:5d}] done")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot: runtime vs length (log-log)
    plt.figure(figsize=(7, 4))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values(variable)
        plt.plot(sub[variable], sub["time_ms_median"], marker="o", linewidth=1.5, label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"Variable under consideration {variable} (log scale)")
    plt.ylabel("Median runtime per row (ms, log scale)")
    plt.title(f"Runtime scaling experiment for variable {variable}")
    plt.legend(fontsize=7)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, fig_name)
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved: {fig_path}")

    return

# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    # list_of_W = [5, 25, 50, 100, 200]
    # error_params = [(1, 0.05), (2, 0.05), (3, 0.05), (4, 0.05), (5, 0.05), (2.5, 0.01), (2.5, 0.02), (2.5, 0.05), (2.5, 0.10), (2.5, 0.20)]
    error_params = [(2.5, 0.05)]
    
    for parameter in error_params:
        benchmarking_group: List[SpectrumSpec] = [
            SpectrumSpec(500, 50, 100)
        ]
        print(f"Started experiment for parameter={parameter}")
        data = generate_synthetic_dataset(groups=benchmarking_group, seed=123, strong_rate=0.05, strong_kind="rect_step", exact_strong=True, error_params=parameter)

        families = {
            "mean": {
                "family": "mean",
                "params": {},
                "mode": "simple",
            },
            "poly_deg1": {
                "family": "poly",
                "params": {"degree": 1, "reg": 1e-8},
                "mode": "simple",
            },
            "poly_deg2": {
                "family": "poly",
                "params": {"degree": 2, "reg": 1e-8},
                "mode": "simple",
            },
            # "krr_gaussian": {
            #     "family": "krr",
            #     "params": {
            #         "kernel": "rbf",
            #         "kernel_param_scale": 8.0,
            #         "reg": 1e-1,
            #     },
            #     "mode": "simple",
            # },
            # "krr_laplace": {
            #     "family": "krr",
            #     "params": {
            #         "kernel": "laplace",
            #         "kernel_param_scale": 8.0,
            #         "reg": 1e-1,
            #     },
            #     "mode": "simple",
            # },
            # "nwkr_gaussian_naive_varlen": {
            #     "mode": "nwkr",
            #     "family": "naive",
            #     "kernel_kind": "gaussian",
            #     "which": "unmasked_varlen",
            #     "buffer": 0,
            #     "sr_factor": 1,
            # },
            # "nwkr_laplace_naive_varlen": {
            #     "mode": "nwkr",
            #     "family": "naive",
            #     "kernel_kind": "laplace",
            #     "which": "unmasked_varlen",
            #     "buffer": 0,
            #     "sr_factor": 1,
            # },
            "nwkr_gaussian_optimized_varlen": {
                "mode": "nwkr",
                "family": "optimized",
                "kernel_kind": "gaussian",
                "which": "unmasked_varlen",
                "buffer": 0,
                "sr_factor": 1,
            },
            "nwkr_laplace_optimized_varlen": {
                "mode": "nwkr",
                "family": "optimized",
                "kernel_kind": "laplace",
                "which": "unmasked_varlen",
                "buffer": 0,
                "sr_factor": 1,
            },
            # "ruptures_kernelcpd_rbf_c": {
            #     "mode": "kernelcpd_rbf_c",
            #     "min_size": 2,
            #     "gamma": 4e-4,
            #     "n_bkps": 8,
            # },
            # "stumpy_cfg": {
            #     "mode": "stumpy",
            #     "m": None,
            #     "z_norm": True,
            #     "ignore_trivial": True,
            # },
            # "kats_cusum": {
            #     "mode": "kats_cusum",
            #     "alpha": 0.01,
            #     "min_len": 2,
            #     "max_cps": 50,
            # },
            # "ruptures_kernelcpd_rbf_c": {
            #     "mode": "kernelcpd_rbf_c",
            #     "min_size": 2,
            #     "gamma": None,
            #     "standardize": True,
            # },
            # "kats_cusum": {
            #     "mode": "kats_cusum",
            #     "alpha": 0.01,
            # },
            # "stumpy_cfg": {
            #     "mode": "stumpy",
            #     "detrend_deg": 2,
            #     "z_norm": False,
            # },
        }
        variable_list = ["n"]
        for variable in variable_list:
            print(f"Started experiment for variable={variable}")
            runtime_scaling_experiment_one_row(
                variable=variable,
                width_frac=0.10,            # keep W proportional; change to 0.05 for “5% width”
                seed=123,
                strong_kind="rect_step",
                strong_rate=1.0,        # always embed anomaly so the code path is consistent
                exact_strong=True,
                error_param=parameter,
                families=families,
                reps=1,
                warmup=True,
                out_dir="data/runtime_scaling",
                csv_name=f"runtimes_{variable}.csv",
                fig_name=f"runtimes_{variable}.png",
            )

        # results, summary = benchmark_dataset(
        #     data,
        #     families,
        #     seed=0,
        # )

        # for name, param in summary.items():
        #     print(
        #         f"{name}: "
        #         f"mean_loc={param['mean_loc_score']:.3f}, "
        #         f"median_loc={param['median_loc_score']:.3f}, "
        #         f"mean_recall={param['mean_coverage']:.3f}, "
        #         f"mean_precision={param['mean_parsimony']:.3f}, "
        #         f"time={param['mean_time']*1000:.1f} ms, "
        #         f"pos={param['n_pos']}, neg={param['n_neg']}"
        #     )

        # print(f"Experiment for parameter={parameter} finished")
