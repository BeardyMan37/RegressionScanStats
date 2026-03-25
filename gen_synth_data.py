# gen_syth_data.py
# Drop-in synthetic spectrum generator tuned to resemble your reference plot
# (band-edge rolloff near 1.0, astronomy-like noise, right-edge absorption notch,
# occasional interference spikes + ignore ranges).

from __future__ import annotations

import os
import math

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# Configuration: dataset groups
# ------------------------------

@dataclass
class SpectrumSpec:
    length: int
    W: int            # typical scan-window cap to mimic
    n_rows: int


DEFAULT_GROUPS: List[SpectrumSpec] = [
    SpectrumSpec(500, 50, 1000)
]


# ------------------------------
# Helpers: smooth continuum
# ------------------------------

def _band_envelope(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Bandpass-like envelope in [~0.90,1] with a rise on the left and fall on the right.
    """
    x = np.linspace(0.0, 1.0, n)

    left_scale  = rng.uniform(0.03, 0.08)
    right_scale = rng.uniform(0.03, 0.08)

    left_rise  = 1.0 - np.exp(-x / left_scale)
    right_fall = 1.0 - np.exp(-(1.0 - x) / right_scale)

    env = left_rise * right_fall
    env /= (env.max() + 1e-12)

    floor = rng.uniform(0.90, 0.96)
    return floor + (1.0 - floor) * env


def _lowpass_random_field(n: int, rng: np.random.Generator, sigma_bins: float) -> np.ndarray:
    """
    Smooth random field with correlation length ~ sigma_bins.
    """
    z = rng.normal(0.0, 1.0, size=n).astype(np.float64)

    r = int(min(n // 2 - 1, max(3, round(4.0 * sigma_bins))))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / max(sigma_bins, 1e-6)) ** 2)
    k /= (k.sum() + 1e-12)

    y = np.convolve(z, k, mode="same")
    y -= y.mean()
    y /= (y.std() + 1e-12)
    return y


def _continuum_spectrum_like(n: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """
    Baseline near 1.0 with band-edge rolloff + gentle trend + low-frequency structure.
    """
    x = np.linspace(0.0, 1.0, n)
    # env = _band_envelope(n, rng)

    level = rng.uniform(0.97, 1.01)
    slope = rng.uniform(-0.03, 0.05)
    quad  = rng.uniform(-0.03, 0.03)
    trend = 1.0 + slope * (x - 0.5) + quad * (x - 0.5) ** 2

    sigma_bins = max(6.0, 0.9 * float(W))
    field = _lowpass_random_field(n, rng, sigma_bins=sigma_bins)
    field_amp = rng.uniform(0.002, 0.010)

    y = level * trend * (1.0 + field_amp * field)

    for _ in range(rng.integers(1, 3)):
        k = rng.uniform(0.5, 2.5)
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.0005, 0.0030)
        y += amp * np.sin(2*np.pi*k*x + phase)

    return y.astype(np.float64)


# ------------------------------
# Helpers: astronomy-like noise
# ------------------------------

def _edge_profile(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    d = np.minimum(x, 1.0 - x)
    prof = 1.0 - (d / (0.5 + 1e-12))     # 1 at edges, 0 at center
    return prof**2


def _add_astronomy_noise(
    y: np.ndarray,
    rng: np.random.Generator,
    # Increase these if you want “more white noise”
    white_sigma_frac: Tuple[float, float] = (0.0015, 0.0060),
    corr_sigma_frac: Tuple[float, float]  = (0.0006, 0.0030),
    rho_range: Tuple[float, float]        = (0.70, 0.93),
    white_frac_range: Tuple[float, float] = (0.70, 0.92),
    edge_beta_range: Tuple[float, float]  = (0.2, 0.8),
) -> None:
    """
    y += sqrt(a)*white + sqrt(1-a)*AR(1), with mild edge heteroskedasticity.
    Fractions are relative to median(y) (so scale stays correct when y~1).
    """
    n = y.size
    level = float(np.median(y))
    level = max(level, 1e-6)

    sigma_w0 = level * rng.uniform(*white_sigma_frac)
    sigma_c0 = level * rng.uniform(*corr_sigma_frac)
    rho = rng.uniform(*rho_range)
    a = rng.uniform(*white_frac_range)
    edge_beta = rng.uniform(*edge_beta_range)

    g = _edge_profile(n)
    sigma_w = sigma_w0 * (1.0 + edge_beta * g)
    sigma_c = sigma_c0 * (1.0 + edge_beta * g)

    eps = rng.normal(0.0, 1.0, size=n)

    # AR(1) with ~unit marginal variance
    xi = np.empty(n, dtype=np.float64)
    xi[0] = rng.normal(0.0, 1.0)
    s = np.sqrt(max(1.0 - rho*rho, 1e-12))
    for i in range(1, n):
        xi[i] = rho * xi[i-1] + s * rng.normal(0.0, 1.0)

    y += np.sqrt(a) * (sigma_w * eps) + np.sqrt(1.0 - a) * (sigma_c * xi)


# ------------------------------
# Strong anomaly: right-biased absorption notch (NWKR-friendly)
# ------------------------------

def _add_right_biased_notch(
    y: np.ndarray,
    W: int,
    rng: np.random.Generator,
    depth_frac_range: Tuple[float, float] = (0.015, 0.06),   # dip depth relative to local level
    width_mult_range: Tuple[float, float] = (0.7, 1.3),      # width ~ W
    right_bias_range: Tuple[float, float] = (0.75, 0.93),    # often near right edge
    add_ringing_prob: float = 0.6,
) -> Tuple[int, int]:
    """
    Adds a localized absorption notch. Returns (start,end) support indices.
    """
    n = y.size
    x = np.arange(n, dtype=np.float64)

    width = int(np.clip(rng.uniform(*width_mult_range) * max(W, 3), 3, n//3))
    center = int(rng.uniform(*right_bias_range) * n)
    center = int(np.clip(center, width, n - width - 1))

    s1 = max(1.0, 0.18 * width)           # narrow
    s2 = max(s1 + 1.0, 0.55 * width)      # broad

    g1 = np.exp(-0.5 * ((x - center) / s1) ** 2)
    g2 = np.exp(-0.5 * ((x - center) / s2) ** 2)

    prof = -(g1 - 0.35 * g2)              # sharp dip with shoulders
    prof /= (np.max(np.abs(prof)) + 1e-12)

    local = float(np.median(y[max(0, center-width):min(n, center+width)]))
    depth = rng.uniform(*depth_frac_range) * max(local, 1e-6)

    y += depth * prof

    if rng.random() < add_ringing_prob:
        t = (x - center) / max(width, 1)
        env = np.exp(-0.5 * (t / 0.35) ** 2)
        cycles = rng.uniform(4.0, 10.0)
        phase  = rng.uniform(0.0, 2*np.pi)
        ring = env * np.sin(2*np.pi*cycles*t + phase)
        y += (0.25 * depth) * ring

    mask = np.abs(prof) > 0.2
    idx = np.where(mask)[0]
    if idx.size == 0:
        return max(0, center-1), min(n-1, center+1)
    return int(idx[0]), int(idx[-1])


# ------------------------------
# Weak spectral lines (small)
# ------------------------------

def _add_weak_line(
    y: np.ndarray,
    rng: np.random.Generator,
    kind: str = "absorption",
    amp_frac_range: Tuple[float, float] = (0.003, 0.012),
    width_frac_range: Tuple[float, float] = (0.003, 0.010),
) -> Tuple[int, int]:
    n = y.size
    ii = np.arange(n, dtype=np.float64)

    center = rng.uniform(0.15, 0.85)
    width_frac = rng.uniform(*width_frac_range)
    sigma = max(1e-6, (width_frac / 2.355) * n)
    cc = center * (n - 1)

    prof = np.exp(-0.5 * ((ii - cc) / sigma) ** 2)

    local = float(np.median(y[max(0, int(cc-30)):min(n, int(cc+30))]))
    amp = rng.uniform(*amp_frac_range) * max(local, 1e-6)

    if kind == "absorption":
        y -= amp * prof
    else:
        y += amp * prof

    mask = prof > 0.2
    idx = np.where(mask)[0]
    if idx.size == 0:
        return int(cc), int(cc)
    return int(idx[0]), int(idx[-1])


# ------------------------------
# Interference spikes (ignored)
# ------------------------------

def _add_interference_spike(
    y: np.ndarray,
    rng: np.random.Generator,
    amp_mult_range: Tuple[float, float] = (3.0, 7.0),
    left_region: Tuple[float, float] = (0.06, 0.18),
) -> Tuple[int, int]:
    """
    Add a narrow positive spike (RFI-like). Returns a small interval to ignore.
    """
    n = y.size
    k = int(rng.uniform(*left_region) * n)
    k = int(np.clip(k, 0, n-1))

    local = float(np.median(y[max(0, k-20):min(n, k+20)]))
    spike = rng.uniform(*amp_mult_range) * max(local, 1e-6)

    y[k] += spike

    s = max(0, k-2)
    e = min(n-1, k+2)
    return s, e

def _add_gaussian_feature(
    y: np.ndarray,
    W: int,
    rng: np.random.Generator,
    kind: str,  # "absorption" or "emission"
    amp_frac_range: Tuple[float, float] = (0.03, 0.10),
    width_mult_range: Tuple[float, float] = (0.7, 1.3),
    center_range: Tuple[float, float] = (0.10, 0.90),
) -> Tuple[int, int]:
    """
    Gaussian absorption/emission feature with width tied to W.
    Returns (start,end) of the labeled support where profile > 5% of peak.
    """
    n = y.size
    ii = np.arange(n, dtype=np.float64)

    width = int(np.clip(rng.uniform(*width_mult_range) * max(W, 3), 3, n // 3))
    center = int(np.clip(rng.uniform(*center_range) * (n - 1), width, n - width - 1))

    sigma = max(1.0, 0.25 * width)
    prof = np.exp(-0.5 * ((ii - center) / sigma) ** 2)

    local = float(np.median(y[max(0, center - width):min(n, center + width)]))
    amp = rng.uniform(*amp_frac_range) * max(local, 1e-6)

    if kind == "absorption":
        y -= amp * prof
    elif kind == "emission":
        y += amp * prof
    else:
        raise ValueError(f"Unknown kind={kind!r}")

    mask = prof > 0.05  # label main mass, exclude tails
    idx = np.where(mask)[0]
    if idx.size == 0:
        return center, center
    return int(idx[0]), int(idx[-1])


def _add_rect_step(
    y: np.ndarray,
    W: int,
    rng: np.random.Generator,
    strength: float = 5.0,          # now interpreted as "k sigma"
    width_frac: float = 0.2,
    center_range: Tuple[float, float] = (0.10, 0.90),
) -> Tuple[int, int]:
    """
    Rectangular step anomaly whose magnitude is k × local noise std.

    strength = k (number of standard deviations)
    """

    n = y.size
    width = max(1, int(width_frac * n))

    start = int(rng.integers(int(center_range[0] * n),
                             int(center_range[1] * n - width)))
    end = start + width - 1

    seg = y[start:end + 1].copy()

    # Estimate local noise level via robust std
    local_std = float(np.std(y))

    # If region is extremely smooth, avoid zero anomaly
    local_std = max(local_std, 1e-8)

    offset = strength * local_std

    if rng.random() < 0.5:
        y[start:end + 1] = seg + offset
    else:
        y[start:end + 1] = seg - offset

    return start, end


def _allocate_exact_strong_counts(groups: List[SpectrumSpec], strong_rate: float) -> List[int]:
    """
    Allocate an exact total number of strong rows across groups.
    Uses floor per group + distributes remainder by fractional parts.
    """
    total_rows = int(sum(int(g.n_rows) for g in groups))
    target_total = int(round(float(strong_rate) * total_rows))

    exp = [float(strong_rate) * int(g.n_rows) for g in groups]
    base = [int(math.floor(e)) for e in exp]
    frac = [e - b for e, b in zip(exp, base)]

    remaining = target_total - sum(base)
    if remaining > 0:
        order = np.argsort(frac)[::-1]
        for t in range(remaining):
            base[int(order[t % len(base)])] += 1
    elif remaining < 0:
        order = np.argsort(frac)  # remove from smallest frac first
        for t in range(-remaining):
            j = int(order[t % len(base)])
            base[j] = max(0, base[j] - 1)

    # safety clamp
    for i, g in enumerate(groups):
        base[i] = int(np.clip(base[i], 0, int(g.n_rows)))

    return base


# ------------------------------
# Main: single spectrum
# ------------------------------

def generate_single_spectrum(
    length: int,
    W: int,
    rng: np.random.Generator,
    p_strong_anom: float = 0.05,
    p_weak_line: float = 0.25,
    # p_interference: float = 0.08,
    # buffer_frac: float = 0.03,
    strong_kind: str = "right_notch",      # "gaussian_absorption" | "rect_step" | "gaussian_emission" | "right_notch"
    force_strong: bool | None = None,      # if set, overrides Bernoulli draw
    error_params: Tuple[float, float] | None = (2.5, 0.05),
) -> Dict[str, Any]:

    """
    Returns dict:
      y, has_strong_anom, strong_intervals, weak_intervals, ignore_ranges
    """
    y = _continuum_spectrum_like(length, W, rng)
    _add_astronomy_noise(y, rng)

    strong_intervals: List[Tuple[int, int]] = []
    # weak_intervals: List[Tuple[int, int]] = []
    # ignore_ranges: List[Tuple[int, int]] = []

    # band-edge buffer ignore (matches gray regions in your plots)
    # buf = int(round(buffer_frac * length))
    # if buf > 0:
    #     ignore_ranges.append((0, buf-1))
    #     ignore_ranges.append((length-buf, length-1))

    # strong anomaly: right-edge absorption notch
    has_strong = (rng.random() < p_strong_anom) if (force_strong is None) else bool(force_strong)

    if has_strong:
        k = strong_kind.lower()
        if k in ("right_notch", "notch"):
            s, e = _add_right_biased_notch(y, W, rng)
        elif k in ("gaussian_absorption", "gauss_abs", "abs"):
            s, e = _add_gaussian_feature(y, W, rng, kind="absorption")
        elif k in ("gaussian_emission", "gauss_em", "em"):
            s, e = _add_gaussian_feature(y, W, rng, kind="emission")
        elif k in ("rect_step", "step"):
            s, e = _add_rect_step(y, W, rng, strength=error_params[0], width_frac=error_params[1])
        else:
            raise ValueError(f"Unknown strong_kind={strong_kind!r}")

        strong_intervals.append((s, e))


    # weak lines: subtle
    # if rng.random() < p_weak_line:
    #     for _ in range(rng.integers(1, 3)):
    #         kind = "absorption" if rng.random() < 0.75 else "emission"
    #         s, e = _add_weak_line(y, rng, kind=kind)
    #         weak_intervals.append((s, e))

    # interference spikes: mostly on non-strong rows; treat as ignored region
    # if (not has_strong) and (rng.random() < p_interference):
    #     s_i, e_i = _add_interference_spike(y, rng)
    #     ignore_ranges.append((s_i, e_i))

    return {
        "y": y.astype(np.float64),
        "has_strong_anom": bool(has_strong),
        "strong_intervals": strong_intervals,
        # "weak_intervals": weak_intervals,
        # "ignore_ranges": ignore_ranges,
    }


# ------------------------------
# Dataset generator (API-compatible)
# ------------------------------

def generate_synthetic_dataset(
    groups: List[SpectrumSpec] | None = None,
    seed: int = 123,
    strong_rate: float = 0.05,
    strong_kind: str = "right_notch",
    exact_strong: bool = True,
    error_params: Tuple[float, float] | None = (2.5, 0.05), 
) -> Dict[str, Any]:

    """
    Returns {"groups": [ group_dict, ... ]}

    Each group_dict:
      {
        "group_id": int,
        "length": int,
        "W": int,
        "spectra": (n_rows, length) float32,
        "has_strong_anom": (n_rows,) bool,
        "strong_labels": List[List[(s,e)]],
        "weak_labels":   List[List[(s,e)]],
        "ignore_ranges": List[List[(s,e)]],   # NEW (safe to ignore in older benchmarks)
      }
    """
    if groups is None:
        groups = DEFAULT_GROUPS

    rng = np.random.default_rng(seed)
    strong_counts = _allocate_exact_strong_counts(groups, strong_rate) if exact_strong else None
    all_groups: List[Dict[str, Any]] = []

    for gid, spec in enumerate(groups):
        length = spec.length
        n_rows = spec.n_rows

        spectra = np.empty((n_rows, length), dtype=np.float64)
        strong_flags = np.zeros(n_rows, dtype=bool)

        strong_labels: List[List[Tuple[int, int]]] = []
        # weak_labels: List[List[Tuple[int, int]]] = []
        # ignore_labels: List[List[Tuple[int, int]]] = []

        strong_set = set()
        if exact_strong:
            k = int(strong_counts[gid])
            if k > 0:
                strong_set = set(rng.choice(np.arange(n_rows), size=k, replace=False).tolist())


        for r in range(n_rows):
            row = generate_single_spectrum(
                length=length,
                W=spec.W,
                rng=rng,
                p_strong_anom=strong_rate,
                p_weak_line=0.25,
                # p_interference=0.08,
                # buffer_frac=0.03,
                strong_kind=strong_kind,
                force_strong=(r in strong_set) if exact_strong else None,
                error_params=error_params,
            )

            spectra[r, :] = row["y"]
            strong_flags[r] = row["has_strong_anom"]
            strong_labels.append(row["strong_intervals"])
            # weak_labels.append(row["weak_intervals"])
            # ignore_labels.append(row["ignore_ranges"])

        all_groups.append({
            "group_id": gid,
            "length": length,
            "W": spec.W,
            "spectra": spectra,
            "has_strong_anom": strong_flags,
            "strong_labels": strong_labels,
            # "weak_labels": weak_labels,
            # "ignore_ranges": ignore_labels,
        })

    return {"groups": all_groups}


# ------------------------------
# Plotting utility (optional)
# ------------------------------

def plot_synthetic_groups_from_dataset(
    dataset: Dict[str, Any],
    out_root: str = "Images/SyntheticPlots",
    n_no_anom: int = 5,
    n_with_anom: int = 5,
    seed: int = 0,
) -> None:
    """
    Plot example spectra per group (no-strong vs strong) including:
      - ignore ranges (gray)
      - weak lines (green)
      - strong anomalies (orange)
    """
    os.makedirs(out_root, exist_ok=True)
    rng = np.random.default_rng(seed)

    for group in dataset["groups"]:
        L = int(group["length"])
        W = int(group["W"])
        specs = group["spectra"]
        has_strong = group["has_strong_anom"]
        strong_labels = group["strong_labels"]
        # weak_labels = group["weak_labels"]
        # ignore_ranges = group.get("ignore_ranges", None)

        n_rows, n_ch = specs.shape
        freqs = np.linspace(0.0, 1.0, n_ch)

        idx_with = np.where(has_strong)[0]
        idx_without = np.where(~has_strong)[0]
        # if len(idx_with) == 0 or len(idx_without) == 0:
        #     continue

        if len(idx_with) > n_with_anom:
            idx_with = rng.choice(idx_with, size=n_with_anom, replace=False)
        if len(idx_without) > n_no_anom:
            idx_without = rng.choice(idx_without, size=n_no_anom, replace=False)

        max_rows = min(max(len(idx_without), len(idx_with)), max(n_no_anom, n_with_anom))

        fig, axes = plt.subplots(
            max_rows, 2, figsize=(12, 2.5 * max_rows),
            squeeze=False, sharex=False, sharey=False
        )

        def _shade(ax, ranges, color, alpha, label=None):
            for (s, e) in ranges:
                if 0 <= s < n_ch and 0 <= e < n_ch and e >= s:
                    ax.axvspan(float(freqs[s]), float(freqs[e]), color=color, alpha=alpha, label=label)
                    label = None

        for row_i in range(max_rows):
            # left col: no-strong
            if row_i < len(idx_without):
                r = int(idx_without[row_i])
                ax = axes[row_i, 0]
                x = specs[r]

                ax.plot(freqs, x, linewidth=1.0)

                # if ignore_ranges is not None:
                #     _shade(ax, ignore_ranges[r], color="0.7", alpha=0.25, label="ignore" if row_i == 0 else None)
                # _shade(ax, weak_labels[r], color="C2", alpha=0.20, label="weak" if row_i == 0 else None)

                ax.set_title(f"L={L}, W={W}, row={r} (no strong anomaly)")
                ax.set_ylabel("Amplitude")
            else:
                axes[row_i, 0].axis("off")

            # right col: strong
            if row_i < len(idx_with):
                r = int(idx_with[row_i])
                ax = axes[row_i, 1]
                x = specs[r]

                ax.plot(freqs, x, linewidth=1.0)

                # if ignore_ranges is not None:
                #     _shade(ax, ignore_ranges[r], color="0.7", alpha=0.25, label="ignore" if row_i == 0 else None)
                # _shade(ax, weak_labels[r], color="C2", alpha=0.20, label="weak" if row_i == 0 else None)
                _shade(ax, strong_labels[r], color="C1", alpha=0.35, label="strong" if row_i == 0 else None)

                ax.set_title(f"L={L}, W={W}, row={r} (strong anomaly)")
                ax.set_ylabel("Amplitude")
            else:
                axes[row_i, 1].axis("off")

        for ax in axes[-1, :]:
            ax.set_xlabel("Frequency (arb. units)")

        handles, labels = axes[0, 1].get_legend_handles_labels()
        if handles:
            axes[0, 1].legend(loc="best", fontsize=8)

        fig.suptitle(f"Synthetic spectra for length L={L}, W={W}", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = os.path.join(out_root, f"synthetic_L{L}_W{W}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


# ------------------------------
# CLI entry point
# ------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-npy", type=str, default="Data/synthetic_spectra.npz",
                        help="Path to save npz with groups/spectra.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--strong-rate", type=float, default=0.05,
                        help="Approx. fraction of rows with strong anomalies.")
    args = parser.parse_args()

    # data_abs  = generate_synthetic_dataset(seed=123, strong_rate=0.05, strong_kind="gaussian_absorption", exact_strong=True)
    # data_em   = generate_synthetic_dataset(seed=123, strong_rate=0.05, strong_kind="gaussian_emission",  exact_strong=True)
    data_step = generate_synthetic_dataset(seed=123, strong_rate=0.05, strong_kind="rect_step", exact_strong=True)
    # plot_synthetic_groups_from_dataset(data_abs, out_root="Images/SyntheticPlots/abs", n_no_anom=5, n_with_anom=5)
    # plot_synthetic_groups_from_dataset(data_em, out_root="Images/SyntheticPlots/em", n_no_anom=5, n_with_anom=5)
    plot_synthetic_groups_from_dataset(data_step, out_root="Images/SyntheticPlots/step", n_no_anom=5, n_with_anom=5)

    # out: Dict[str, np.ndarray] = {}
    # for g in data["groups"]:
    #     gid = g["group_id"]
    #     prefix = f"g{gid}"
    #     out[f"{prefix}_length"] = np.array([g["length"]], dtype=np.int32)
    #     out[f"{prefix}_W"] = np.array([g["W"]], dtype=np.int32)
    #     out[f"{prefix}_spectra"] = g["spectra"]
    #     out[f"{prefix}_has_strong"] = g["has_strong_anom"].astype(np.int8)
    #     out[f"{prefix}_strong_labels"] = np.array(g["strong_labels"], dtype=object)
    #     out[f"{prefix}_weak_labels"] = np.array(g["weak_labels"], dtype=object)
    #     out[f"{prefix}_ignore_ranges"] = np.array(g["ignore_ranges"], dtype=object)


    # os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    # np.savez(args.out_npy, **out)
    # print(f"Wrote {args.out_npy}")
