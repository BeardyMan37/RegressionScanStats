"""
ar2_spectrum_model.py
=====================
Pure AR(2) + white noise spectrum model replicating gen_synth_data.py.

Signal structure:
    x[t] = level * (1 + ar2_amp * z[t]) + noise[t] + step[t]

where z[t] is a stationary AR(2) process, noise[t] is white Gaussian,
and step[t] is an optional rectangular anomaly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters  (all ranges match gen_synth_data.py exactly)
# ---------------------------------------------------------------------------

@dataclass
class AR2SpectrumParams:
    # AR(2) process
    phi1_range:      Tuple[float, float] = (1.980, 1.990)
    phi2_range:      Tuple[float, float] = (-0.990, -0.980)
    ar2_amp_range:   Tuple[float, float] = (0.002, 0.010)
    ar2_burnin:      int                 = 200

    # Signal level
    level_range:     Tuple[float, float] = (0.97,  1.01)

    # White noise (fraction of signal level, with edge heteroskedasticity)
    sigma_w_range:   Tuple[float, float] = (0.05, 0.1)
    edge_beta_range: Tuple[float, float] = (0.2,    0.8)

    # Rectangular step anomaly
    step_strength:   float               = 2.5    # k * sigma (SNR)
    step_width_frac: float               = 0.2    # fraction of n
    step_center_range: Tuple[float, float] = (0.10, 0.90)


# ---------------------------------------------------------------------------
# AR(2) process
# ---------------------------------------------------------------------------

def _ar2_process(
    n: int,
    rng: np.random.Generator,
    phi1_range: Tuple[float, float] = (1.980, 1.990),
    phi2_range: Tuple[float, float] = (-0.990, -0.980),
    burnin: int = 40,
) -> np.ndarray:
    """
    Generate a very smooth AR(2) process with only a few broad turns
    and scale it to lie in [-1, 1].

    This is intentionally not a typical stationary-noise-looking AR(2):
    it is biased toward near-unit positive roots so the series has
    long broad moves and only a small number of turning points.
    """
    phi1 = 1.985
    phi2 = -0.985

    # Sample only from a very persistent, non-oscillatory regime.
    for _ in range(128):
        p1 = rng.uniform(*phi1_range)
        p2 = rng.uniform(*phi2_range)

        disc = p1 * p1 + 4.0 * p2
        if disc < 0.0:
            continue

        s = np.sqrt(disc)
        r1 = 0.5 * (p1 + s)
        r2 = 0.5 * (p1 - s)

        # Real, positive, very close to 1 -> broad slow motion
        if 0.975 < r2 < r1 < 0.997:
            phi1 = p1
            phi2 = p2
            break

    m = n + burnin
    x = np.zeros(m, dtype=np.float64)

    # Tiny ongoing innovations
    eps = rng.normal(0.0, 0.004, size=m)

    # Let the initial condition shape the path
    x[0] = rng.normal(0.0, 1.0)
    if m > 1:
        x[1] = x[0] + rng.normal(0.0, 0.06)

    for t in range(2, m):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + eps[t]

    x = x[burnin:]
    x -= x.mean()
    x /= (np.max(np.abs(x)) + 1e-12)

    if rng.random() < 0.5:
        x *= -1.0

    return x, phi1, phi2


# ---------------------------------------------------------------------------
# Edge heteroskedasticity profile  (matches gen_synth_data._edge_profile)
# ---------------------------------------------------------------------------

def _edge_profile(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    d = np.minimum(x, 1.0 - x)
    return (1.0 - d / 0.5) ** 2


# ---------------------------------------------------------------------------
# Single spectrum
# ---------------------------------------------------------------------------

def generate_ar2_spectrum(
    n: int,
    W: int,
    rng: np.random.Generator,
    params: AR2SpectrumParams = AR2SpectrumParams(),
    force_step: Optional[bool] = None,
    step_strength: Optional[float] = None,
    step_width_frac: Optional[float] = None,
) -> dict:
    """
    Generate one spectrum.

    Returns
    -------
    dict with keys:
        y               : (n,) signal
        has_anomaly     : bool
        step_interval   : (s, e) inclusive, or None
        phi1, phi2      : AR(2) coefficients used
        level           : signal level
        sigma_w         : noise std at centre
    """
    p = params

    # 1. AR(2) baseline
    z, phi1, phi2 = _ar2_process(n, rng, p.phi1_range, p.phi2_range, p.ar2_burnin)
    level  = max(float(np.median(np.abs(z))), 1e-6)
    baseline = level * (1.0 + z)

    # 2. White noise with edge heteroskedasticity
    sigma_w0   = level * rng.uniform(*p.sigma_w_range)
    edge_beta  = rng.uniform(*p.edge_beta_range)
    sigma_w    = sigma_w0 * (1.0 + edge_beta * _edge_profile(n))
    noise      = sigma_w * rng.normal(0.0, 1.0, size=n)

    y = baseline + noise

    # 3. Rectangular step anomaly
    k    = step_strength   if step_strength   is not None else p.step_strength
    wf   = step_width_frac if step_width_frac is not None else p.step_width_frac

    has_anomaly = bool(force_step) if force_step is not None else False
    step_interval = None

    if has_anomaly:
        width  = max(1, int(wf * n))
        lo     = int(p.step_center_range[0] * n)
        hi     = max(lo + 1, int(p.step_center_range[1] * n) - width)
        start  = int(rng.integers(lo, hi))
        end    = start + width - 1

        local_std = float(np.std(y))
        local_std = max(local_std, 1e-8)
        offset    = k * local_std

        if rng.random() < 0.5:
            y[start:end+1] += offset
        else:
            y[start:end+1] -= offset

        step_interval = (start, end)

    return {
        "y":             y.astype(np.float64),
        "has_anomaly":   has_anomaly,
        "step_interval": step_interval,
        "phi1":          phi1,
        "phi2":          phi2,
        "level":         level,
        "sigma_w":       float(sigma_w0),
    }


# ---------------------------------------------------------------------------
# Dataset generator  (matches generate_synthetic_dataset API)
# ---------------------------------------------------------------------------

def generate_ar2_dataset(
    n: int = 500,
    W: int = 50,
    n_rows: int = 100,
    strong_rate: float = 0.05,
    seed: int = 123,
    params: AR2SpectrumParams = AR2SpectrumParams(),
    step_strength: Optional[float] = None,
    step_width_frac: Optional[float] = None,
) -> dict:
    """
    Generate a dataset of spectra.

    Returns dict matching generate_synthetic_dataset output:
        spectra         : (n_rows, n) float64
        has_strong_anom : (n_rows,) bool
        strong_labels   : list of list of (s, e)
        phi1, phi2      : arrays of AR(2) coefficients per row
    """
    rng = np.random.default_rng(seed)

    # Exact count of strong rows
    n_strong   = int(round(strong_rate * n_rows))
    strong_set = set(rng.choice(n_rows, size=n_strong, replace=False).tolist())

    spectra       = np.empty((n_rows, n), dtype=np.float64)
    has_strong    = np.zeros(n_rows, dtype=bool)
    strong_labels = []
    phi1s         = np.empty(n_rows)
    phi2s         = np.empty(n_rows)

    for r in range(n_rows):
        row = generate_ar2_spectrum(
            n=n, W=W, rng=rng, params=params,
            force_step=(r in strong_set),
            step_strength=step_strength,
            step_width_frac=step_width_frac,
        )
        spectra[r]   = row["y"]
        has_strong[r] = row["has_anomaly"]
        strong_labels.append([row["step_interval"]] if row["step_interval"] else [])
        phi1s[r] = row["phi1"]
        phi2s[r] = row["phi2"]

    return {
        "spectra":       spectra,
        "has_strong_anom": has_strong,
        "strong_labels": strong_labels,
        "phi1":          phi1s,
        "phi2":          phi2s,
        "n": n, "W": W,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ar2_dataset(
    dataset: dict,
    n_no_anom: int = 5,
    n_with_anom: int = 5,
    out_path: str = "data/images/synthetic_plots/ar2_spectra.png",
    seed: int = 0,
) -> None:
    rng      = np.random.default_rng(seed)
    spectra  = dataset["spectra"]
    has_str  = dataset["has_strong_anom"]
    labels   = dataset["strong_labels"]
    n_ch     = spectra.shape[1]
    freqs    = np.linspace(0.0, 1.0, n_ch)

    idx_with    = np.where(has_str)[0]
    idx_without = np.where(~has_str)[0]

    if len(idx_with)    > n_with_anom: idx_with    = rng.choice(idx_with,    n_with_anom, replace=False)
    if len(idx_without) > n_no_anom:   idx_without = rng.choice(idx_without, n_no_anom,   replace=False)

    n_rows = max(len(idx_with), len(idx_without))
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.5 * n_rows), squeeze=False)

    for ri in range(n_rows):
        # Left: no anomaly
        if ri < len(idx_without):
            r  = int(idx_without[ri])
            ax = axes[ri, 0]
            ax.plot(freqs, spectra[r], lw=0.8, color="C0")
            ax.set_title(f"row={r}  no anomaly  φ₁={dataset['phi1'][r]:.2f} φ₂={dataset['phi2'][r]:.2f}", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=7)
        else:
            axes[ri, 0].axis("off")

        # Right: with anomaly
        if ri < len(idx_with):
            r  = int(idx_with[ri])
            ax = axes[ri, 1]
            ax.plot(freqs, spectra[r], lw=0.8, color="C0")
            for (s, e) in labels[r]:
                ax.axvspan(freqs[s], freqs[e], color="C1", alpha=0.35, label="step")
            ax.set_title(f"row={r}  STEP ANOMALY  φ₁={dataset['phi1'][r]:.2f} φ₂={dataset['phi2'][r]:.2f}", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=7)
            if ri == 0:
                ax.legend(fontsize=7, loc="upper left")
        else:
            axes[ri, 1].axis("off")

    for ax in axes[-1]:
        ax.set_xlabel("Frequency (arb.)", fontsize=8)

    fig.suptitle(
        f"AR(2) synthetic spectra  (n={dataset['n']}, W={dataset['W']}, "
        f"SNR={AR2SpectrumParams().step_strength}σ, "
        f"step_frac={AR2SpectrumParams().step_width_frac})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = AR2SpectrumParams(
        step_strength=2.5,
        step_width_frac=0.2,
    )
    data = generate_ar2_dataset(
        n=500, W=50, n_rows=200,
        strong_rate=0.15,
        seed=123,
        params=params,
    )
    plot_ar2_dataset(data, n_no_anom=5, n_with_anom=5,
                     out_path="images/synthetic_plots/ar2_spectra.png")
    print(f"Strong rows: {data['has_strong_anom'].sum()} / {len(data['has_strong_anom'])}")
    print(f"phi1 range: [{data['phi1'].min():.2f}, {data['phi1'].max():.2f}]")
    print(f"phi2 range: [{data['phi2'].min():.2f}, {data['phi2'].max():.2f}]")


def plot_parameter_comparison(
    out_path: str = "images/synthetic_plots/ar2_params.png",
) -> None:
    """Show how step_strength and noise level affect the signal."""
    strengths  = [1.0, 2.5, 5.0]
    noise_mults = [0.5, 1.0, 2.0]   # multiplier on sigma_w_range

    fig, axes = plt.subplots(
        len(noise_mults), len(strengths),
        figsize=(13, 8), squeeze=False, sharey=False,
    )
    rng = np.random.default_rng(42)

    for ri, nm in enumerate(noise_mults):
        for ci, k in enumerate(strengths):
            p = AR2SpectrumParams(
                sigma_w_range=(0.0015 * nm, 0.006 * nm),
                step_strength=k,
                step_width_frac=0.2,
            )
            row = generate_ar2_spectrum(500, 50, rng, p, force_step=True)
            y   = row["y"]
            s, e = row["step_interval"]
            freqs = np.linspace(0.0, 1.0, 500)
            ax = axes[ri][ci]
            ax.plot(freqs, y, lw=0.8, color="C0")
            ax.axvspan(freqs[s], freqs[e], color="C1", alpha=0.35)
            ax.set_title(f"SNR={k}σ,  noise×{nm}", fontsize=8)
            if ci == 0: ax.set_ylabel("Amplitude", fontsize=7)
            if ri == len(noise_mults)-1: ax.set_xlabel("Frequency", fontsize=7)

    fig.suptitle("AR(2) model: effect of SNR (columns) and noise level (rows)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
