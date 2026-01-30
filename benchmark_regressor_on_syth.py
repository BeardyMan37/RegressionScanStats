from __future__ import annotations
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from gen_synth_data import generate_synthetic_dataset
import cosmicai.regressors as rg
from cosmicai.config import ref_freq, set_kernel_kind
from cosmicai.scan import scan_row_with_nwkr



# -------------------------------------------------
# Helpers: interval IoU and subset SSE per family
# -------------------------------------------------

def interval_iou(ab, cd) -> float:
    """IoU for two closed integer intervals [a,b], [c,d]."""
    a, b = ab
    c, d = cd
    inter = max(0, min(b, d) - max(a, c) + 1)
    union = (b - a + 1) + (d - c + 1) - inter
    return inter / union if union > 0 else 0.0


def sse_subset(x: np.ndarray, idxs: np.ndarray,
               family: str,
               family_kwargs: dict) -> float:
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
        kernel = family_kwargs.get("kernel", "rbf")
        kernel_param = float(family_kwargs.get("kernel_param", 20.0))
        reg_val = float(family_kwargs.get("reg", 1e-1))
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

def scan_row_krr_two_model(
    x: np.ndarray,
    W: int,
    h0_params: dict,
    h1_family: str = "poly",
    h1_params: dict | None = None,
) -> tuple[float, tuple[int, int]]:
    """
    Scan row using:
      - H0: smooth global KRR (fixed hyperparameters, 'h0_params')
      - H1: same as H0 outside the window, but *local flexible model* (h1_family)
            inside the window.

    Score:
        score(I) = 1 - (SSE_out_H0 + SSE_in_local) / SSE_all_H0

    Parameters
    ----------
    x : np.ndarray
        1D row (length n).
    W : int
        Maximum window length.
    h0_params : dict
        Params for the global KRR under H0, e.g.
        {"kernel": "rbf", "kernel_param": 50.0, "reg": 1e-1}
    h1_family : {"mean","poly","krr"}
        Family for the local model inside the window.
    h1_params : dict
        Params for the local model (degree, reg, kernel_param, etc.)

    Returns
    -------
    best_score : float
    best_ab    : (a, b)
    """
    if h1_params is None:
        h1_params = {}

    n = x.size
    if n == 0:
        return -np.inf, (0, 0)

    # ---- 1) Global *smooth* KRR as H0 ----
    # This gives:
    #   sra_all = SSE_all under H0
    #   pred_all = predictions under H0
    sra_all, ssr_array, pred_all, K_full, alpha_full, ssr_ps = rg.build_regressor_sra(
        x, family="krr", **h0_params
    )
    # Make sure sra_all is the SSE under H0:
    resid_all = x - pred_all
    se_all = resid_all * resid_all
    sse_all = float(se_all.sum())

    # ---- 2) Scan windows, replacing only INSIDE with flexible model ----
    best_score = -np.inf
    best_ab = (0, 0)

    idx_all = np.arange(n, dtype=np.int32)
    max_W = int(min(W, n))

    for m in range(1, max_W + 1):
        for a in range(0, n - m + 1):
            b = a + m - 1
            inside = idx_all[a:b + 1]

            # SSE contribution of H0 inside the window (for reference)
            sse_in_h0 = float(se_all[a:b + 1].sum())
            # SSE outside DOES NOT change under H1: we keep H0 there
            sse_out = sse_all - sse_in_h0

            # SSE of local model inside the window (more flexible)
            sse_in_local = sse_subset(x, inside, h1_family, h1_params)

            sse_h1 = sse_out + sse_in_local

            # LRT-like normalized score (higher is "more anomalous")
            score = 1.0 - (sse_h1 / sse_all)

            if score > best_score:
                best_score = score
                best_ab = (a, b)

    return best_score, best_ab

# -------------------------------------------------
# Naive scan using a regression family
# -------------------------------------------------

def scan_row_with_family(x: np.ndarray,
                         W: int,
                         family: str,
                         family_kwargs: dict) -> tuple[float, tuple[int, int]]:
    """
    Naive O(n^2 W) scan for a single row with a given regression family.

    Returns
    -------
    best_score : float
        Max LLR-style score (SRA_all - (SSE_in + SSE_out)).
    best_ab    : (a, b)
        Best interval (0-based indices, inclusive).
    """
    n = x.size

    # Global SRA (null model) – reuses regressors implementation
    sra_all, ssr_array, pred_all, K, alpha, ssr_ps = rg.build_regressor_sra(
        x, family, **family_kwargs
    )
    # sra_all is SSE under H0. For each I we compute SSE_in + SSE_out under H1.

    best_score = -np.inf
    best_ab = (0, 0)

    idx_all = np.arange(n, dtype=np.int32)

    for m in range(1, W + 1):
        for a in range(0, n - m + 1):
            b = a + m - 1
            inside = idx_all[a:b + 1]
            outside = np.concatenate([idx_all[:a], idx_all[b + 1:]])

            sse_in = sse_subset(x, inside, family, family_kwargs)
            sse_out = sse_subset(x, outside, family, family_kwargs)

            score = (-(sse_in + sse_out) / sra_all) + 1 
            if score > best_score:
                best_score = score
                best_ab = (a, b)

    return best_score, best_ab

def scan_row_with_nwkr_for_benchmark(
    x: np.ndarray,
    W: int,
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

    set_kernel_kind(kernel_kind)  # "gaussian" or "laplace" (matching your config)

    out = scan_row_with_nwkr((0, x, ignore_ranges, freqs, buffer, sr_factor, W, 3 * W, W))

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
            "ious": [],
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
                    score, (a, b) = scan_row_with_family(x, W, fam, params)

                elif mode == "nwkr":
                    score, (a, b) = scan_row_with_nwkr_for_benchmark(
                        x,
                        W=W,
                        kernel_kind=cfg.get("kernel_kind", "gaussian"),
                        which=cfg.get("which", "unmasked_varlen"),
                        buffer=int(cfg.get("buffer", 0)),
                        sr_factor=int(cfg.get("sr_factor", 1)),
                    )


                else:
                    raise ValueError(f"Unknown mode {mode!r} for family {name!r}")

                dt = time.time() - t0

                if is_strong:
                    # best IoU with any strong interval
                    iou_best = 0.0
                    for (s, e) in strong_ints:
                        iou_best = max(iou_best, interval_iou((a, b), (s, e)))
                    results[name]["scores_pos"].append(score)
                    results[name]["ious"].append(iou_best)
                else:
                    results[name]["scores_neg"].append(score)

                results[name]["times"].append(dt)

    # Compute summary metrics
    summary = {}
    for name, r in results.items():
        scores_pos = np.array(r["scores_pos"])
        scores_neg = np.array(r["scores_neg"])
        ious = np.array(r["ious"])
        times = np.array(r["times"])

        hit_rate = float(np.mean(ious >= iou_threshold)) if ious.size else 0.0
        mean_time = float(np.mean(times)) if times.size else 0.0

        summary[name] = {
            "hit_rate": hit_rate,
            "mean_time": mean_time,
            "n_pos": int(scores_pos.size),
            "n_neg": int(scores_neg.size),
        }

    return results, summary


# -------------------------------------------------
# Plotting: score distributions + simple ROC
# -------------------------------------------------

def plot_benchmark_results(results, summary, out_prefix="bench"):
    """
    Make three figures:
      1) For each family: score histograms (pos vs neg) + IoU histogram side-by-side.
      2) ROC-style curves (TPR vs FPR) with approximate AUC.
      3) Runtime comparison (median / spread) across families.
    """
    os.makedirs("Images/SyntheticPlots", exist_ok=True)
    families = list(results.keys())

    # -------------------------------------------------
    # 1) Score histograms + IoU histograms (per family)
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
        ious       = np.array(r["ious"])

        # Left: score distributions (pos vs neg)
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
            f"{name} – hit={s['hit_rate']:.2f}, "
            f"t_row={s['mean_time']*1000:.1f} ms "
            f"(pos={s['n_pos']}, neg={s['n_neg']})"
        )
        ax_scores.set_xlabel("Best window score")
        ax_scores.set_ylabel("Density")
        ax_scores.legend(loc="best")

        # Right: IoU distribution for positive rows
        ax_iou = axes[row, 1]
        if ious.size:
            ax_iou.hist(
                ious,
                bins=np.linspace(0.0, 1.0, 21),
                alpha=0.8,
                color="C2",
                edgecolor="k",
            )
            ax_iou.set_xlim(0.0, 1.0)
        ax_iou.set_xlabel("IoU (detected vs strong interval)")
        ax_iou.set_ylabel("Count")
        ax_iou.set_title(f"{name} – IoU distribution (strong rows only)")

    fig.tight_layout()
    fig_path1 = f"Images/SyntheticPlots/{out_prefix}_scores_iou.png"
    plt.savefig(fig_path1, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path1}")

    # -------------------------------------------------
    # 2) ROC-style curves + AUC (per-family score)
    # -------------------------------------------------
    plt.figure(figsize=(6, 5))
    for name in families:
        r = results[name]
        scores_pos = np.array(r["scores_pos"])
        scores_neg = np.array(r["scores_neg"])
        if scores_pos.size == 0 or scores_neg.size == 0:
            continue

        all_scores = np.concatenate([scores_pos, scores_neg])
        # thresholds sorted descending across the actual score range
        thresh = np.linspace(all_scores.min(), all_scores.max(), 80)

        tpr = []
        fpr = []
        for t in thresh:
            tpr.append(np.mean(scores_pos >= t))
            fpr.append(np.mean(scores_neg >= t))

        tpr = np.asarray(tpr)
        fpr = np.asarray(fpr)

        # Approximate AUC by trapezoidal rule
        # (sort by fpr just in case numerical issues make them slightly non-monotone)
        order = np.argsort(fpr)
        fpr_sorted = fpr[order]
        tpr_sorted = tpr[order]
        auc = float(np.trapz(tpr_sorted, fpr_sorted))

        plt.plot(fpr_sorted, tpr_sorted, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC-style curves (per-family best-window score)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig_path2 = f"Images/SyntheticPlots/{out_prefix}_roc.png"
    plt.savefig(fig_path2, dpi=150)
    plt.close()
    print(f"Saved {fig_path2}")

    # -------------------------------------------------
    # 3) Runtime comparison across families
    # -------------------------------------------------
    # Use median runtime with error bars (25–75% quantiles) in milliseconds.
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
        plt.title("Runtime comparison across regression families\n(median ± IQR)")
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig_path3 = f"Images/SyntheticPlots/{out_prefix}_runtime.png"
        plt.savefig(fig_path3, dpi=150)
        plt.close()
        print(f"Saved {fig_path3}")
    else:
        print("No runtime data to plot.")



# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    data = generate_synthetic_dataset(seed=123, strong_rate=0.05)

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
        "krr_gaussian": {
            "family": "krr",
            "params": {
                "kernel": "rbf",
                "kernel_param_scale": 8.0,
                "reg": 1e-1,
            },
            "mode": "simple",
        },
        "krr_laplace": {
            "family": "krr",
            "params": {
                "kernel": "laplace",
                "kernel_param_scale": 8.0,
                "reg": 1e-1,
            },
            "mode": "simple",
        },
        "nwkr_gaussian_varlen": {
            "mode": "nwkr",
            "kernel_kind": "gaussian",
            "which": "unmasked_varlen",
            "buffer": 0,
            "sr_factor": 1,
        },
        "nwkr_laplace_varlen": {
            "mode": "nwkr",
            "kernel_kind": "laplace",
            "which": "unmasked_varlen",
            "buffer": 0,
            "sr_factor": 1,
        },
    }

    results, summary = benchmark_dataset(
        data,
        families,
        iou_threshold=0.1,
        max_rows_per_group=10,
        seed=0,
    )

    for name, s in summary.items():
        print(
            f"{name}: hit_rate={s['hit_rate']:.3f}, "
            f"mean_time={s['mean_time']*1000:.1f} ms, "
            f"pos={s['n_pos']}, neg={s['n_neg']}"
        )

    plot_benchmark_results(results, summary, out_prefix="bench_regressors")
