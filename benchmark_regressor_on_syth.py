from __future__ import annotations
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from cosmicai.gaussian_state import gaussian_state_init
from cosmicai.kernels import get_kernel_and_denom
from cosmicai.laplace_fast import calculate_laplace_sra_fast
from cosmicai.predictors import predict_on_idxs
from cosmicai.scoring import calculate_gaussian_sra_with_nd, laplace_sri, laplace_sro_nearband
from gen_synth_data import generate_synthetic_dataset
import cosmicai.regressors as rg
from cosmicai.config import KernelKind, ref_freq, set_kernel_kind
from cosmicai.scan import scan_row_with_nwkr

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

    coverage = inter / gt_len
    parsimony = inter / pred_len

    loc_score = (coverage * parsimony) ** 0.5

    return coverage, parsimony, loc_score



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
        reg_val = float(family_kwargs.get("reg", 1e-1))
        preds = rg.predict_on_idxs_krr(
            x,
            idxs,
            kernel=kernel,
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
    flags = []

    set_kernel_kind(kernel_kind)  # "gaussian" or "laplace" (matching your config)

    out = scan_row_with_nwkr((0, x, ignore_ranges, flags, freqs, buffer, sr_factor, W // 2, W, W // 2))

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
                    score, (a, b) = scan_row_with_family(x, W, fam, params)

                    if is_strong and a < b:
                        freqs = np.linspace(0.0, 1.0, x.size)
                        inside = np.arange(a, b + 1, dtype=np.int64)

                        # ---- SRA (global) ----
                        _, _, pred_sra, _, _, _ = rg.build_regressor_sra(x, fam, **params)

                        # ---- SRI (inside only) ----
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
                            pred_sri = rg.predict_on_idxs_poly(x, inside, **params)
                            degree = params["degree"]
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
                            pred_sri = rg.predict_on_idxs_krr(x, inside, **params)
                            kernel = params["kernel"]
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

                        else:  # Laplace
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




# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    data = generate_synthetic_dataset(seed=123, strong_rate=0.05, strong_kind="rect_step")

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
        seed=0,
    )

    for name, s in summary.items():
        print(
            f"{name}: "
            f"mean_loc={s['mean_loc_score']:.3f}, "
            f"median_loc={s['median_loc_score']:.3f}, "
            f"mean_cov={s['mean_coverage']:.3f}, "
            f"mean_pars={s['mean_parsimony']:.3f}, "
            f"time={s['mean_time']*1000:.1f} ms, "
            f"pos={s['n_pos']}, neg={s['n_neg']}"
        )
