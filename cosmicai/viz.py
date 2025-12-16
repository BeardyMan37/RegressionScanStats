from __future__ import annotations
import os, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Any
import logging

def _clamp_pair(a: int, b: int, n: int) -> tuple[int, int]:
    a = int(a); b = int(b)
    if a > b:
        a, b = b, a
    a = max(0, min(a, n - 1))
    b = max(0, min(b, n - 1))
    return a, b


def plot_top_k(
    df: pd.DataFrame,
    actual_spec_arrays: np.ndarray,
    freqs: np.ndarray,
    # windows (native coords, inclusive)
    windows_masked: List[Tuple[int, int]],
    windows_unmasked: List[Tuple[int, int]],
    windows_fixed: List[Tuple[int, int]],
    # scores
    scores_masked: List[float],
    scores_unmasked: List[float],
    scores_fixed: List[float],
    # overlap stats (fractions in 0..1 and counts) for unmasked + fixed
    overlap_unmasked_pct: List[float] | None,
    overlap_fixed_pct:    List[float] | None,
    # misc/meta
    atm_interfs: List[List[Tuple[int, int]]],
    meta: Dict[str, Any],
    ws: List[int],
    # viz/IO
    k: int = 10,
    per_fig: int = 10,
    buffer: int = 10,
    out_dir: str = "Images/latest_run",
    data_dir: str = "Data/latest_run",
    # predictions
    sra_preds: List[np.ndarray] | None = None,
    sri_idxs_masked:  List[np.ndarray | None] | None = None,
    sri_vals_masked:  List[np.ndarray | None] | None = None,
    sri_idxs_unmasked:List[np.ndarray | None] | None = None,
    sri_vals_unmasked:List[np.ndarray | None] | None = None,
    sri_idxs_fixed:   List[np.ndarray | None] | None = None,
    sri_vals_fixed:   List[np.ndarray | None] | None = None,
    sr_factor: int = 1,
    fixed_bins_nat: List[int] | None = None,
    # ranking
    rank_by: str = "masked",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    buf_orig = buffer

    # Choose score vector for ranking
    score_map_all = {
        "masked":   np.asarray(scores_masked, dtype=float),
        "unmasked": np.asarray(scores_unmasked, dtype=float),
        "fixed":    np.asarray(scores_fixed, dtype=float),
    }
    if rank_by not in score_map_all:
        raise ValueError("rank_by must be one of: 'masked', 'unmasked', 'fixed'")
    scores_rank = score_map_all[rank_by]

    # Top ordering (desc)
    finite = np.isfinite(scores_rank)
    idx_all = np.where(finite)[0]
    order_desc = idx_all[np.argsort(scores_rank[finite])[:]][::-1]

    # Build CSV (for *all* rows in the filtered ordering)
    top_uids = np.asarray(meta["uid"])[order_desc]
    kernel_sizes = np.asarray(ws, dtype=object)[order_desc]

    sub_df = df.loc[df["uid"].isin(top_uids)].copy()

    # Map all columns by uid
    def _mk_map(vec): return dict(zip(top_uids, np.asarray(vec, dtype=object)[order_desc]))

    sub_df["score_masked"]   = sub_df["uid"].map(_mk_map(scores_masked))
    sub_df["score_unmasked"] = sub_df["uid"].map(_mk_map(scores_unmasked))
    sub_df["score_fixed"]    = sub_df["uid"].map(_mk_map(scores_fixed))
    sub_df["kernel_size"]    = sub_df["uid"].map(_mk_map(kernel_sizes))

    # Windows
    sub_df[["win_masked_start", "win_masked_end"]]     = sub_df["uid"].map(_mk_map(windows_masked)).apply(pd.Series)
    sub_df[["win_unmasked_start", "win_unmasked_end"]] = sub_df["uid"].map(_mk_map(windows_unmasked)).apply(pd.Series)
    sub_df[["win_fixed_start", "win_fixed_end"]]       = sub_df["uid"].map(_mk_map(windows_fixed)).apply(pd.Series)

    # Overlaps (optional)
    if overlap_unmasked_pct is not None:
        sub_df["overlap_unmasked_pct"] = sub_df["uid"].map(_mk_map(overlap_unmasked_pct))
    if overlap_fixed_pct is not None:
        sub_df["overlap_fixed_pct"] = sub_df["uid"].map(_mk_map(overlap_fixed_pct))
    if fixed_bins_nat is not None:
        sub_df["fixed_bins_native"] = sub_df["uid"].map(_mk_map(fixed_bins_nat))

    # Sort by chosen score
    sub_df["rank_score"] = sub_df["uid"].map(_mk_map(scores_rank))
    sub_df = sub_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    sub_df.insert(0, "uid", sub_df.pop("uid"))

    out_file = os.path.join(
        data_dir,
        f"output_length_{actual_spec_arrays.shape[1]}.parquet",
    )
    sub_df.to_parquet(out_file, engine='pyarrow', compression="zstd")
    logging.info("Wrote summary CSV (all windows): %s", out_file)

    # ---- Plot top-k with all windows overlaid ----
    top = order_desc[:min(k, len(order_desc))]
    n_figs = math.ceil(len(top) / per_fig)
    for fig_i in range(n_figs):
        batch = top[fig_i * per_fig : (fig_i + 1) * per_fig]
        fig_k = len(batch)
        fig, axes = plt.subplots(fig_k, 1, figsize=(10, 3 * fig_k))
        if fig_k == 1:
            axes = [axes]

        for ax, i0 in zip(axes, batch):
            spec = actual_spec_arrays[i0]
            freq = np.asarray(freqs[i0], dtype=float)
            a_m, b_m = windows_masked[i0]
            a_u, b_u = windows_unmasked[i0]
            a_f, b_f = windows_fixed[i0]

            # Atmospheric interference shading
            for (c, d) in atm_interfs[i0]:
                ci, di = _clamp_pair(c, d, len(freq))
                if ci <= di:
                    ax.axvspan(float(freq[ci]), float(freq[di]), color="C9", alpha=0.15, label=None)


            # Raw spectrum
            # x = np.arange(len(spec))
            ax.plot(freq, spec, color="C0", label="Actual")

            # Buffer shading
            if buf_orig > 0 and len(freq) > 0:
                li0, li1 = _clamp_pair(0, buf_orig - 1, len(freq))
                if li0 <= li1:
                    ax.axvspan(float(freq[li0]), float(freq[li1]), color="gray", alpha=0.15, label=None)
                ri0_start = max(len(freq) - buf_orig, 0)
                ri0, ri1 = _clamp_pair(ri0_start, len(freq) - 1, len(freq))
                if ri0 <= ri1:
                    ax.axvspan(float(freq[ri0]), float(freq[ri1]), color="gray", alpha=0.15, label=None)


            # Window overlays (distinct alphas)
            am0, am1 = _clamp_pair(a_m, b_m, len(freq))
            au0, au1 = _clamp_pair(a_u, b_u, len(freq))
            af0, af1 = _clamp_pair(a_f, b_f, len(freq))
            if am0 < am1: ax.axvspan(float(freq[am0]), float(freq[am1]), color="C1", alpha=0.35, label=None)
            if au0 < au1: ax.axvspan(float(freq[au0]), float(freq[au1]), color="C2", alpha=0.25, label=None)
            if af0 < af1: ax.axvspan(float(freq[af0]), float(freq[af1]), color="C3", alpha=0.25, label=None)


            # SRA predicted curve (shared)
            if sra_preds is not None and sra_preds[i0] is not None and len(sra_preds[i0]) > 0:
                pred_full = np.full(len(spec), np.nan)
                sra_up = np.repeat(sra_preds[i0], sr_factor)
                end = len(spec) - buf_orig
                pred_full[buf_orig:end] = sra_up[: max(0, end - buf_orig)]
                ax.plot(freq, pred_full, ".", ms=2, color="C4", label="SRA pred")

            # SRI overlays per window (optional)
            def _plot_sri(idx_list, val_list, label: str, color: str):
                if (idx_list is None) or (val_list is None):
                    return
                if (idx_list[i0] is None) or (val_list[i0] is None):
                    return
                if (idx_list[i0][0] == 0) and (idx_list[i0][1] == 0):
                    return
                sri_full = np.full(len(spec), np.nan)
                idx_sr  = np.asarray(idx_list[i0], dtype=int)
                val_sr  = np.asarray(val_list[i0], dtype=float)
                idx_orig_start = idx_sr * sr_factor
                for p, v in zip(idx_orig_start, val_sr):
                    p_end = min(p + sr_factor, len(spec))
                    sri_full[p:p_end] = v
                ax.plot(freq, sri_full, ".", ms=2, color= color, label=label)

            _plot_sri(sri_idxs_masked,   sri_vals_masked,   "SRI masked", "C5")
            _plot_sri(sri_idxs_unmasked, sri_vals_unmasked, "SRI unmasked", "C6")
            _plot_sri(sri_idxs_fixed,    sri_vals_fixed,    "SRI fixed", "C7")

            # Title with all scores (+overlaps if available)
            parts = [
                f"UID={i0}",
                f"S_masked={scores_masked[i0]:.2f}",
                f"S_unmasked={scores_unmasked[i0]:.2f}",
                f"S_fixed={scores_fixed[i0]:.2f}",
            ]
            if overlap_unmasked_pct is not None:
                parts.append(f"ovl_unm={100*overlap_unmasked_pct[i0]:.1f}%")
            if overlap_fixed_pct is not None:
                parts.append(f"ovl_fix={100*overlap_fixed_pct[i0]:.1f}%")
            if fixed_bins_nat is not None:
                parts.append(f"fixed_bins={fixed_bins_nat[i0]}")
            parts.append(f"W={ws[i0]}")
            ax.set_title("  ".join(parts))

            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Amplitude")

        legend_elements = [
            Line2D([0], [0], color="C0", label="Actual"),
            Line2D([0], [0], marker=".", linestyle="None", color="C4", label="SRA pred"),
            Line2D([0], [0], marker=".", linestyle="None", color="C5", label="SRI masked"),
            Line2D([0], [0], marker=".", linestyle="None", color="C6", label="SRI unmasked"),
            Line2D([0], [0], marker=".", linestyle="None", color="C7", label="SRI fixed"),
            Patch(facecolor="C1", alpha=0.35, label="Masked window"),
            Patch(facecolor="C2", alpha=0.25, label="Unmasked window"),
            Patch(facecolor="C3", alpha=0.25, label="Fixed window"),
            Patch(facecolor="C9", alpha=0.15, label="Interference"),
            Patch(facecolor="gray", alpha=0.15, label="Buffer"),
        ]

        plt.tight_layout()

        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=5,
            frameon=True,
            bbox_to_anchor=(0.5, 0.985),
        )

        fig.subplots_adjust(top=0.92)

        fig.suptitle(
            f"Top {min(k, len(order_desc))} ranked by {rank_by} — batch {fig_i+1}/{n_figs}",
            y=0.995
        )

        outpath = os.path.join(out_dir, f"top_{min(k, len(order_desc))}_by_{rank_by}_fig{fig_i + 1}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info("Wrote figure: %s", outpath)
