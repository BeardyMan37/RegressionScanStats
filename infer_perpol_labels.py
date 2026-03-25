#!/usr/bin/env python3
"""
infer_perpol_labels.py  (pair-by-pair, EB×SPW thresholds)

What it does
------------
1) Builds a per-(EB_UID, SPW) reference of "good" scores using qa2flag in {0,2}.
2) Computes median and nMAD of that good pool and sets thresholds:
     t_ambig = median + 3*nMAD        (or median - 3*nMAD if lower-is-worse)
     t_bad   = median + 4*nMAD        (or median - 4*nMAD if lower-is-worse)
3) Within BAD data (qa2flag in --bad-flags), groups by (EB_UID, SPW, Antenna)
   to form 2-pol “bags”:
     - 2 pols: classify each pol independently vs thresholds (0/0.5/1).
     - 1 pol: leave unclassified (rule=single-pol).
     - >2 rows or other anomalies: write to *_errors.csv and mark unclassified.
4) Writes:
     - per-pol CSV (bad bags only): inferred_pol_label ∈ {0,0.5,1} (+ *_str)
     - stats CSV per (EB_UID, SPW): mean/median/nMAD/t_ambig/t_bad/n_good_pols
     - error CSV with any unexpected conditions.

CLI deps: pandas, numpy, matplotlib. Parquet requires pyarrow or fastparquet.

Usage: python infer_perpol_labels.py \
  --input QA2_Bandpass_Data_Labeled_Filtered_Scored.parquet \
  --output output_infer_QA2_fixed.csv \
  --stats stats_infer_QA2_fixed.txt \
  --png plots_infer_QA2_fixed \
  --score-col score_fixed --ant-col antenna_name --pol-col pol_id --qa2flag-col "QA2 Flag(s)" \
  --qa2flag-mapping QA2_bandpassflag_mapping.in \
  --compute-win-chans-using-cols win_fixed_start win_fixed_end \
  --compute-segment-width-using-cols frequency_array win_fixed_start win_fixed_end
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from utils.io import read_table
from utils import qa2flag_util

# ---- tweakable in code (not CLI) ---------------------------------------------
# Bags with these qa2flag values are treated as "good" reference samples
GOOD_BAG_FLAGS = {0, 2}
# -----------------------------------------------------------------------------


def nmad(x: np.ndarray) -> float:
    """Normalized MAD: 1.4826 * median(|x - median(x)|)."""
    x = np.asarray(x)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def ensure_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def aggregate_scores(df: pd.DataFrame, score_col: str, agg: str) -> float:
    if agg == "max":
        return float(df[score_col].max())
    elif agg == "mean":
        return float(df[score_col].mean())
    elif agg == "median":
        return float(df[score_col].median())
    else:
        raise ValueError(f"Unknown agg '{agg}'")


def percentile_of_good(good_vals: np.ndarray, s: float, higher_is_worse: bool) -> float:
    """Percentile of s within the good distribution (diagnostic only)."""
    if len(good_vals) == 0 or not pd.notna(s):
        return float("nan")
    good_vals = np.asarray(good_vals)
    if higher_is_worse:
        return float(np.mean(good_vals <= s))
    else:
        return float(np.mean(good_vals >= s))


def classify_one(s, t_ambig, t_bad, higher_is_worse: bool):
    """Independent per-pol classification using median±k*nMAD thresholds.

    higher_is_worse:
      s < t_ambig -> 0 (good)
      t_ambig <= s < t_bad -> 0.5 (ambiguous)
      s >= t_bad -> 1 (bad)

    lower_is_worse:
      s > t_ambig -> 0
      t_bad < s <= t_ambig -> 0.5
      s <= t_bad -> 1
    """
    if not pd.notna(s):
        return 0.5, "nan-score"
    if higher_is_worse:
        if s < t_ambig:
            return 0.0, "indep"
        if s >= t_bad:
            return 1.0, "indep"
        return 0.5, "indep"
    else:
        if s > t_ambig:
            return 0.0, "indep"
        if s <= t_bad:
            return 1.0, "indep"
        return 0.5, "indep"


def main():
    ap = argparse.ArgumentParser(
        description="Infer per-pol labels with EB×SPW med±k·nMAD thresholds; pair-by-pair inside bad bags."
    )
    ap.add_argument("--input", required=True, help="Input parquet/csv")
    ap.add_argument(
        "--output", required=True, help="Output per-pol CSV (bad bags only)"
    )
    ap.add_argument("--stats", required=True, help="Output EB×SPW stats CSV")
    ap.add_argument("--png", default=None, help="Optional scatter PNG prefix")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--eb-col", default="eb_uid")
    ap.add_argument("--spw-col", default="spw_name_ms")
    ap.add_argument("--ant-col", default="antenna")
    ap.add_argument("--pol-col", default="polarization")
    ap.add_argument("--qa2flag-col", default="qa2flag")
    ap.add_argument(
        "--qa2flag-mapping",
        help=(
            "A file contains a list of mapping, in the format of "
            "'(wildcard:|regex:|)pattern -> value'. When provided, "
            "treat qa2flag as a list of string and apply the mapping "
            "in the provided file."
        ),
    )
    ap.add_argument(
        "--compute-win-chans-using-cols",
        nargs=2,
        help="The win_start and win_end columns used for computing win_chans.",
    )
    ap.add_argument(
        "--compute-segment-width-using-cols",
        nargs=3,
        help=(
            "The frequency_array, win_start, and win_end columns used for "
            "computing segment_width by looking up the win_start-th and "
            "win_end-th element in  frequency_array."
        ),
    )
    ap.add_argument(
        "--bad-flags",
        nargs="+",
        type=int,
        default=[1],
        help="qa2flag values considered BAD",
    )
    ap.add_argument(
        "--agg",
        choices=["max", "mean", "median"],
        default="max",
        help="Aggregate rows per (bag,pol)",
    )
    ap.add_argument(
        "--higher-is-worse",
        action="store_true",
        default=True,
        help="Interpret higher scores as worse",
    )
    ap.add_argument(
        "--lower-is-worse", action="store_true", help="Flip: lower scores are worse"
    )
    args = ap.parse_args()

    higher_is_worse = not args.lower_is_worse

    # Load
    df = read_table(Path(args.input))
    ensure_cols(
        df,
        [
            args.eb_col,
            args.spw_col,
            args.ant_col,
            args.pol_col,
            args.score_col,
            args.qa2flag_col,
        ],
    )

    # Normalize pol names
    df[args.pol_col] = df[args.pol_col].astype(str).str.strip().str.upper()

    # Transform qa2flag is qa2flag mapping exists
    if args.qa2flag_mapping is not None:
        qa2flag_mapping_list = qa2flag_util.parse_qa2flag_mapping_file(
            args.qa2flag_mapping
        )
        df[args.qa2flag_col] = df[args.qa2flag_col].transform(
            partial(
                qa2flag_util.transform,
                qa2flag_mapping_list=qa2flag_mapping_list,
                default=0,
            )
        )

    # Compute win_chans if specified. Existing value is overwritten.
    if args.compute_win_chans_using_cols is not None:
        win_start_col, win_end_col = args.compute_win_chans_using_cols
        df["win_chans"] = df.apply(
            lambda row: abs(row[win_end_col] - row[win_start_col]), axis="columns"
        ).astype(int)

    # Compute segment_width if specified. Existing value is overwritten.
    if args.compute_segment_width_using_cols is not None:
        (frequency_array_col,) = filter(
            lambda col: df[col].dtype != int, args.compute_segment_width_using_cols
        )
        win_start_col, win_end_col = filter(
            lambda col: df[col].dtype == int, args.compute_segment_width_using_cols
        )
        df["segment_width"] = df.apply(
            lambda row: abs(
                row[frequency_array_col][row[win_end_col]]
                - row[frequency_array_col][row[win_start_col]]
            )
            / 1e6,  # MHz
            axis="columns",
        )

    # Aggregate to one score per (EB, SPW, ANT, POL)
    key_cols = [args.eb_col, args.spw_col, args.ant_col, args.pol_col]
    agg_df = (
        df.groupby(key_cols, as_index=False)
        .apply(
            lambda g: pd.Series({
                args.score_col: aggregate_scores(g, args.score_col, args.agg),
                args.qa2flag_col: g[args.qa2flag_col].iloc[
                    0
                ],  # assume same within group
                "segment_width": g["segment_width"].max()
                if "segment_width" in g.columns
                else np.nan,
                "win_chans": g["win_chans"].max()
                if "win_chans" in g.columns
                else np.nan,
            })
        )
        .reset_index(drop=True)
    )

    out_rows, stats_rows, error_rows = [], [], []

    # Iterate per EB×SPW to compute good distribution + thresholds
    for (eb, spw), g in agg_df.groupby([args.eb_col, args.spw_col], as_index=False):
        good_mask = g[args.qa2flag_col].isin(GOOD_BAG_FLAGS)
        good_vals = g.loc[good_mask, args.score_col].dropna().values

        if len(good_vals) < 4:
            mean_g = med_g = nmad_g = float("nan")
            t_ambig = t_bad = float("nan")
        else:
            mean_g = float(np.mean(good_vals))
            med_g = float(np.median(good_vals))
            nmad_g = float(nmad(good_vals))
            if higher_is_worse:
                t_ambig = float(med_g + 3.0 * nmad_g)
                t_bad = float(med_g + 4.0 * nmad_g)
            else:
                t_ambig = float(med_g - 3.0 * nmad_g)
                t_bad = float(med_g - 4.0 * nmad_g)

        print(
            f"[{eb} | {spw}] good_mean={mean_g:.6g}  good_median={med_g:.6g}  good_nMAD={nmad_g:.6g}  "
            f"t_ambig={t_ambig:.6g}  t_bad={t_bad:.6g}  n_good={int(good_mask.sum())}"
        )

        stats_rows.append({
            args.eb_col: eb,
            args.spw_col: spw,
            "good_mean": mean_g,
            "good_median": med_g,
            "good_nmad": nmad_g,
            "t_ambig": t_ambig,
            "t_bad": t_bad,
            "n_good_pols": int(good_mask.sum()),
        })

        # Pair-by-pair on BAD groups: (EB, SPW, ANT)
        bad_mask = g[args.qa2flag_col].isin(args.bad_flags)
        for ant, gg in g.loc[bad_mask].groupby(args.ant_col):
            # If thresholds missing, emit unclassified and log error
            if not pd.notna(t_ambig) or not pd.notna(t_bad):
                for _, row in gg.iterrows():
                    out_rows.append({
                        args.eb_col: eb,
                        args.spw_col: spw,
                        args.ant_col: ant,
                        args.pol_col: str(row[args.pol_col]),
                        args.score_col: float(row[args.score_col])
                        if pd.notna(row[args.score_col])
                        else float("nan"),
                        args.qa2flag_col: int(row[args.qa2flag_col]),
                        "segment_width": float(row.get("segment_width", float("nan")))
                        if "segment_width" in g.columns
                        else float("nan"),
                        "win_chans": float(row.get("win_chans", float("nan")))
                        if "win_chans" in g.columns
                        else float("nan"),
                        "good_percentile": float("nan"),
                        "inferred_pol_label": float("nan"),
                        "inferred_pol_label_str": "unclassified",
                        "rule": "no-good-stats",
                        "t_ambig": t_ambig,
                        "t_bad": t_bad,
                    })
                error_rows.append({
                    "eb_uid": eb,
                    "spw_name_ms": spw,
                    "antenna": ant,
                    "reason": "no-good-stats",
                    "details": f"Insufficient reference: n_good={len(good_vals)}",
                    "n_rows_in_group": len(gg),
                })
                continue

            # Identify X-like and Y-like
            X_like = gg[gg[args.pol_col].isin(["X", "R", "H"])]
            Y_like = gg[gg[args.pol_col].isin(["Y", "L", "V"])]
            pol_count = int(len(X_like) > 0) + int(len(Y_like) > 0)

            if pol_count == 0:
                error_rows.append({
                    "eb_uid": eb,
                    "spw_name_ms": spw,
                    "antenna": ant,
                    "reason": "no-pols-in-bag",
                    "details": "No X-like or Y-like rows present.",
                    "n_rows_in_group": len(gg),
                })
                for _, row in gg.iterrows():
                    out_rows.append({
                        args.eb_col: eb,
                        args.spw_col: spw,
                        args.ant_col: ant,
                        args.pol_col: str(row[args.pol_col]),
                        args.score_col: float(row[args.score_col])
                        if pd.notna(row[args.score_col])
                        else float("nan"),
                        args.qa2flag_col: int(row[args.qa2flag_col]),
                        "segment_width": float(row.get("segment_width", float("nan")))
                        if "segment_width" in g.columns
                        else float("nan"),
                        "win_chans": float(row.get("win_chans", float("nan")))
                        if "win_chans" in g.columns
                        else float("nan"),
                        "good_percentile": float("nan"),
                        "inferred_pol_label": float("nan"),
                        "inferred_pol_label_str": "unclassified",
                        "rule": "no-pols-in-bag",
                        "t_ambig": t_ambig,
                        "t_bad": t_bad,
                    })
                continue

            if pol_count > 2 or len(gg) > 2:
                error_rows.append({
                    "eb_uid": eb,
                    "spw_name_ms": spw,
                    "antenna": ant,
                    "reason": "unexpected-pol-count",
                    "details": f"Found {len(gg)} rows; expected <=2 (one X-like, one Y-like).",
                    "n_rows_in_group": len(gg),
                })
                for _, row in gg.iterrows():
                    out_rows.append({
                        args.eb_col: eb,
                        args.spw_col: spw,
                        args.ant_col: ant,
                        args.pol_col: str(row[args.pol_col]),
                        args.score_col: float(row[args.score_col])
                        if pd.notna(row[args.score_col])
                        else float("nan"),
                        args.qa2flag_col: int(row[args.qa2flag_col]),
                        "segment_width": float(row.get("segment_width", float("nan")))
                        if "segment_width" in g.columns
                        else float("nan"),
                        "win_chans": float(row.get("win_chans", float("nan")))
                        if "win_chans" in g.columns
                        else float("nan"),
                        "good_percentile": float("nan"),
                        "inferred_pol_label": float("nan"),
                        "inferred_pol_label_str": "unclassified",
                        "rule": "unexpected-pol-count",
                        "t_ambig": t_ambig,
                        "t_bad": t_bad,
                    })
                continue

            if pol_count == 1:
                row = X_like.iloc[0] if len(X_like) else Y_like.iloc[0]
                out_rows.append({
                    args.eb_col: eb,
                    args.spw_col: spw,
                    args.ant_col: ant,
                    args.pol_col: str(row[args.pol_col]),
                    args.score_col: float(row[args.score_col])
                    if pd.notna(row[args.score_col])
                    else float("nan"),
                    args.qa2flag_col: int(row[args.qa2flag_col]),
                    "segment_width": float(row.get("segment_width", float("nan")))
                    if "segment_width" in g.columns
                    else float("nan"),
                    "win_chans": float(row.get("win_chans", float("nan")))
                    if "win_chans" in g.columns
                    else float("nan"),
                    "good_percentile": percentile_of_good(
                        good_vals,
                        float(row[args.score_col])
                        if pd.notna(row[args.score_col])
                        else float("nan"),
                        higher_is_worse,
                    ),
                    "inferred_pol_label": float("nan"),
                    "inferred_pol_label_str": "single-pol",
                    "rule": "single-pol",
                    "t_ambig": t_ambig,
                    "t_bad": t_bad,
                })
                continue

            # Two-pol case
            sx = float(X_like[args.score_col].iloc[0]) if len(X_like) else float("nan")
            sy = float(Y_like[args.score_col].iloc[0]) if len(Y_like) else float("nan")

            if (not pd.notna(sx)) or (not pd.notna(sy)):
                error_rows.append({
                    "eb_uid": eb,
                    "spw_name_ms": spw,
                    "antenna": ant,
                    "reason": "nan-score",
                    "details": f"NaN score(s) in pair: sx={sx}, sy={sy}",
                    "n_rows_in_group": len(gg),
                })

            # only classify the LOWER score; leave the higher one alone
            x_flag = float(X_like[args.qa2flag_col].iloc[0])
            y_flag = float(Y_like[args.qa2flag_col].iloc[0])

            if sx >= sy:  # X has the higher (numeric) score
                lx, rx = x_flag, "left-alone"
                ly, ry = classify_one(sy, t_ambig, t_bad, higher_is_worse)
            else:  # Y has the higher score
                lx, rx = classify_one(sx, t_ambig, t_bad, higher_is_worse)
                ly, ry = y_flag, "left-alone"
            rule = "lower-only"

            # lx, rx = classify_one(sx, t_ambig, t_bad, higher_is_worse)
            # ly, ry = classify_one(sy, t_ambig, t_bad, higher_is_worse)
            # rule = "pair-indep" if rx == "indep" or ry == "indep" else (rx if rx != "nan-score" else ry)

            px = percentile_of_good(good_vals, sx, higher_is_worse)
            py = percentile_of_good(good_vals, sy, higher_is_worse)

            out_rows.append({
                args.eb_col: eb,
                args.spw_col: spw,
                args.ant_col: ant,
                args.pol_col: str(X_like[args.pol_col].iloc[0]),
                args.score_col: sx,
                args.qa2flag_col: int(X_like[args.qa2flag_col].iloc[0]),
                "segment_width": float(X_like["segment_width"].iloc[0])
                if "segment_width" in X_like.columns
                else float("nan"),
                "win_chans": float(X_like["win_chans"].iloc[0])
                if "win_chans" in X_like.columns
                else float("nan"),
                "good_percentile": px,
                "inferred_pol_label": float(lx),
                "inferred_pol_label_str": (
                    "good" if lx == 0.0 else ("ambiguous" if lx == 0.5 else "bad")
                ),
                "rule": rule,
                "t_ambig": t_ambig,
                "t_bad": t_bad,
            })
            out_rows.append({
                args.eb_col: eb,
                args.spw_col: spw,
                args.ant_col: ant,
                args.pol_col: str(Y_like[args.pol_col].iloc[0]),
                args.score_col: sy,
                args.qa2flag_col: int(Y_like[args.qa2flag_col].iloc[0]),
                "segment_width": float(Y_like["segment_width"].iloc[0])
                if "segment_width" in Y_like.columns
                else float("nan"),
                "win_chans": float(Y_like["win_chans"].iloc[0])
                if "win_chans" in Y_like.columns
                else float("nan"),
                "good_percentile": py,
                "inferred_pol_label": float(ly),
                "inferred_pol_label_str": (
                    "good" if ly == 0.0 else ("ambiguous" if ly == 0.5 else "bad")
                ),
                "rule": rule,
                "t_ambig": t_ambig,
                "t_bad": t_bad,
            })

    # Write outputs
    perpol_out = pd.DataFrame(out_rows)
    stats_out = pd.DataFrame(stats_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    perpol_out.to_csv(args.output, index=False)
    stats_out.to_csv(args.stats, index=False)

    # Error CSV (always created)
    err_path = Path(args.output).with_name(Path(args.output).stem + "_errors.csv")
    if len(error_rows) > 0:
        pd.DataFrame(error_rows).to_csv(err_path, index=False)
        print(f"⚠️  error log written → {err_path.resolve()}")
    else:
        pd.DataFrame(
            columns=[
                "eb_uid",
                "spw_name_ms",
                "antenna",
                "reason",
                "details",
                "n_rows_in_group",
            ]
        ).to_csv(err_path, index=False)
        print(f"ℹ️  no errors recorded; empty log created → {err_path.resolve()}")

    # Optional plotting
    if args.png:
        if perpol_out.empty:
            print("No per-pol output rows to plot; skipping PNGs.")
        else:
            # 1) segment_width vs score
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            color_map = {0.0: "green", 1.0: "red", 0.5: "yellow"}
            for lab, marker in [(0.0, "o"), (1.0, "x"), (0.5, "^")]:
                sub = perpol_out[perpol_out["inferred_pol_label"] == lab]
                if not sub.empty:
                    if lab == 1.0:
                        ax1.scatter(
                            sub.get("segment_width", np.nan),
                            sub[args.score_col],
                            marker=marker,
                            facecolors=color_map[lab],
                            alpha=0.9,
                            label=str(lab),
                        )
                    else:
                        ax1.scatter(
                            sub.get("segment_width", np.nan),
                            sub[args.score_col],
                            marker=marker,
                            facecolors="none",
                            edgecolors=color_map[lab],
                            alpha=0.9,
                            label=str(lab),
                        )
            ax1.set_xlabel(
                "segment_width [MHz]"
                if "segment_width" in perpol_out.columns
                else "segment_width"
            )
            ax1.set_ylabel(args.score_col)
            ax1.set_title("Segment width vs score (o=good/0, ^=ambig/0.5, x=bad/1)")
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(args.png, dpi=160)
            plt.close(fig1)

            # 2) win_chans vs score
            png2 = str(Path(args.png).with_name(Path(args.png).stem + "_chan.png"))
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            for lab, marker in [(0.0, "o"), (1.0, "x"), (0.5, "^")]:
                sub = perpol_out[perpol_out["inferred_pol_label"] == lab]
                if not sub.empty and "win_chans" in sub.columns:
                    ax2.scatter(
                        sub["win_chans"],
                        sub[args.score_col],
                        marker=marker,
                        facecolors="none",
                        edgecolors=color_map[lab],
                        alpha=0.9,
                        label=str(lab),
                    )
            ax2.set_xlabel(
                "win_chans [channels]"
                if "win_chans" in perpol_out.columns
                else "win_chans"
            )
            ax2.set_ylabel(args.score_col)
            ax2.set_title("Win-channels vs score (o=good/0, ^=ambig/0.5, x=bad/1)")
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig(png2, dpi=160)
            plt.close(fig2)

            # 3) segment_width vs normalized score: (score - median_good) / nMAD_good
            png3 = str(Path(args.png).with_name(Path(args.png).stem + "_nscore.png"))

            # Join per-pol outputs with per-EB×SPW stats to get median & nMAD
            join_cols = [args.eb_col, args.spw_col]
            merged = perpol_out.merge(
                stats_out[join_cols + ["good_median", "good_nmad"]],
                on=join_cols,
                how="left",
            ).copy()

            # Compute normalized score; guard against zero/NaN nMAD
            eps = 1e-12
            den = merged["good_nmad"].where(merged["good_nmad"].abs() > eps, np.nan)
            merged["norm_score"] = (
                merged[args.score_col] - merged["good_median"]
            ) / den

            # Drop rows where we can't normalize
            plot_df = merged.dropna(subset=["norm_score", "segment_width"])

            fig3, ax3 = plt.subplots(figsize=(8, 6))
            color_map = {0.0: "green", 1.0: "red", 0.5: "yellow"}

            for lab, marker in [(0.0, "o"), (1.0, "x"), (0.5, "^")]:
                sub = plot_df[plot_df["inferred_pol_label"] == lab]
                if sub.empty:
                    continue

                if marker in ["o", "^"]:
                    ax3.scatter(
                        sub["segment_width"],
                        sub["norm_score"],
                        marker=marker,
                        facecolors="none",  # hollow for circle/triangle
                        edgecolors=color_map[lab],
                        alpha=0.9,
                        label=str(lab),
                    )
                else:  # for 'x'
                    ax3.scatter(
                        sub["segment_width"],
                        sub["norm_score"],
                        marker=marker,
                        color=color_map[lab],  # solid color for x
                        alpha=0.9,
                        label=str(lab),
                    )

            ax3.set_xlabel("segment_width [MHz]")
            ax3.set_ylabel("(score - median_good) / nMAD_good")
            ax3.set_title(
                "Normalized score vs segment width (o=0 good, ^=0.5 ambig, x=1 bad)"
            )
            ax3.axhline(0.0, linestyle="--", linewidth=1.0)
            fig3.tight_layout()
            fig3.savefig(png3, dpi=160)
            plt.close(fig3)

            print(f"✅  normalized-score plot → {png3}")

            print(
                f"✅  plots saved → {Path(args.png).resolve()} and {Path(png2).resolve()}"
            )


if __name__ == "__main__":
    main()
