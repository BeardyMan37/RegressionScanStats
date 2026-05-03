"""
Clean ALMA spectral data in three stages:
  1. Discard rows with low variation or edge plateau (group-majority vote)
  2. Drop entire rows that have ANY atmospheric interference
  3. Trim flagged indices (True in flag_array) from surviving rows

Produces a new DataFrame with cleaned arrays and drops rows
where no valid data remains after cleaning.
"""

from __future__ import annotations
import os, ast, numpy as np, pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.signal import find_peaks, peak_widths
from itertools import groupby


# ---------------------------------------------------------------------------
# Helpers (reused from the original loader)
# ---------------------------------------------------------------------------

def match_and_correct(freq_array, trans_freqs, trans_vals) -> np.ndarray:
    idxs = np.searchsorted(trans_freqs, freq_array)
    idxs[idxs == len(trans_freqs)] = len(trans_freqs) - 1
    left = np.maximum(idxs - 1, 0)
    right = idxs
    dl = np.abs(freq_array - trans_freqs[left])
    dr = np.abs(trans_freqs[right] - freq_array)
    nearest = np.where(dl <= dr, left, right)
    return trans_vals[nearest]


def _parse_freqs(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=float) / 1e9


def _parse_float_list(s) -> np.ndarray:
    if isinstance(s, str):
        return np.array(ast.literal_eval(s), dtype=float)
    return np.asarray(s, dtype=float)


def _parse_bool_list(s) -> Optional[np.ndarray]:
    if s is None or (isinstance(s, str) and s.strip() in ("", "[]")):
        return None
    if isinstance(s, str):
        return np.array(ast.literal_eval(s), dtype=bool)
    return np.asarray(s, dtype=bool)


def _compute_atmospheric_interference(
    freqs: np.ndarray, trans: np.ndarray
) -> List[Tuple[int, int]]:
    """Detect atmospheric absorption troughs and return affected index ranges."""
    troughs, _ = find_peaks(-trans, prominence=1)
    if len(troughs) == 0:
        return []

    _, _, left_ips, right_ips = peak_widths(-trans, troughs, rel_height=0.75)

    left_freqs = np.interp(left_ips, np.arange(len(freqs)), freqs)
    right_freqs = np.interp(right_ips, np.arange(len(freqs)), freqs)
    widths_freq = right_freqs - left_freqs

    trough_freqs = freqs[troughs]
    trough_ranges = np.column_stack(
        (trough_freqs - widths_freq / 2.0, trough_freqs + widths_freq / 2.0)
    )

    closest_idxs: List[Tuple[int, int]] = []
    for start_f, end_f in trough_ranges:
        s_idx = int(np.abs(freqs - start_f).argmin())
        e_idx = int(np.abs(freqs - end_f).argmin())
        closest_idxs.append((s_idx, e_idx))
    return closest_idxs


def _to_np(x):
    """Convert to float array, dropping non-finite values."""
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def classify_row(amplitude, min_len=16, abs_std_thresh=1e-2,
                 edge_frac=0.03, k_min=4, z_edge=10.0, rel_edge=0.10,
                 low_var_cv_thresh=1e-4):
    """
    Classify a single amplitude array as one of:
      - "low_var"       : near-constant signal (likely bad / fill-value data)
      - "edge_plateau"  : both edges deviate from the centre in the same
                          direction by a large amount (likely bandpass artefact)
      - "regular"       : normal data worth keeping
 
    Parameters (tuned to minimise false positives)
    ----------
    min_len : int
        Arrays shorter than this are always "regular" (too short to judge).
    abs_std_thresh : float
        If max |y - mean| is below this, the row is "low_var".  Raised to
        1e-2 so that only truly flat / fill-value spectra are caught.
    low_var_cv_thresh : float
        Secondary low-var check: coefficient of variation (std/|mean|) must
        also be below this threshold.  Prevents flagging spectra that happen
        to have a small absolute range but meaningful relative variation.
    edge_frac : float
        Fraction of the band used to compute each edge mean (default 3%).
    k_min : int
        Minimum number of channels per edge (default 4).
    z_edge : float
        Number of centre-region standard deviations an edge must deviate by
        to be considered anomalous.  Raised to 10 to be very conservative.
    rel_edge : float
        Minimum fractional deviation (|edge - centre| / |centre|) for an
        edge to count as anomalous.  Raised to 10% so normal bandpass
        roll-off is not flagged.
    """
    y = _to_np(amplitude)
    n = y.size
    if n < min_len:
        return "regular"
 
    # ---- Low-variance check ----
    mu = y.mean()
    max_dev = np.max(np.abs(y - mu))
    if max_dev < abs_std_thresh:
        # Double-check with coefficient of variation to avoid false positives
        cv = np.std(y) / max(abs(mu), 1e-15)
        if cv < low_var_cv_thresh:
            return "low_var"
 
    # ---- Edge-plateau check ----
    k = max(int(edge_frac * n), k_min)
    k = min(k, n // 4)
    if k < 2:
        return "regular"
 
    left_mean  = float(np.mean(y[:k]))
    right_mean = float(np.mean(y[-k:]))
    center     = y[k:-k] if n >= 2 * k + 1 else y
    c_med      = float(np.median(center))
    c_std      = float(np.std(center))
 
    # Threshold: must exceed BOTH the z-score AND the relative threshold
    abs_thr = z_edge * c_std
    rel_thr = rel_edge * max(abs(c_med), 1e-9)
    thr = max(abs_thr, rel_thr)
 
    left_dev  = left_mean  - c_med
    right_dev = right_mean - c_med
 
    if (abs(left_dev) > thr
            and abs(right_dev) > thr
            and np.sign(left_dev) == np.sign(right_dev)):
        return "edge_plateau"
 
    return "regular"


def _majority_with_tiebreak(s: pd.Series) -> str:
    """
    Return the most frequent label in *s*.  On ties, prefer the more
    aggressive discard label (edge_plateau > low_var > regular).
    """
    priority = {"edge_plateau": 2, "low_var": 1, "regular": 0}
    vc = s.value_counts()
    best, bestc = None, -1
    for k, c in vc.items():
        if c > bestc or (c == bestc and priority.get(k, -1) > priority.get(best, -1)):
            best, bestc = k, c
    return best


# ---------------------------------------------------------------------------
# Main cleanup function
# ---------------------------------------------------------------------------

def clean_spectral_data(
    data_path: str,
    interference_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load raw spectral data, classify and clean it in three stages,
    and return a cleaned DataFrame.

    Stages
    ------
    1. **Low-var / edge-plateau removal** — rows are classified per-row,
       then a group-majority vote (by eb_uid/antenna/spw/pol) decides
       whether the group is bad.  Groups voted as ``low_var`` or
       ``edge_plateau`` are discarded entirely.
    2. **Atmospheric interference** — any row whose frequency band
       overlaps a detected atmospheric absorption trough is dropped.
    3. **Flag trimming** — indices marked ``True`` in ``flag_array``
       (typically at band edges) are removed from the arrays of
       surviving rows.  Rows left with no valid channels are dropped.

    Parameters
    ----------
    data_path : str
        Path to the raw data file (.csv pipe-delimited or .parquet).
    interference_path : str
        Path to the atmospheric transmission parquet file.
    output_path : str, optional
        If provided, save the cleaned DataFrame here (.parquet or .csv).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """

    # ---- Load raw data ----
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path, sep="|", dtype=str, header=0)
        df["frequency_array"] = df["frequency_array"].apply(_parse_freqs)
        df["amplitude"] = df["amplitude"].apply(_parse_float_list)
        df["phase"] = df["phase"].apply(_parse_float_list)
        df["flag_array"] = df["flag_array"].apply(_parse_bool_list)
        # If CSV has amplitude_corr_tsys instead of amplitude, use that
        if "amplitude_corr_tsys" in df.columns and "amplitude" not in df.columns:
            df["amplitude"] = df["amplitude_corr_tsys"].apply(_parse_float_list)

    elif data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
        df["frequency_array"] = df["frequency_array"].apply(
            lambda xs: np.array(xs, dtype=float)
        )
        df["amplitude"] = df["amplitude"].apply(lambda x: np.asarray(x, dtype=float))
        df["phase"] = df["phase"].apply(lambda x: np.asarray(x, dtype=float))
        if "flag_array" in df.columns:
            df["flag_array"] = df["flag_array"].apply(
                lambda x: np.asarray(x, dtype=bool) if x is not None else None
            )
        else:
            df["flag_array"] = None
    else:
        raise ValueError(f"Unsupported file extension: {data_path!r}")

    # ---- Load atmospheric transmission reference ----
    trans_df = pd.read_parquet(interference_path)
    trans_freqs = trans_df["Frequency (GHz)"].to_numpy()
    trans_vals = trans_df["Transmission (%)"].to_numpy()

    # ---- Stage 1: Discard low-variation and edge-plateau rows ----
    total_in = len(df)
    # Classify each row individually, then take group-majority vote
    # so that isolated misclassifications don't cause false drops.
    grp_cols = [c for c in ["eb_uid", "uid", "antenna_name", "spw_name_ms", "pol_id"]
                if c in df.columns]

    df["_row_type"] = df["amplitude"].apply(classify_row)

    if grp_cols:
        grp_majority = (
            df.groupby(grp_cols, dropna=False)["_row_type"]
              .apply(_majority_with_tiebreak)
              .rename("_grp_type")
              .reset_index()
        )
        df = df.merge(grp_majority, on=grp_cols, how="left")
    else:
        # No grouping columns available — fall back to per-row classification
        df["_grp_type"] = df["_row_type"]

    n_lowvar = int((df["_grp_type"] == "low_var").sum())
    n_edge   = int((df["_grp_type"] == "edge_plateau").sum())
    print(f"Stage 1 — dropped {n_lowvar} low-var rows, {n_edge} edge-plateau rows")

    df = df[df["_grp_type"] == "regular"].drop(
        columns=["_row_type", "_grp_type"]
    ).reset_index(drop=True)

    total_after_stage1 = len(df)

    # ---- Stage 2: Drop rows with ANY atmospheric interference ----
    keep_row = []
    has_interference_count = 0

    for i in df.index:
        freqs = np.asarray(df.at[i, "frequency_array"], dtype=float)
        amps = np.asarray(df.at[i, "amplitude"], dtype=float)

        # Drop all-zero spectra
        if np.all(amps == 0.0):
            keep_row.append(False)
            continue

        # Check for atmospheric interference — drop entire row if any found
        trans = match_and_correct(freqs, trans_freqs, trans_vals)
        atm_ranges = _compute_atmospheric_interference(freqs, trans)
        if len(atm_ranges) > 0:
            keep_row.append(False)
            has_interference_count += 1
            continue

        keep_row.append(True)

    df["_keep"] = keep_row
    df = df[df["_keep"]].drop(columns=["_keep"]).reset_index(drop=True)
    print(f"Stage 2 — dropped {has_interference_count} rows due to atmospheric interference")

    # ---- Stage 3: Trim flagged indices from surviving rows ----
    clean_freqs = []
    clean_amps = []
    clean_phases = []
    final_keep = []

    for i in df.index:
        freqs = np.asarray(df.at[i, "frequency_array"], dtype=float)
        amps = np.asarray(df.at[i, "amplitude"], dtype=float)
        phases = np.asarray(df.at[i, "phase"], dtype=float)
        n = len(freqs)

        # Build discard mask from flag_array (True → discard)
        discard = np.zeros(n, dtype=bool)
        flag_arr = df.at[i, "flag_array"]
        if flag_arr is not None:
            flag_arr = np.asarray(flag_arr, dtype=bool)
            if len(flag_arr) == n:
                discard |= flag_arr

        valid = ~discard
        if not np.any(valid):
            final_keep.append(False)
            clean_freqs.append(np.array([]))
            clean_amps.append(np.array([]))
            clean_phases.append(np.array([]))
        else:
            final_keep.append(True)
            clean_freqs.append(freqs[valid])
            clean_amps.append(amps[valid])
            clean_phases.append(phases[valid])

    df["frequency_array"] = clean_freqs
    df["amplitude"] = clean_amps
    df["phase"] = clean_phases
    df["_keep"] = final_keep

    df = df[df["_keep"]].drop(columns=["_keep", "flag_array"]).reset_index(drop=True)

    # Select output columns
    output_cols = [
        c for c in [
            "eb_uid", "uid", "time", "receiver_band", "ref_antenna_name",
            "antenna_name", "baseband_name", "spw_name_ms", "pol_id",
            "frequency_array", "amplitude", "phase", "QA2 Flag(s)"
        ]
        if c in df.columns
    ]
    df = df[output_cols]

    # ---- Optionally save ----
    if output_path:
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        elif output_path.endswith(".csv"):
            df.to_csv(output_path, index=False)
        else:
            df.to_parquet(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

    n_flag_dropped = sum(not k for k in final_keep)
    print(f"Stage 3 — dropped {n_flag_dropped} rows after flag trimming (no channels left)")

    total_out = len(df)
    print(f"\nSummary: {total_in} rows in → {total_out} rows out "
          f"(dropped {total_in - total_out} total)")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean ALMA spectral data")
    parser.add_argument("--data-path", help="Path to raw data (.csv or .parquet)")
    parser.add_argument("--interference-path", help="Path to atmospheric transmission .parquet")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path for cleaned data (.parquet or .csv)"
    )
    args = parser.parse_args()

    cleaned = clean_spectral_data(args.data_path, args.interference_path, args.output)
    print(cleaned.head())