from __future__ import annotations
import os, ast, numpy as np, pandas as pd
from typing import Dict, List, Tuple
from scipy.signal import find_peaks, peak_widths
from itertools import groupby

def match_and_correct(freq_array, trans_freqs, trans_vals) -> np.ndarray:
    idxs = np.searchsorted(trans_freqs, freq_array)
    idxs[idxs == len(trans_freqs)] = len(trans_freqs) - 1
    left = np.maximum(idxs - 1, 0); right = idxs
    dl = np.abs(freq_array - trans_freqs[left])
    dr = np.abs(trans_freqs[right] - freq_array)
    nearest = np.where(dl <= dr, left, right)
    return trans_vals[nearest]

def _parse_freqs(s: str) -> np.ndarray:
    freqs = np.array(ast.literal_eval(s), dtype=float)
    return freqs / 1e9

def load_data_by_length(data_path: str, interference_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")
    if not os.path.exists(interference_path):
        raise FileNotFoundError(f"interference_path not found: {interference_path}")

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path, sep="|", dtype=str, header=0)
        df["uid"] = df.index.copy()
        df = df.reset_index(drop=True)
        df["frequency_array"] = df["frequency_array"].apply(_parse_freqs)
        trans_df = pd.read_parquet(interference_path)
        trans_freqs = trans_df["Frequency (GHz)"].to_numpy()
        trans_vals = trans_df["Transmission (%)"].to_numpy()

        df["transmission_array"] = df.apply(
            lambda row: match_and_correct(
                np.asarray(row["frequency_array"], dtype=float), trans_freqs, trans_vals
            ),
            axis=1,
        )

        interference: List[List[Tuple[int, int]]] = []
        for index in df.index:
            freqs = np.asarray(df.at[index, "frequency_array"], dtype=float)
            trans = np.asarray(df.at[index, "transmission_array"], dtype=float)

            troughs, _ = find_peaks(-trans, prominence=1)
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
            interference.append(closest_idxs)

        df["atmospheric_interference"] = interference

        actual_specs = [np.array(ast.literal_eval(s), dtype=float)
                        for s in df["amplitude_corr_tsys"]]
        freqs = [np.array(x, dtype=float) for x in df["frequency_array"].tolist()]

    elif data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
        df["uid"] = df.index.copy()
        df = df.reset_index(drop=True)
        # Hz->GHz
        df["frequency_array"] = df["frequency_array"].apply(lambda xs: [f / 1e9 for f in xs])
        trans_df = pd.read_parquet(interference_path)
        trans_freqs = trans_df["Frequency (GHz)"].to_numpy()
        trans_vals = trans_df["Transmission (%)"].to_numpy()

        df["transmission_array"] = df.apply(
            lambda row: match_and_correct(
                np.asarray(row["frequency_array"], dtype=float), trans_freqs, trans_vals
            ),
            axis=1,
        )

        interference = []
        for index in df.index:
            freqs = np.asarray(df.at[index, "frequency_array"], dtype=float)
            trans = np.asarray(df.at[index, "transmission_array"], dtype=float)

            troughs, _ = find_peaks(-trans, prominence=1)
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
            interference.append(closest_idxs)

        df["atmospheric_interference"] = interference

        def true_ranges(arr):
            ranges = []
            idx = 0
            for val, group in groupby(arr):
                length = sum(1 for _ in group)
                if val:
                    ranges.append((idx, idx + length - 1))
                idx += length
            return ranges
        
        df["flag_ranges"] = df["flag_array"].apply(true_ranges)
        actual_specs = [np.asarray(x, dtype=float) for x in df["amplitude"].tolist()]
        freqs = [np.asarray(x, dtype=float) for x in df["frequency_array"].tolist()]

    else:
        raise ValueError(f"Unsupported extension: {data_path!r}")

    df["_actual_spec"] = actual_specs
    df["_freqs"] = freqs
    df["_keep"] = [not np.all(s == 0.0) for s in df["_actual_spec"]]
    df = df[df["_keep"]].reset_index(drop=True)

    actual_specs = list(df["_actual_spec"])
    freqs = list(df["_freqs"])
    atm_intrf = list(df["atmospheric_interference"])
    flag_ranges = list(df["flag_ranges"])
    uid = df["uid"].to_numpy()
    ref = df["ref_antenna_name"].to_numpy()
    ant = df["antenna_name"].to_numpy()
    pol = df["pol_id"].to_numpy()

    length_groups: Dict[int, List[int]] = {}
    for i, s in enumerate(actual_specs):
        L = s.shape[0]
        length_groups.setdefault(L, []).append(i)

    groups: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Tuple[int, int]]]], List[List[Tuple[int, int]]]] = {}
    for L, idxs in length_groups.items():
        actual_specs_L = np.vstack([actual_specs[i] for i in idxs])
        freqs_L = np.vstack([freqs[i] for i in idxs])
        atm_intrf_L = [atm_intrf[i] for i in idxs]
        flag_ranges_L = [flag_ranges[i] for i in idxs]
        uid_L = uid[idxs]
        ref_L = ref[idxs]
        ant_L = ant[idxs]
        pol_L = pol[idxs]
        groups[L] = (actual_specs_L, uid_L, ref_L, ant_L, pol_L, freqs_L, atm_intrf_L, flag_ranges_L)

    return df, groups
