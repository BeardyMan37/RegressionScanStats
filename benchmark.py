from __future__ import annotations

import time
from typing import Dict, List, Tuple

import os
import numpy as np
import matplotlib.pyplot as plt

from cosmicai.config import ref_freq, set_kernel_kind
from cosmicai.parallel_exec import polynomial_scan_ranges_parallel
from cosmicai.scan import scan_row
from cosmicai.warmup import warmup_numba_and_caches


GroupsType = Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray, np.ndarray,
                             List[List[Tuple[int, int]]]]]
GroundTruthType = Dict[int, np.ndarray]


def generate_synthetic_groups(
    max_length: int = 4096,
    lengths: List[int] | None = None,
    rows_per_length: int = 128,
    anomaly_frac: float = 0.05,
    seed: int = 0,
) -> tuple[GroupsType, GroundTruthType]:
    rng = np.random.default_rng(seed)

    if lengths is None:
        base_lengths = np.array(
            [256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
            dtype=int,
        )
        lengths = [int(L) for L in base_lengths if L <= max_length]

    groups: GroupsType = {}
    ground_truth: GroundTruthType = {}

    uid_counter = 0

    for L in sorted(lengths):
        n_rows = rows_per_length

        specs = np.empty((n_rows, L), dtype=np.float64)
        freqs = np.empty((n_rows, L), dtype=np.float64)
        atm_interfs: List[List[Tuple[int, int]]] = []

        uid = np.arange(uid_counter, uid_counter + n_rows, dtype=np.int64)
        uid_counter += n_rows


        ref = np.array(["SYNTH_REF"] * n_rows, dtype=object)
        ant = np.array(["SYNTH_ANT"] * n_rows, dtype=object)
        pol = np.array(["SYNTH_POL"] * n_rows, dtype=object)

        gt = np.full((n_rows, 2), fill_value=-1, dtype=np.int64)


        n_anom = max(1, int(round(anomaly_frac * n_rows)))
        anom_rows = set(rng.choice(n_rows, size=n_anom, replace=False).tolist())

        freq_step = ref_freq * 16.0 / float(L)
        base_freq = 1.0

        freq_grid = base_freq + np.arange(L, dtype=np.float64) * freq_step

        for row_idx in range(n_rows):
            freqs[row_idx, :] = freq_grid
            atm_interfs.append([])

            i = np.arange(L, dtype=np.float64)
            t = i / L

            offset = rng.normal(0.0, 0.5)
            slope = rng.normal(0.0, 0.5)
            curv = rng.normal(0.0, 0.2)

            baseline = offset + slope * t + curv * (t - 0.5) ** 2

            noise = rng.normal(0.0, 1.0, size=L)
            x = baseline + noise

            if row_idx in anom_rows:
                m_min = max(2, L // 64)
                m_max = max(m_min, L // 16)
                m = int(rng.integers(m_min, m_max + 1))

                margin = max(L // 8, m)
                start_lo = max(0, margin)
                start_hi = min(L - margin - m, L - m)
                if start_hi <= start_lo:
                    start = max(0, (L - m) // 2)
                else:
                    start = int(rng.integers(start_lo, start_hi + 1))
                end = start + m - 1

                if rng.random() < 0.5:
                    amp = rng.uniform(3.0, 5.0)
                    x[start:end + 1] = baseline[start:end + 1] + amp + noise[start:end + 1]
                else:
                    amp = rng.uniform(2.0, 4.0)
                    center = 0.5 * (start + end)
                    idx_seg = np.arange(start, end + 1, dtype=np.float64)
                    shape = (idx_seg - center) / max(1.0, 0.5 * m)
                    x[start:end + 1] = baseline[start:end + 1] + amp * shape + noise[start:end + 1]

                gt[row_idx, 0] = start
                gt[row_idx, 1] = end

            specs[row_idx, :] = x

        groups[L] = (specs, uid, ref, ant, pol, freqs, atm_interfs)
        ground_truth[L] = gt

    return groups, ground_truth

def plot_synthetic_groups(
    groups,
    ground_truth,
    out_root: str = "SyntheticPlots",
    n_no_anom: int = 5,
    n_with_anom: int = 5,
    seed: int = 0,
) -> None:
    os.makedirs(out_root, exist_ok=True)
    rng = np.random.default_rng(seed)

    for L in sorted(groups.keys()):
        specs, uid, ref, ant, pol, freqs, atm_interfs = groups[L]
        gt = ground_truth[L]
        n_rows, n_ch = specs.shape

        anom_mask = gt[:, 0] >= 0
        idx_with = np.where(anom_mask)[0]
        idx_without = np.where(~anom_mask)[0]

        if len(idx_with) == 0 or len(idx_without) == 0:
            continue

        if len(idx_with) > n_with_anom:
            idx_with = rng.choice(idx_with, size=n_with_anom, replace=False)
        if len(idx_without) > n_no_anom:
            idx_without = rng.choice(idx_without, size=n_no_anom, replace=False)

        max_rows = max(len(idx_without), len(idx_with))
        max_rows = min(max_rows, max(n_no_anom, n_with_anom))

        fig, axes = plt.subplots(
            max_rows, 2,
            figsize=(12, 2.5 * max_rows),
            squeeze=False,
            sharex=False,
            sharey=False,
        )

        for row_i in range(max_rows):
            if row_i < len(idx_without):
                r = idx_without[row_i]
                ax = axes[row_i, 0]
                x = specs[r]
                f = freqs[r]

                ax.plot(f, x, color="C0")
                ax.set_title(f"L={L}, row={r} (no anomaly)")
                ax.set_ylabel("Amp")
            else:
                axes[row_i, 0].axis("off")

            if row_i < len(idx_with):
                r = idx_with[row_i]
                ax = axes[row_i, 1]
                x = specs[r]
                f = freqs[r]
                s_gt, e_gt = gt[r]

                ax.plot(f, x, color="C0")
                if 0 <= s_gt < n_ch and 0 <= e_gt < n_ch and e_gt >= s_gt:
                    ax.axvspan(
                        float(f[s_gt]),
                        float(f[e_gt]),
                        color="C1",
                        alpha=0.3,
                        label="Injected anomaly",
                    )
                ax.set_title(f"L={L}, row={r} (anomaly)")
                ax.set_ylabel("Amp")
            else:
                axes[row_i, 1].axis("off")

        for ax in axes[-1, :]:
            ax.set_xlabel("Frequency (GHz)")

        fig.suptitle(f"Synthetic spectra for length L={L}", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = os.path.join(out_root, f"synthetic_L{L}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def run_scan_for_kernel(
    groups: GroupsType,
    ground_truth: GroundTruthType,
    kernel_kind: str = "gaussian",
    buffer: int = 0,
    max_workers: int | None = None,
) -> tuple[float, float]:
    set_kernel_kind(kernel_kind)  # controls behavior inside scan_row 

    total_time = 0.0
    total_rows = 0
    total_anom = 0
    total_tp = 0

    print(f"\n=== Running kernel_kind={kernel_kind!r} ===")

    for L in sorted(groups.keys()):
        specs, uid, ref, ant, pol, freqs, atm_interfs = groups[L]
        n_rows, n_ch = specs.shape
        gt = ground_truth[L]
        assert gt.shape[0] == n_rows

        t2 = time.perf_counter()
        (
            windows_masked,
            scores_masked,
            sra_preds,
            sri_idxs_m,
            sri_vals_m,
            ws,
            range_caps,
            windows_unmasked,
            scores_unmasked,
            overlap_unm_pct,
            sri_idxs_unm,
            sri_vals_unm,
            windows_fixed,
            scores_fixed,
            overlap_fix_pct,
            sri_idxs_fix,
            sri_vals_fix,
            fixed_bins_nat,
        ) = polynomial_scan_ranges_parallel(
            spec_arrays=specs,
            score_fn=scan_row,
            atm_interfs=atm_interfs,
            freq_arrays=freqs,
            buffer=buffer,
            sr_factor=1,         # no superresolution for this synthetic benchmark :contentReference[oaicite:10]{index=10}
            max_workers=max_workers,
        )
        t3 = time.perf_counter()
        elapsed = t3 - t2

        total_time += elapsed
        total_rows += n_rows

        n_anom = int((gt[:, 0] >= 0).sum())
        tp = 0
        if n_anom > 0:
            for i in range(n_rows):
                s_gt, e_gt = gt[i]
                if s_gt < 0:
                    continue
                s_pred, e_pred = windows_unmasked[i]
                if e_pred <= s_pred:
                    continue
                inter = max(0, min(e_gt, e_pred) - max(s_gt, s_pred) + 1)
                union = (e_gt - s_gt + 1) + (e_pred - s_pred + 1) - inter
                iou = inter / union if union > 0 else 0.0
                if iou >= 0.5:
                    tp += 1

        total_anom += n_anom
        total_tp += tp
        hit_rate = (tp / n_anom) if n_anom > 0 else float("nan")

        print(
            f"{kernel_kind:8s} | L={L:4d} | rows={n_rows:4d} | "
            f"time={elapsed:7.3f}s | anomalies={n_anom:3d} | hit@IoU≥0.5={hit_rate:5.2f}"
        )

    global_hit = (total_tp / total_anom) if total_anom > 0 else float("nan")
    print(
        f"\n{kernel_kind:8s} total: time={total_time:7.3f}s over "
        f"{total_rows} rows | global hit@IoU≥0.5={global_hit:5.2f}"
    )
    return total_time, global_hit


def main() -> None:
    groups, ground_truth = generate_synthetic_groups(
        max_length=4096,
        lengths=None,
        rows_per_length=128,
        anomaly_frac=0.05,
        seed=0,
    )

    plot_synthetic_groups(
        groups,
        ground_truth,
        out_root="Images/SyntheticPlots",
        n_no_anom=5,
        n_with_anom=5,
        seed=0,
    )

    # warmup_numba_and_caches(groups, kinds=("laplace", "gaussian"), sample_per_length=1, small_n=128)

    # gaussian_time, gaussian_hit = run_scan_for_kernel(
    #     groups,
    #     ground_truth,
    #     kernel_kind="gaussian",
    #     buffer=0,
    #     max_workers=None,
    # )
    # laplace_time, laplace_hit = run_scan_for_kernel(
    #     groups,
    #     ground_truth,
    #     kernel_kind="laplace",
    #     buffer=0,
    #     max_workers=None,
    # )

    # print("\n=== Summary ===")
    # print(f"Gaussian: total_time={gaussian_time:7.3f}s, global hit@IoU≥0.5={gaussian_hit:5.2f}")
    # print(f"Laplace : total_time={laplace_time:7.3f}s, global hit@IoU≥0.5={laplace_hit:5.2f}")


if __name__ == "__main__":
    main()
