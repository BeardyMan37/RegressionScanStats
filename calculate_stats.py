from __future__ import annotations
import argparse, logging, time, os
from typing import Dict
from cosmicai.config import set_kernel_kind, get_kernel_kind
from cosmicai.io_preprocess import load_data_by_length
from cosmicai.warmup import warmup_numba_and_caches
from cosmicai.scan import scan_row_with_nwkr
from cosmicai.parallel_exec import polynomial_scan_ranges_parallel
from cosmicai.superres import sr_factor, superresolve_ranges, superresolve, refine_all_windows_exact_for_length
from cosmicai.viz import plot_top_k

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan spectrograms for best windows using NWKR.")
    p.add_argument("--data-path", required=True)
    p.add_argument("--interference-path", required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--per-fig", type=int, default=10)
    p.add_argument("--buffer-coeff", type=int, default=20)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-root", default="images/latest_run")
    p.add_argument("--data-root", default="data/latest_run")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--kernel-kind", default="gaussian", choices=["gaussian","laplace","laplace_rt"])
    return p

def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    set_kernel_kind(args.kernel_kind)

    t0 = time.perf_counter()
    df, groups = load_data_by_length(args.data_path, args.interference_path)
    t1 = time.perf_counter()
    logging.info("Loaded & grouped data in %.3fs", (t1 - t0))
    logging.info("Found lengths: %s", sorted(groups.keys()))

    length_SR_FACTOR_map = {L: sr_factor(L) for L in sorted(groups.keys())}

    warmup_numba_and_caches(groups, kinds=("laplace","gaussian"), sample_per_length=1, small_n=128)

    for length in sorted(groups):
        BUFFER = length // args.buffer_coeff
        actual_specs, uid, ref, ant, pol, freqs, atm_interfs, flag_ranges = groups[length]
        n_rows, row_len = actual_specs.shape
        logging.info("Before Preprocessing: Length=%d: %d rows, %d channels", length, n_rows, row_len)

        SR_FACTOR = length_SR_FACTOR_map.get(length, 1)
        atm_interfs_sr = superresolve_ranges(atm_interfs, factor=SR_FACTOR)
        flag_ranges_sr = superresolve_ranges(flag_ranges, factor=SR_FACTOR)
        actual_specs_sr = superresolve(actual_specs, factor=SR_FACTOR)
        freqs_sr = superresolve(freqs, factor=SR_FACTOR)

        logging.info("After Preprocessing: Length=%d: %d rows, %d channels, SR_factor %d",
                     length, actual_specs_sr.shape[0], actual_specs_sr.shape[1], SR_FACTOR)

        t2 = time.perf_counter()
        (windows_sr_masked, scores_masked, sra_preds, sri_idxs_m, sri_vals_m, ws, range_caps,
         windows_sr_unmasked, scores_unmasked, overlap_unm_pct, sri_idxs_unm, sri_vals_unm,
         windows_sr_fixed, scores_fixed, overlap_fix_pct, sri_idxs_fix, sri_vals_fix, fixed_bins_nat
        ) = polynomial_scan_ranges_parallel(
            spec_arrays=actual_specs_sr,
            score_fn=scan_row_with_nwkr,
            atm_interfs=atm_interfs_sr,
            flag_ranges=flag_ranges_sr,
            freq_arrays=freqs_sr,
            buffer=BUFFER // SR_FACTOR,
            sr_factor=SR_FACTOR,
            max_workers=args.workers,
        )
        t3 = time.perf_counter()
        logging.info("  Scan time: %.3fs", (t3 - t2))

        out_dir = os.path.join(args.out_root, f"length_{length}")
        data_dir = os.path.join(args.data_root, f"length_{length}")
        os.makedirs(out_dir, exist_ok=True); os.makedirs(data_dir, exist_ok=True)

        if SR_FACTOR > 1:
            windows_exact_masked, windows_exact_unmasked, windows_exact_fixed = refine_all_windows_exact_for_length(
                actual_specs, freqs, windows_sr_masked, windows_sr_unmasked, windows_sr_fixed,
                atm_interfs, ws, range_caps, SR_FACTOR, BUFFER)
        else:
            windows_exact_masked   = windows_sr_masked
            windows_exact_unmasked = windows_sr_unmasked
            windows_exact_fixed    = windows_sr_fixed

        meta = {"uid": uid, "ref": ref, "ant": ant, "pol": pol, "freq": freqs}
        plot_top_k(
            df=df,
            actual_spec_arrays=actual_specs,
            freqs=freqs,
            windows_masked=windows_exact_masked,
            windows_unmasked=windows_exact_unmasked,
            windows_fixed=windows_exact_fixed,
            scores_masked=scores_masked,
            scores_unmasked=scores_unmasked,
            scores_fixed=scores_fixed,
            overlap_unmasked_pct=overlap_unm_pct,
            overlap_fixed_pct=overlap_fix_pct,
            atm_interfs=atm_interfs,
            meta=meta,
            ws=ws,
            k=min(args.top_k, actual_specs.shape[0]),
            per_fig=args.per_fig,
            buffer=BUFFER,
            out_dir=out_dir,
            data_dir=data_dir,
            sra_preds=sra_preds,
            sri_idxs_masked=sri_idxs_m,
            sri_vals_masked=sri_vals_m,
            sri_idxs_unmasked=sri_idxs_unm,
            sri_vals_unmasked=sri_vals_unm,
            sri_idxs_fixed=sri_idxs_fix,
            sri_vals_fixed=sri_vals_fix,
            sr_factor=SR_FACTOR,
            fixed_bins_nat=fixed_bins_nat,
            rank_by="masked",
        )

if __name__ == "__main__":
    main()
