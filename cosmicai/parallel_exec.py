from __future__ import annotations
from typing import Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from .warmup import worker_warmup

def polynomial_scan_ranges_parallel(spec_arrays, score_fn, atm_interfs, freq_arrays, buffer, sr_factor, max_workers=None):
    n_rows, _ = spec_arrays.shape
    params = [
        (i, spec_arrays[i], atm_interfs[i], freq_arrays[i], buffer, sr_factor)
        for i in range(n_rows)
    ]

    results: List[Tuple[Any, ...]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futs = [exe.submit(worker_warmup, k, 64) for k in ("laplace", "gaussian")]
        for _ in as_completed(futs):
            pass

        futures = [exe.submit(score_fn, p) for p in params]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x[0])
    
    # masked var-len
    windows_masked   = [r[1]  for r in results]
    scores_masked    = [r[2]  for r in results]
    sra_preds        = [r[3]  for r in results]
    sri_idxs_masked  = [r[4]  for r in results]
    sri_vals_masked  = [r[5]  for r in results]
    ws               = [r[6]  for r in results]
    range_caps       = [r[7]  for r in results]

    # unmasked var-len
    windows_unmasked = [r[8]  for r in results]
    scores_unmasked  = [r[9]  for r in results]
    overlap_pct_unm  = [r[10] for r in results]
    sri_idxs_unm     = [r[11] for r in results]
    sri_vals_unm     = [r[12] for r in results]

    # fixed-len linear
    windows_fixed    = [r[13] for r in results]
    scores_fixed     = [r[14] for r in results]
    overlap_pct_fix  = [r[15] for r in results]
    sri_idxs_fixed   = [r[16] for r in results]
    sri_vals_fixed   = [r[17] for r in results]
    fixed_bins_nat   = [r[18] for r in results]

    return (windows_masked, scores_masked, sra_preds, sri_idxs_masked, sri_vals_masked, ws, range_caps,
            windows_unmasked, scores_unmasked, overlap_pct_unm, sri_idxs_unm, sri_vals_unm,
            windows_fixed, scores_fixed, overlap_pct_fix, sri_idxs_fixed, sri_vals_fixed, fixed_bins_nat)
