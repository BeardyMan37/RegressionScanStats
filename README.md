# RegressionScanStats

Automated spectral line detection in ALMA bandpass calibration data using Nadaraya-Watson Kernel Regression (NWKR) scan statistics.

---

## Installation

```bash
git clone https://github.com/BeardyMan37/RegressionScanStats.git
cd RegressionScanStats
conda env create -f environment.yml
conda activate RegressionScanStats
```

---

## Repository Structure

```
RegressionScanStats/
├── helpers/                        # Core library
│   ├── config.py                    # Global config (kernel kind, ref_freq)
│   ├── scan.py                      # NWKR scan row (optimized + naive)
│   ├── kernels.py                   # Gaussian and Laplace kernel vectors
│   ├── kernel_optimized_state.py    # Incremental O(r) state updates
│   ├── scoring.py                   # SRA/SRI computation, epidemic score
│   ├── superres.py                  # Superresolution + window refinement
│   ├── regressors.py                # Mean/poly/KRR regression baselines
│   ├── predictors.py                # Truncated kernel prediction
│   ├── parallel_exec.py             # Parallelised scan over spectrum groups
│   ├── io_preprocess.py             # Data loading and grouping by length
│   ├── warmup.py                    # Numba/cache warmup
│   └── viz.py                       # Plotting utilities
│
├── benchmark_regressor_on_syth.py   # Benchmarking harness (runtime + parameter)
├── score_experiment.py              # Parallelised real-data scoring experiment
├── gen_synth_data.py                # Synthetic dataset generation (i.i.d.)
├── gen_ar2_data.py                  # AR(2) correlated background generation
├── create_dataset.py                # Labelled real dataset construction
├── scan_statistics.py               # CLI entry point for production scanning
├── calculate_stats.py               # Score/window statistics computation
├── merge_stats.py                   # Merge results across spectrum groups
├── infer_perpol_labels.py           # Per-polarisation label inference
├── make_score_plot.py               # Score distribution plotting
│
├── explore_data.ipynb               # Data exploration
├── explore_scan_stat.ipynb          # Scan statistic exploration
├── filter_data.ipynb                # Data filtering
├── gen_synth_data.ipynb             # Synthetic data exploration
├── calc_window_score.ipynb          # Window score calculation
│
├── utils/                           # Utility scripts
├── Kats/                            # Kats detector integration
└── environment.yml                  # Conda environment
```

---

## Running the Scanner

```bash
python scan_statistics.py \
    --data-path data/my_spectra.parquet \
    --interference-path data/atmospheric_transmission.parquet \
    --kernel-kind gaussian \
    --out-root images/latest_run \
    --data-root data/latest_run
```

| Argument | Description |
|----------|-------------|
| `--data-path` | Path to raw spectra parquet |
| `--interference-path` | Path to atmospheric transmission parquet |
| `--kernel-kind` | `gaussian` or `laplace` (default: `gaussian`) |
| `--out-root` | Output directory for plots |
| `--data-root` | Output directory for result parquets |
| `--workers` | Number of parallel workers (default: all cores) |
| `--buffer-coeff` | Buffer coefficient for edge trimming (default: 20) |

---

## Building the Labelled Dataset

```bash
python create_dataset.py \
    --parquet data/qa2_raw.parquet \
    --out data/qa2_labelled_dataset.parquet
```

---

## Benchmarking

### Runtime scaling

```bash
# vs spectrum length n
python benchmark_regressor_on_syth.py runtime --variable n

# vs window cap w
python benchmark_regressor_on_syth.py runtime --variable w --n 1000

# vs truncation radius r
python benchmark_regressor_on_syth.py runtime --variable r --n 1000
```

### Detection quality (AR(2) synthetic data)

```bash
# SNR sweep
python benchmark_regressor_on_syth.py parameter \
    --variable snr \
    --w 25 \
    --snr 1.0 2.0 3.0 4.0 5.0 \
    --noise 0.05 \
    --methods mean,poly_deg1,poly_deg2,nwkr_gaussian,nwkr_laplace \
    --out-dir data/parameter_benchmark_ar2

# Anomaly width sweep
python benchmark_regressor_on_syth.py parameter \
    --variable noise \
    --w 25 \
    --snr 2.5 \
    --noise 0.01 0.02 0.05 0.10 0.20 \
    --methods mean,poly_deg1,poly_deg2,nwkr_gaussian,nwkr_laplace \
    --out-dir data/parameter_benchmark_ar2
```

Available method keys: `mean`, `poly_deg1`, `poly_deg2`, `nwkr_gaussian`, `nwkr_laplace`, `nwkr_gaussian_naive`, `nwkr_laplace_naive`, `lrt`, `bocpd`, `capa`, `stumpy`, `ruptures_kernelcpd`

---

## Real Data Evaluation

### Parallelised scoring experiment

```bash
python score_experiment.py \
    --parquet data/qa2_labelled_dataset.parquet \
    --methods mean,poly_deg1,poly_deg2,nwkr_gaussian,nwkr_laplace,lrt,capa \
    --max-rows 1000 \
    --workers 16 \
    --out-dir data/score_experiment
```

> Note: `stumpy` cannot run in subprocesses due to numba/OpenMP conflicts. Exclude it from `--methods` and run it separately or use the serial fallback in the script.

### IoU gridsearch

Run from a notebook after the scoring experiment:

```python
from benchmark_regressor_on_syth import run_iou_gridsearch, ALL_METHODS, _resolve_methods

families = _resolve_methods("mean,poly_deg1,nwkr_gaussian,nwkr_laplace", [])
run_iou_gridsearch(
    parquet_path = "data/qa2_labelled_dataset.parquet",
    families     = families,
    iou_values   = [round(v, 2) for v in np.arange(0.50, 1.01, 0.05)],
    max_rows     = 1000,
    out_dir      = "data/real_data_benchmark",
)
```

### Score threshold gridsearch

```python
from benchmark_regressor_on_syth import run_score_threshold_gridsearch

run_score_threshold_gridsearch(
    results_path = "data/score_experiment/score_experiment_results.csv",
    score_values = [round(v, 2) for v in np.arange(0.30, 1.01, 0.05)],
    iou_thresh   = 0.75,
    out_dir      = "data/score_experiment",
)
```
