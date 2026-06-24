# ML-utility experiments (paper Section 5)

Evaluation code that **consumes** the generated WDN datasets to produce the
experimental evidence for the paper. It does not modify the simulation pipeline:
all data access goes through the public `wdn_pipeline.query.DatasetQuery` and the
PyTorch loader in `wdn_pipeline.ml`.

The experiments answer **RQ2** (is the generated data useful for training and
benchmarking ML anomaly detectors, and is it non-trivial?) and refresh the
supporting studies that answer **RQ1** (physical validity, breadth, scalability),
so every number in the paper comes from one consistent corpus and machine.

## Studies

- **Study 1 - physics validation:** mass-balance and Hazen-Williams energy
  residual distributions across the corpus.
- **Study 2 - network sweep:** every compatible LeakG3PD network run through the
  pipeline (`docs/network_sweep_report.md`).
- **Study 3 - parallel scaling:** batch speedup vs worker count
  (`docs/parallelisation_benchmark.md`).
- **Study 4 - downstream ML utility (primary):** three detectors (z-score
  baseline, LSTM forecaster, LSTM autoencoder) trained on normal data and
  evaluated on difficulty-swept leak and sensor faults, with threshold-free
  (AUC-PR) and event/range-aware metrics, detection delay and mandatory
  difficulty curves over leak area and sensor gain factor.
- **Study 5 - cross-dataset sanity vs LeakG3PD:** matched-Hanoi leaks compared
  against LeakG3PD's documented WNTR leak law.

## Dependencies

```bash
pip install -r experiments/ml_utility/requirements.txt   # scikit-learn; torch via [ml]
```

## Reproduce (run from the repo root, with `PYTHONPATH=.`)

```bash
# 1. Build the dedicated experiment dataset (15-min/24h, Net3+Hanoi, ~712 scenarios).
python experiments/ml_utility/generate_experiment_dataset.py --workers 4

# 2. Train + evaluate the detectors, write metrics.json and the difficulty figures.
python experiments/ml_utility/run_experiments.py --seeds 0 1 2

# 3. Cross-dataset sanity check vs LeakG3PD.
python experiments/ml_utility/study5_leakg3pd.py

# 4. Refresh Studies 1-3 (physics residuals, network sweep, parallel benchmark).
python experiments/ml_utility/refresh_supporting_studies.py

# 5. Assemble the consolidated results document.
python experiments/ml_utility/build_report.py        # writes docs/experiments_report.md

# Metric unit tests (also runnable under pytest).
python experiments/ml_utility/test_metrics.py
```

A fast smoke variant of the dataset (a handful of scenarios per type) is
available with `generate_experiment_dataset.py --quick`.

## Determinism

Every seed is fixed and recorded: the dataset master seed (in `manifest.json`),
the scenario-level train/eval split seed, and the per-detector numpy/torch
seeds (in `metrics.json`). Each detector is run over several seeds and the report
carries mean +/- std. CPU LSTM training has minor run-to-run nondeterminism from
threaded BLAS reductions, which is why multiple seeds are run.

## Layout

```
generate_experiment_dataset.py   dataset builder (configs + DuckDB + manifest.json)
detectors.py                     z-score / LSTM forecaster / LSTM autoencoder
metrics.py                       AUC-PR/ROC + range-aware P/R + PA-F1 + delay
test_metrics.py                  metric unit tests
run_experiments.py               Study 4 train/evaluate + difficulty figures
study5_leakg3pd.py               Study 5 cross-check
refresh_supporting_studies.py    Studies 1-3 refresh
build_report.py                  assembles docs/experiments_report.md
data/                            experiment.duckdb, manifest.json, metrics.json, study*.json  (gitignored)
figures/                         difficulty + residual + study5 figures  (gitignored)
configs/                         generated scenario configs  (gitignored)
```

Artefacts under `data/`, `figures/` and `configs/` are regenerable from the
scripts and are gitignored.
