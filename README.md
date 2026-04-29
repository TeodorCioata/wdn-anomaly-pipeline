# WDN Anomaly Pipeline

A modular, configuration-driven Python pipeline that wraps [EPANET/WNTR](https://usepa.github.io/WNTR/) to systematically generate large-scale, labelled time-series datasets for water distribution networks (WDNs). The pipeline covers normal baseline scenarios, pipe leaks (abrupt and incipient), sensor faults (stuck, drift, dropout, bias) and cumulative anomalies, producing reproducible, ML-ready datasets in Parquet and CSV formats.

**MSc internship project** — Teodor Cioata, MSc Computing Science, University of Groningen, 2026.
Supervisors: Dilek Düştegör, Alexander Lazovik, Samer Ahmed.

---

## Project Structure

```
wdn-anomaly-pipeline/
├── configs/          # YAML scenario configuration files
├── networks/         # EPANET .inp topology files
├── scripts/          # Helper scripts (e.g. plot generation)
├── src/
│   └── wdn_pipeline/
│       ├── config.py        # Pydantic v2 schema + YAML loader
│       ├── network.py       # WaterNetworkModel loader
│       ├── demand.py        # Pluggable demand-pattern strategies
│       ├── simulation.py    # WNTR simulation wrapper
│       ├── labelling.py     # Per-timestep labels + scenario metadata
│       ├── validation.py    # Physics-based validation (PASS/WARNING/FAIL)
│       ├── output.py        # Extensible writer interface (parquet, csv, ...)
│       ├── runner.py        # End-to-end orchestrator + Typer CLI
│       └── faults/          # Leak and sensor fault injectors (Phase 3-4)
├── notebooks/        # Jupyter exploration notebooks
├── tests/            # pytest unit and integration tests
├── outputs/          # Generated datasets and plots (gitignored)
└── docs/             # Additional documentation
```

---

## Setup

**Requirements:** Python 3.12, Git.

```bash
# 1. Clone the repo
git clone <repo-url>
cd wdn-anomaly-pipeline

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install the package and dev dependencies
pip install -e ".[dev]"

# 4. Verify WNTR is working
python -c "import wntr; print('wntr', wntr.__version__)"
```

---

## Running the Pipeline

### CLI

The package installs a `wdn-pipeline` console script (Typer-based, decision D8). Pass a YAML config and an optional `--verbose` flag:

```bash
wdn-pipeline configs/normal_net3.yaml
wdn-pipeline configs/normal_hanoi.yaml --verbose
```

### Python API

```python
from wdn_pipeline.runner import run_from_config_file

summary = run_from_config_file("configs/normal_net3.yaml")
print(summary.format())
print("output files:", summary.output_paths)
print("validation severity:", summary.validation.severity)
```

`summary.validation.severity` is one of `"ok"`, `"warning"`, `"fail"` (decision D12).

---

## Config Schema

A scenario is described by one YAML file. All fields shown below have sensible defaults except `network.inp_path` and `simulation.duration_seconds`.

```yaml
network:
  inp_path: Net3              # bundled WNTR name OR path to .inp file
  name: net3                  # short ID used in filenames

simulation:
  duration_seconds: 86400     # required
  hydraulic_timestep_seconds: 3600
  report_timestep_seconds: 3600
  pattern_timestep_seconds: 3600  # optional; defaults to .inp value
  demand_model: DDA           # DDA | PDD

seed: 42

demand:
  mode: default               # default | scale | fourier
  scale:
    multiplier: 1.0
  fourier:
    base: 1.0
    amplitude: 0.3
    period_hours: 24.0
    phase_shift_hours: 0.0
    noise_std: 0.0

scenario:
  type: normal                # normal | leak | sensor_fault | cumulative
  label: normal               # used in filenames + label column

validation:
  pressure_min_m: 0.0
  pressure_min_warning_tolerance_m: 1.0   # below this -> fail
  pressure_max_m: 150.0
  mass_balance_tol_m3s: 1.0e-3

output:
  directory: outputs
  formats: [parquet, csv]     # one or both
  write_metadata_sidecar: true
```

### Demand modes (decision D4)

- `default` — leave the .inp file's patterns untouched.
- `scale` — multiply every junction's base demand by `scale.multiplier`.
- `fourier` — replace patterns with a synthetic diurnal pattern: `m(t) = base + amplitude * sin(2π t / period + phase) + N(0, noise_std)`. Multipliers are clipped at zero. Deterministic given the seed.

Adding a new strategy means subclassing `DemandStrategy` in `demand.py` and registering it in `STRATEGIES`.

### Output writers (decision D6)

`parquet` and `csv` ship out of the box. Each writer subclasses `Writer` in `output.py` and is registered in `WRITERS`. Adding DuckDB later (planned) is purely additive: a new subclass and registry entry, no other code changes.

### Validation severity (decision D12)

Three checks run on every scenario:

| Check | OK | WARNING | FAIL |
|---|---|---|---|
| `finite_values` | no NaN / inf | — | any NaN or inf |
| `pressure_bounds` | within strict range | within tolerance band | beyond tolerance |
| `mass_balance` | residual ≤ tol | — | residual > tol |

`WARNING` does not cause a non-zero exit code. The motivating example: Net3 produces a small negative pressure (~-0.66 m) at node `10` due to its known elevation/tank-cycle quirk. Both `WNTRSimulator` and the `EpanetSimulator` reference produce this; the validator surfaces it as a warning instead of failing the run.

---

## Running Tests

```bash
pytest
```

Currently 50 tests covering every module: config schema, network loading, demand strategies, simulation, labelling, validation severity, output writers, and end-to-end runner.

---

## Generating the Phase 2 Plots

After at least one successful pipeline run:

```bash
python scripts/generate_phase2_plots.py
```

Outputs to `outputs/plots/`:

- `pressure_timeseries_net3.png` — diurnal cycle at five representative nodes.
- `flow_timeseries_net3.png` — flow at five representative pipes.
- `pressure_heatmap_net3.png` — every node's pressure over time.
- `determinism_residual_net3.png` — pressure residual between two pipeline runs on the same config; bounded at the numerical noise floor (~1e-14 m).
- `validation_summary.txt` — full severity report for the example configs.

---

## Status

### Phase 1: Setup and exploration (Weeks 1-2) — complete

- Environment setup (Python 3.12, venv, dependencies, repo structure)
- Reference repos cloned (DiTEC-WDN, LeakG3PD)
- WNTR basic experiments
- Findings recorded in `notebooks/01_wntr_basics.ipynb`

### Phase 2: Pipeline architecture and normal scenarios (Week 3) — complete

- Pydantic v2 config schema (D1, D2)
- WaterNetworkModel loader with state isolation
- Pluggable demand strategies (D4)
- WNTRSimulator wrapper (D3)
- Per-timestep labels + sidecar metadata YAML (D10)
- PASS/WARNING/FAIL severity validation (D12) with finite, pressure-bound and mass-balance checks
- Extensible Parquet/CSV writers (D6)
- Typer CLI (D8)
- 50 pytest tests, deterministic to ~1e-14 m
- End-to-end working on Net3 (DDA) and Hanoi (Fourier demand)

### Upcoming

- Phase 3: pipe leak injection (abrupt + incipient)
- Phase 4: sensor fault models (bias, drift, stuck, dropout)
- Phase 5: dataset organisation and DuckDB queryable export
- Phase 6: report writing

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `wntr` | EPANET/WNTR hydraulic simulation |
| `pandas` + `pyarrow` | Tabular data and Parquet serialisation |
| `pydantic` | Config schema validation |
| `pyyaml` | YAML config loading |
| `typer` | CLI |
| `matplotlib` | Visualisation |
| `duckdb` | Queryable dataset exports (planned) |
| `pytest` + `ruff` | Testing and linting |
