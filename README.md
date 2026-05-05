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
│       └── faults/          # Leak and sensor fault injectors 
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

faults:
  leaks: []                   # list of LeakSpec entries (see below)
  sensor_faults: []           # placeholder until Phase 4

validation:
  pressure_min_m: 0.0
  pressure_min_warning_tolerance_m: 1.0   # below this -> fail
  pressure_max_m: 150.0
  mass_balance_tol_m3s: 1.0e-3
  leak_pressure_drop_min_m: 0.01          # warning floor for leak-aware drop check

output:
  directory: outputs
  formats: [parquet, csv]     # one or both
  write_metadata_sidecar: true
```

### Demand modes

- `default` — leave the .inp file's patterns untouched.
- `scale` — multiply every junction's base demand by `scale.multiplier`.
- `fourier` — replace patterns with a synthetic diurnal pattern: `m(t) = base + amplitude * sin(2π t / period + phase) + N(0, noise_std)`. Multipliers are clipped at zero. Deterministic given the seed.

Adding a new strategy means subclassing `DemandStrategy` in `demand.py` and registering it in `STRATEGIES`.

### Output writers

`parquet` and `csv` ship out of the box. Each writer subclasses `Writer` in `output.py` and is registered in `WRITERS`. Adding DuckDB later (planned) is purely additive: a new subclass and registry entry, no other code changes.

### Validation severity

Three baseline checks run on every scenario, plus one extra for leak scenarios:

| Check | OK | WARNING | FAIL |
|---|---|---|---|
| `finite_values` | no NaN / inf | — | any NaN or inf |
| `pressure_bounds` | within strict range | within tolerance band | beyond tolerance |
| `mass_balance` | residual ≤ tol | — | residual > tol |
| `leak_pressure_drop` (leak only) | drop ≥ floor at every leak | drop below floor or missing baseline | — |

`WARNING` does not cause a non-zero exit code. The motivating example: Net3 produces a small negative pressure (~-0.66 m) at node `10` due to its known elevation/tank-cycle quirk. Both `WNTRSimulator` and the `EpanetSimulator` reference produce this; the validator surfaces it as a warning instead of failing the run. Leak scenarios amplify this dip slightly under PDD + leak conditions, so the `leak_abrupt_net3` config raises its `pressure_min_warning_tolerance_m` to 2 m.

The mass-balance residual subtracts `leak_demand` from the demand side. Without that correction every leak scenario would fail by exactly the leak outflow at every active step.

### Leak scenarios

Add a list of `LeakSpec` entries under `faults.leaks`. Leaks require `simulation.demand_model: PDD` (D16) — the config validator rejects `DDA + leaks` outright with no silent auto-promotion.

```yaml
simulation:
  demand_model: PDD           # required for any leak

faults:
  leaks:
    - pipe: "40"              # explicit pipe name (or omit for random)
      split_fraction: 0.5     # 0..1 along the pipe (or omit for random)
      area_m2: 0.005          # OR diameter_m: 0.05 (mutually exclusive)
      discharge_coeff: 0.75   # WNTR default
      start_time_seconds: 21600
      end_time_seconds: 64800
      profile: abrupt         # abrupt | linear | step
      profile_steps: 30       # only used by linear/step (default 30)
      name: midday_central_leak
```

#### Profile semantics

- `abrupt`: leak is fully active for the entire `[start_time, end_time]` window.
- `linear`: area grows from `target_area / n` to `target_area` over `n = profile_steps` step controls. The default `n` is 30 — fine enough that the staircase reads as a smooth ramp at typical 1-h report timesteps.
- `step`: same step-function rendering as `linear`; provided as a user-facing label so explicit "staircase" leaks remain semantically distinct in configs and metadata.

#### Random vs explicit

- Either `pipe` or `split_fraction` (or both) may be `None`; the runner draws values from `numpy.random.default_rng(seed)`. Random draws are reproducible: the same seed always selects the same pipe and fraction.
- Provide exactly one of `area_m2` or `diameter_m` per leak. Diameters are converted internally via `area = π * (d/2)²`.
- Multiple concurrent leaks are supported. Each leak inserts its own junction via `wntr.morph.split_pipe`; concurrent leaks may overlap in time.

#### Example configs

| Config | What it shows |
|---|---|
| `configs/leak_abrupt_net3.yaml` | Single explicit abrupt leak, 24 h Net3 with PDD, midday leak window |
| `configs/leak_incipient_hanoi.yaml` | Linear-profile leak on Hanoi with Fourier demand, 30-step ramp |
| `configs/leak_multi_jilin.yaml` | Two concurrent leaks (one abrupt, one incipient) on Jilin |
| `configs/leak_random_fowm.yaml` | Fully random pipe + split fraction, demonstrating seed-driven reproducibility on FOWM |

---

## Running Tests

```bash
pytest
```

Currently 75 tests covering every module: config schema (with leak-spec validators), network loading, demand strategies, simulation, leak injection (abrupt + linear), labelling, validation severity (including leak-aware mass balance and pressure drop), output writers, and end-to-end runner.

---

## Generating the Phase 3 Plots

After at least one successful pipeline run:

```bash
python scripts/generate_phase3_plots.py
```

Outputs to `outputs/plots/`:

- `pressure_timeseries_{network}.png` and `flow_timeseries_{network}.png` for FOWM, Jilin and Hanoi normal baselines.
- `leak_pressure_drop_net3.png` — pressure at the leak node and 3 nearby nodes, baseline vs leak.
- `leak_demand_profile_net3.png` — abrupt leak demand profile (zero outside the window, constant during).
- `leak_incipient_profile_hanoi.png` — linear-profile leak demand showing the staircase rise.
- `leak_pressure_heatmap_jilin.png` — every node's pressure over time for the multi-leak scenario, with leak onsets marked.
- `leak_multi_demand_jilin.png` — both concurrent leaks' demand profiles overlaid.
- `leak_residual_net3.png` — heatmap of `baseline − leak` pressure to isolate the leak's effect from the diurnal pattern.
- `leak_residual_timeseries_net3.png` — line-plot version of the residual at nearby nodes.
- `leak_determinism_fowm.png` — pressure residual between two identical leak runs (~1e-12 m, Newton noise floor).
- `validation_summary_phase3.txt` — full severity report and key metrics for every config.

---

## Status

### Phase 1: Setup and exploration (Weeks 1) — complete

- Environment setup (Python 3.12, venv, dependencies, repo structure)
- Reference repos cloned (DiTEC-WDN, LeakG3PD)
- WNTR basic experiments
- Findings recorded in `notebooks/01_wntr_basics.ipynb`

### Phase 2: Pipeline architecture and normal scenarios (Week 2) — complete

- Pydantic v2 config schema
- WaterNetworkModel loader with state isolation
- Pluggable demand strategies
- WNTRSimulator wrapper
- Per-timestep labels + sidecar metadata YAML
- PASS/WARNING/FAIL severity validation with finite, pressure-bound and mass-balance checks
- Extensible Parquet/CSV writers
- Typer CLI
- End-to-end working on Net3 (DDA) and Hanoi (Fourier demand)

### Phase 3: pipe leak injection (Week 3)

- Typed `LeakSpec` config model with mutual-exclusion (area xor diameter), seed-driven random pipe and split-fraction selection
- Abrupt + linear/step incipient profiles via WNTR controls
- Explicit PDD requirement enforced at config-load time
- `leak_demand` exposed as a fourth output table (parquet + csv)
- Leak-aware mass balance, leak pressure-drop check, leak labels in metadata
- 75 pytest tests, deterministic to the WNTR Newton noise floor
- Six new configs: normal + leak scenarios on Net3, Hanoi, Jilin and FOWM

### Upcoming

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
