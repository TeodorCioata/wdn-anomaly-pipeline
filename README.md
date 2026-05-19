# WDN Anomaly Pipeline

A modular, configuration-driven Python pipeline that wraps [EPANET/WNTR](https://usepa.github.io/WNTR/) to systematically generate large-scale, labelled time-series datasets for water distribution networks (WDNs). The pipeline covers normal baseline scenarios, pipe leaks (abrupt and incipient), sensor faults (bias, drift, stuck, dropout, noise, gain) and cumulative anomalies (leak plus sensor faults in one scenario), producing reproducible, ML-ready datasets in Parquet and CSV formats.

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
│       ├── postprocess.py   # Optional leak-node cleanup (D28)
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
  sensor_faults: []           # list of SensorFaultSpec entries (see below)

validation:
  pressure_min_m: 0.0
  pressure_min_warning_tolerance_m: 1.0   # below this -> fail
  pressure_max_m: 150.0
  mass_balance_tol_m3s: 1.0e-3
  leak_pressure_drop_min_m: 0.01          # warning floor for leak-aware drop check
  gain_detectability_min_ratio: 0.05      # gain-fault detectability floor (D27)

output:
  directory: outputs
  formats: [parquet, csv]     # one or both
  write_metadata_sidecar: true
  remove_leak_nodes: false    # D28: revert leak-node split artefacts on output
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
| `leak_demand_active` (leak only) | leak_demand > 0 in `[start, end)` and exactly 0 outside | — | mismatch (window or solver bug) |
| `leak_pressure_drop` (leak only) | drop ≥ floor at every leak | drop below floor (diurnal-confound, diagnostic) | — |

`WARNING` does not cause a non-zero exit code. The motivating example: Net3 produces a small negative pressure (~-0.66 m) at node `10` due to its known elevation/tank-cycle quirk. Both `WNTRSimulator` and the `EpanetSimulator` reference produce this; the validator surfaces it as a warning instead of failing the run. Leak scenarios amplify this dip slightly under PDD + leak conditions, so the `leak_abrupt_net3` config raises its `pressure_min_warning_tolerance_m` to 2 m.

The mass-balance residual subtracts `leak_demand` from the demand side. Without that correction every leak scenario would fail by exactly the leak outflow at every active step.

Per-timestep labels and the `leak_demand_active` check both use a **half-open active window** `[start_time, end_time)`. WNTR's end control flips `leak_status=False` at `end_time_seconds`, so the reported `leak_demand` at that exact step is already zero by design. Including the end timestep would mark a normal frame as anomalous; the `leak_demand_active` validator polices this invariant and fails if the label disagrees with the simulator.

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

### Sensor fault scenarios (Phase 4 Week 4)

Sensor faults corrupt the reported pressure or flowrate at a single channel **after** the hydraulic simulation has run (decision D5). The simulator output is preserved verbatim in `pressure_clean` / `flowrate_clean` so downstream consumers always have the uncorrupted ground truth.

```yaml
faults:
  sensor_faults:
    - type: bias              # bias | drift | stuck | dropout | noise | gain
      quantity: pressure      # pressure (junction) | flowrate (link)
      target: "15"            # explicit name OR null for random
      start_time_seconds: 21600
      end_time_seconds: 64800
      name: midday_bias_node_15

      # Per-type fields. Provide the ones for the chosen `type`:
      bias_value: 2.0         # required for type=bias (non-zero)
      slope_per_second: 0.001 # required for type=drift, units/sec
      intervals:              # required for type=dropout
        - [21600, 28800]      # half-open sub-windows inside [start, end)
      fill_value: null        # dropout: null = NaN, else numeric
      sigma: 1.0              # required for type=noise (> 0)
      rng_offset: 0           # optional per-fault offset for noise RNG
      gain_factor: 1.10       # required for type=gain (not 1.0 or 0.0)
```

#### Fault formulas

For true reading `y(t)` over the half-open window `[start, end)`:

| Fault | Formula |
|---|---|
| `bias` | `y'(t) = y(t) + bias_value` |
| `drift` | `y'(t) = y(t) + slope_per_second * (t - start)` |
| `stuck` | `y'(t) = y(start)` |
| `dropout` | `y'(t) = NaN` (or `fill_value`) for `t` in each sub-interval; samples outside the sub-intervals are unchanged |
| `noise` | `y'(t) = y(t) + epsilon`, `epsilon ~ N(0, sigma^2)` |
| `gain` | `y'(t) = gain_factor * y(t)` |

#### Gain fault detectability (D27)

A gain factor very close to `1.0` corrupts low-variance channels by an amount comparable to ordinary sensor noise, which would mislabel ML training data. After applying the gain the `sensor_fault_signal_applied` check compares the residual standard deviation to the clean-signal standard deviation; below `validation.gain_detectability_min_ratio` (default `0.05`) it emits a **warning** (never a hard failure). The config validator also rejects `gain_factor` of exactly `1.0` (a no-op) or `0.0` (zeroes the signal).

#### Random target selection (D19)

Set `target: null` to draw a target from the scenario RNG. Pressure faults sample from `wn.junction_name_list`; flowrate faults sample from `wn.pipe_name_list`. Random draws are reproducible: the same seed always picks the same target.

#### Output layout for sensor scenarios

- `*_pressure` and `*_flowrate` tables: corrupted signals, with the per-channel mask columns `{fault_type}_mask_{target}` appended (decision D26).
- `*_pressure_clean` and `*_flowrate_clean` tables: uncorrupted simulator output, identical to the corrupted siblings when no sensor faults run.
- The sidecar metadata YAML records every resolved sensor fault under `fault_summary.sensor_faults` (long-form event list, source of truth per D22).
- Per-timestep `label` is the **union** of every active leak + sensor fault window.

#### Validation (D23)

Two new structural checks run for any scenario that includes sensor faults:

| Check | OK | FAIL |
|---|---|---|
| `sensor_fault_mask_consistent` | per-channel mask covers exactly `[start, end)` (or, for dropout, the sub-intervals) | mask disagrees with the spec at any timestep |
| `sensor_fault_signal_applied` | corrupted minus clean matches the fault formula to numerical tolerance (for noise: residual mean within `4*sigma/sqrt(n)` and std within 30% of sigma) | residual deviates from the spec |

`sensor_fault_signal_applied` can also emit a `warning` for the gain detectability check (D27) without failing.

#### Example sensor configs

| Config | Fault | What it shows |
|---|---|---|
| `configs/sensor_bias_net3.yaml` | bias | +2 m offset on junction 15 between 06:00 and 18:00 |
| `configs/sensor_drift_net3.yaml` | drift | 0.001 m/s linear ramp on junction 15 between 04:00 and 20:00 |
| `configs/sensor_stuck_net3.yaml` | stuck | Junction 15 frozen at 08:00 value through the end of the day |
| `configs/sensor_dropout_net3.yaml` | dropout | Three NaN-filled 2-hour gaps on junction 15 |
| `configs/sensor_noise_net3.yaml` | noise | sigma=1 m Gaussian noise on junction 15, full-day |
| `configs/sensor_gain_net3.yaml` | gain | +10% multiplicative gain on junction 15 between 06:00 and 18:00 |

### Cumulative scenarios (Phase 4 Week 5)

A cumulative scenario carries **both** `faults.leaks` and `faults.sensor_faults`. The two fault families compose through the existing runner flow with no special-case orchestration: leaks modify the hydraulic model before simulation, sensor faults corrupt the DataFrames afterwards. Set `scenario.type: cumulative` and populate both lists.

- The per-timestep `label` is the union of every leak and sensor-fault window; per-fault-type masks stay as separate columns.
- The cumulative validator runs the leak-specific checks (`leak_demand_active`, `leak_pressure_drop`) and the sensor-fault structural checks together. The hydraulic checks read the uncorrupted `*_clean` frames, so a sensor fault on the leak junction cannot mask the leak's hydraulic signature.
- If a sensor fault targets a leak node (or a leaked pipe) the overlap is recorded under `fault_summary.interactions` in the sidecar metadata. This is **informational only** and never gates the run.

| Config | What it shows |
|---|---|
| `configs/cumulative_leak_bias_net3.yaml` | Abrupt leak plus a bias sensor on Net3 |
| `configs/cumulative_leak_drift_hanoi.yaml` | Incipient leak plus a drifting sensor on Hanoi |
| `configs/cumulative_leak_dropout_jilin.yaml` | Two concurrent leaks plus a dropout sensor on Jilin |
| `configs/cumulative_leak_noise_fowm.yaml` | Random-location leak plus a noise sensor on FOWM |
| `configs/cumulative_leak_gain_net3.yaml` | Abrupt leak plus a gain sensor on Net3 |

### Leak-node cleanup (D28)

Injecting a leak splits a pipe and inserts a new junction, so a leak scenario's output schema gains a leak-node column and an extra pipe segment relative to a normal scenario. Setting `output.remove_leak_nodes: true` runs an optional post-processing step that reverts these artefacts: leak-node columns are dropped from the node tables, the upstream split-segment flow column is dropped and the downstream segment is renamed back to the original pipe name. The `leak_demand` diagnostic table is never touched. With cleanup enabled the output column schema matches the original network exactly. The default is `false` (explicit opt-in). See `configs/leak_abrupt_net3_clean.yaml` for a worked example and `docs/supervisor_leak_cleanup_reference.py` for the supervisor's reference logic.

### PDD normal scenarios

From Week 5 onwards the project default for normal scenarios is the pressure-dependent demand model. The `configs/normal_*_pdd.yaml` configs are the PDD counterparts of the DDA `normal_*` configs (kept for reference). Under PDD the simulator reduces delivered demand as pressure falls instead of forcing flow at negative pressure. Note that PDD does **not** remove the known Net3 node "10" negative-pressure dip: node "10" carries no consumer demand, so the dip is a pure elevation/topology artefact that is bit-identical under DDA and PDD.

---

## Running Tests

```bash
pytest
```

Currently 157 tests covering every module: config schema (with leak-spec and sensor-fault validators), network loading, demand strategies, simulation, leak injection (abrupt + linear + multi), sensor fault injection (bias / drift / stuck / dropout / noise / gain), leak-node cleanup post-processing, cumulative scenarios, PDD normal scenarios, labelling, validation severity (leak-aware mass balance, leak demand active, sensor fault mask consistency, sensor fault signal applied, gain detectability), output writers, and end-to-end runner determinism.

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

### Phase 1: Setup and exploration (Week 1) — complete

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

### Phase 3: pipe leak injection (Week 3) — complete

- Typed `LeakSpec` config model with mutual-exclusion (area xor diameter), seed-driven random pipe and split-fraction selection
- Abrupt + linear/step incipient profiles via WNTR controls
- Explicit PDD requirement enforced at config-load time
- `leak_demand` exposed as a fourth output table (parquet + csv)
- Leak-aware mass balance, leak pressure-drop check, leak labels in metadata
- 75 pytest tests, deterministic to the WNTR Newton noise floor
- Six new configs: normal + leak scenarios on Net3, Hanoi, Jilin and FOWM

### Phase 4 Week 4: sensor fault models — complete

- Typed `SensorFaultSpec` with discriminated `type` field (bias / drift / stuck / dropout / noise)
- Post-simulation injector preserving clean signals in `pressure_clean` / `flowrate_clean`
- Half-open `[start, end)` fault windows consistent with Phase 3 leaks
- Random vs explicit target selection mirroring Phase 3 leak placement
- Two structural validators that can fail (`sensor_fault_mask_consistent`, `sensor_fault_signal_applied`)
- Per-channel mask columns in the corrupted output tables (`{type}_mask_{target}`)
- Five Net3 sensor configs and `scripts/generate_week4_plots.py`

### Phase 4 Week 5: cumulative anomalies, gain fault, PDD normals, leak cleanup — complete

- Cumulative scenarios (leak + sensor faults) run end-to-end with a dedicated `validate_cumulative_scenario`
- Sixth sensor fault type `gain` with the D27 runtime detectability warning
- Optional leak-node cleanup post-processing (`postprocess.py`, D28) reverting split artefacts on output
- Four PDD normal configs (the new project default for normal scenarios)
- Five cumulative configs, `sensor_gain_net3` and `leak_abrupt_net3_clean`
- Cumulative interaction recording (sensor fault on a leak node) in the sidecar metadata
- 157 tests total, ruff clean, and `scripts/generate_week5_plots.py`

### Upcoming

- Phase 4 Week 6: batch generation, dataset polish, Phase 4 plots
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
