# WDN Anomaly Pipeline

A modular, configuration-driven Python pipeline that wraps [EPANET/WNTR](https://usepa.github.io/WNTR/) to systematically generate large-scale, labelled time-series datasets for water distribution networks (WDNs). The pipeline covers normal baseline scenarios, pipe leaks (abrupt and incipient), sensor faults (stuck, drift, dropout, bias) and cumulative anomalies — producing reproducible, ML-ready datasets in Parquet and CSV formats.

**MSc internship project** — Teodor Cioata, MSc Computing Science, University of Groningen, 2026.  
Supervisors: Dilek Düştegör, Alexander Lazovik, Samer Ahmed.

---

## Project Structure

```
wdn-anomaly-pipeline/
├── configs/          # YAML scenario configuration files
├── networks/         # EPANET .inp topology files
├── src/
│   └── wdn_pipeline/ # Main Python package
│       └── faults/   # Leak and sensor fault injectors
├── notebooks/        # Jupyter exploration notebooks
├── tests/            # pytest unit and integration tests
├── outputs/          # Generated datasets (gitignored)
└── docs/             # Additional documentation
```

---

## Setup

**Requirements:** Python 3.10+ (Python 3.12 recommended), Git.

```bash
# 1. Clone the repo
git clone <repo-url>
cd wdn-anomaly-pipeline

# 2. Create and activate a virtual environment
python3 -m virtualenv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package and all dependencies
pip install -e ".[dev]"

# 4. Verify WNTR is working
python -c "import wntr; print('wntr', wntr.__version__)"
```

## Running the Notebooks

```bash
source .venv/bin/activate
jupyter lab
```

Navigate to `notebooks/01_wntr_basics.ipynb` for the initial WNTR exploration.

---

## Running Tests

```bash
source .venv/bin/activate
pytest
```

---

## Status

**Phase 1: Environment setup and exploration** (Week 1–2, April 2026)

- [x] Literature review and context gathering
- [x] Environment setup (dependencies, repo structure, venv)
- [x] Reference repos cloned (DiTEC-WDN, LeakG3PD)
- [ ] WNTR basic experiments (Net3 simulation, leak injection)
- [ ] Findings documented in `notebooks/01_wntr_basics.ipynb`

Upcoming: Phase 2 — config schema design and normal scenario pipeline.

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `wntr` | EPANET/WNTR hydraulic simulation |
| `pandas` + `pyarrow` | Tabular data and Parquet serialisation |
| `pydantic` | Config schema validation |
| `pyyaml` | YAML config loading |
| `matplotlib` | Visualisation |
| `duckdb` | Queryable dataset exports |
| `pytest` + `ruff` | Testing and linting |
