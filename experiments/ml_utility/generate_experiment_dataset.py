"""Build the dedicated Study 4 experiment dataset.

The pipeline's existing scale run is hourly (25 steps per scenario), which gives
only one window per scenario and no detection-delay signal. This script builds a
finer-resolution, difficulty-swept dataset purpose-built for the downstream
anomaly-detection study:

- **15-minute timestep, 24 h horizon** -> 97 reported timesteps per scenario.
- **Net3 and Hanoi** (two networks, so the study can speak to cross-network
  behaviour rather than a single topology).
- **Normal** scenarios for the (semi-supervised) training set, plus
  **difficulty-swept** leak and sensor-fault scenarios for evaluation:
  leak hole area on a log-spaced grid (abrupt and incipient), and sensor gain /
  bias / drift over magnitude grids. All sensor faults target *pressure* because
  the detectors read pressure (a flowrate fault would be invisible to a
  pressure detector by construction).

Everything is driven by one master seed, so re-running reproduces byte-identical
configs and (up to the simulator's documented ~1e-12 m numerical noise) an
identical dataset. The script writes the configs, runs them through the batch
driver into a consolidated long-only DuckDB file, and emits a ``manifest.json``
recording every scenario's difficulty parameters and the full sampling design.

Usage::

    python experiments/ml_utility/generate_experiment_dataset.py
    python experiments/ml_utility/generate_experiment_dataset.py --workers 4
    python experiments/ml_utility/generate_experiment_dataset.py --quick   # tiny smoke set
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from wdn_pipeline.batch import run_batch
from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.output import build_basename

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
DATA_DIR = HERE / "data"
DUCKDB_PATH = DATA_DIR / "experiment.duckdb"
MANIFEST_PATH = DATA_DIR / "manifest.json"
REPORT_DIR = DATA_DIR / "batch_report"

MASTER_SEED = 20260623
HOUR = 3600
TIMESTEP = 900  # 15 minutes
DURATION = 24 * HOUR  # 24 h -> 97 reported steps (incl. t=0)

# (config inp_path, network.name, demand mode).
NETWORKS: list[tuple[str, str, str]] = [
    ("Net3", "net3", "default"),
    ("networks/Hanoi.inp", "hanoi", "fourier"),
]

# Difficulty grids (the heart of the "easy-to-hard" demonstration).
LEAK_AREAS_M2 = [float(a) for a in np.geomspace(1e-4, 1e-2, 6)]
GAIN_FACTORS = [1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0]
BIAS_MAGNITUDES_M = [0.5, 1.0, 2.0, 5.0, 10.0]
DRIFT_TOTALS_M = [0.5, 1.0, 2.0, 5.0, 10.0]

# Conservative validation tolerances: documented physical artefacts (Net3 node
# "10", leak-induced dips, uncalibrated sub-zero pressure) should warn, never
# fail, across a few hundred scenarios.
VALIDATION = {
    "pressure_min_m": 0.0,
    "pressure_min_warning_tolerance_m": 20.0,
    "pressure_max_m": 400.0,
    "mass_balance_tol_m3s": 1.0e-3,
}


@dataclass
class Counts:
    """Per-type scenario counts, configurable for the smoke run."""

    normal: int = 150
    leak_abrupt_reps: int = 10
    leak_incipient_reps: int = 5
    gain_reps: int = 8
    bias_reps: int = 6
    drift_reps: int = 6


@dataclass
class ManifestEntry:
    """One scenario's record in the manifest (difficulty + provenance)."""

    basename: str
    network: str
    scenario_type: str
    seed: int
    fault_subtype: str = "none"
    quantity: str = "none"
    # Difficulty parameters (only the relevant ones are set per type).
    leak_area_m2: float | None = None
    leak_profile: str | None = None
    gain_factor: float | None = None
    bias_value_m: float | None = None
    drift_total_m: float | None = None
    onset_seconds: int | None = None
    end_seconds: int | None = None
    extra: dict = field(default_factory=dict)


def _window(rng: np.random.Generator) -> tuple[int, int]:
    """Sample a fault window with onset mid-morning and ample post-onset room.

    Onset in [6 h, 10 h], duration 10-12 h, clipped to the 24 h horizon. This
    leaves several hours of normal data before the onset (negative windows) and
    after detection (delay measurement).
    """

    onset_h = int(rng.integers(6, 11))
    dur_h = int(rng.integers(10, 13))
    end_h = min(onset_h + dur_h, 24)
    return onset_h * HOUR, end_h * HOUR


def _base_config(
    inp_path: str, net_name: str, mode: str, seed: int, label: str, scenario_type: str
) -> dict:
    simulation: dict = {
        "duration_seconds": DURATION,
        "hydraulic_timestep_seconds": TIMESTEP,
        "report_timestep_seconds": TIMESTEP,
        "demand_model": "PDD",
    }
    if mode == "fourier":
        simulation["pattern_timestep_seconds"] = TIMESTEP
    return {
        "network": {"inp_path": inp_path, "name": net_name},
        "simulation": simulation,
        "seed": seed,
        "demand": {"mode": mode},
        "scenario": {"type": scenario_type, "label": label},
        "validation": dict(VALIDATION),
        "output": {
            "directory": "experiments/ml_utility/data/scenarios",
            "formats": ["parquet"],
            "write_metadata_sidecar": True,
        },
    }


def _write(cfg: dict, net_name: str, label: str) -> str:
    """Validate and write one config; return its scenario basename."""

    PipelineConfig.model_validate(cfg)
    path = CONFIG_DIR / f"{net_name}_{label}.yaml"
    header = (
        "# Study 4 experiment config generated by "
        "experiments/ml_utility/generate_experiment_dataset.py\n"
        f"# master_seed={MASTER_SEED} | 15-min/24h PDD | difficulty-swept\n"
    )
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return build_basename(net_name, label, cfg["seed"])


def build_configs(counts: Counts) -> list[ManifestEntry]:
    """Generate every experiment config and return the manifest entries."""

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CONFIG_DIR.glob("*.yaml"):
        stale.unlink()

    rng = np.random.default_rng(MASTER_SEED)
    entries: list[ManifestEntry] = []

    for inp_path, net_name, mode in NETWORKS:
        # -- normal (training pool + held-out eval negatives) --------------
        for i in range(counts.normal):
            label = f"normal_{i}"
            cfg = _base_config(
                inp_path, net_name, mode, seed=i, label=label, scenario_type="normal"
            )
            cfg["simulation"]["demand_multiplier"] = float(rng.uniform(0.9, 1.1))
            basename = _write(cfg, net_name, label)
            entries.append(ManifestEntry(basename, net_name, "normal", i, fault_subtype="none"))

        # -- leak difficulty sweep (abrupt + incipient) --------------------
        for area in LEAK_AREAS_M2:
            for profile, reps in (
                ("abrupt", counts.leak_abrupt_reps),
                ("linear", counts.leak_incipient_reps),
            ):
                for _ in range(reps):
                    seed = int(rng.integers(0, 2**31 - 1))
                    onset, end = _window(rng)
                    tag = f"a{area:.0e}".replace("-", "m").replace("+", "")
                    label = f"leak_{profile}_{tag}_{seed}"
                    cfg = _base_config(
                        inp_path, net_name, mode, seed=seed, label=label, scenario_type="leak"
                    )
                    leak: dict = {
                        "area_m2": area,
                        "discharge_coeff": 0.75,
                        "start_time_seconds": onset,
                        "end_time_seconds": end,
                        "profile": profile,
                    }
                    if profile == "linear":
                        leak["profile_steps"] = 30
                    cfg["faults"] = {"leaks": [leak]}
                    basename = _write(cfg, net_name, label)
                    entries.append(
                        ManifestEntry(
                            basename,
                            net_name,
                            "leak",
                            seed,
                            fault_subtype=profile,
                            quantity="pressure",
                            leak_area_m2=area,
                            leak_profile=profile,
                            onset_seconds=onset,
                            end_seconds=end,
                        )
                    )

        # -- sensor gain sweep (pressure) ----------------------------------
        for gain in GAIN_FACTORS:
            for _ in range(counts.gain_reps):
                seed = int(rng.integers(0, 2**31 - 1))
                onset, end = _window(rng)
                label = f"sensor_gain_{int(round(gain * 100))}_{seed}"
                cfg = _base_config(
                    inp_path, net_name, mode, seed=seed, label=label, scenario_type="sensor_fault"
                )
                cfg["faults"] = {
                    "sensor_faults": [
                        {
                            "type": "gain",
                            "quantity": "pressure",
                            "gain_factor": gain,
                            "start_time_seconds": onset,
                            "end_time_seconds": end,
                        }
                    ]
                }
                basename = _write(cfg, net_name, label)
                entries.append(
                    ManifestEntry(
                        basename,
                        net_name,
                        "sensor_fault",
                        seed,
                        fault_subtype="gain",
                        quantity="pressure",
                        gain_factor=gain,
                        onset_seconds=onset,
                        end_seconds=end,
                    )
                )

        # -- sensor bias sweep (pressure) ----------------------------------
        for mag in BIAS_MAGNITUDES_M:
            for _ in range(counts.bias_reps):
                seed = int(rng.integers(0, 2**31 - 1))
                onset, end = _window(rng)
                sign = 1.0 if rng.integers(0, 2) else -1.0
                value = sign * mag
                label = f"sensor_bias_{mag:g}_{seed}"
                cfg = _base_config(
                    inp_path, net_name, mode, seed=seed, label=label, scenario_type="sensor_fault"
                )
                cfg["faults"] = {
                    "sensor_faults": [
                        {
                            "type": "bias",
                            "quantity": "pressure",
                            "bias_value": value,
                            "start_time_seconds": onset,
                            "end_time_seconds": end,
                        }
                    ]
                }
                basename = _write(cfg, net_name, label)
                entries.append(
                    ManifestEntry(
                        basename,
                        net_name,
                        "sensor_fault",
                        seed,
                        fault_subtype="bias",
                        quantity="pressure",
                        bias_value_m=value,
                        onset_seconds=onset,
                        end_seconds=end,
                    )
                )

        # -- sensor drift sweep (pressure) ---------------------------------
        for total in DRIFT_TOTALS_M:
            for _ in range(counts.drift_reps):
                seed = int(rng.integers(0, 2**31 - 1))
                onset, end = _window(rng)
                sign = 1.0 if rng.integers(0, 2) else -1.0
                slope = sign * total / (end - onset)
                label = f"sensor_drift_{total:g}_{seed}"
                cfg = _base_config(
                    inp_path, net_name, mode, seed=seed, label=label, scenario_type="sensor_fault"
                )
                cfg["faults"] = {
                    "sensor_faults": [
                        {
                            "type": "drift",
                            "quantity": "pressure",
                            "slope_per_second": slope,
                            "start_time_seconds": onset,
                            "end_time_seconds": end,
                        }
                    ]
                }
                basename = _write(cfg, net_name, label)
                entries.append(
                    ManifestEntry(
                        basename,
                        net_name,
                        "sensor_fault",
                        seed,
                        fault_subtype="drift",
                        quantity="pressure",
                        drift_total_m=sign * total,
                        onset_seconds=onset,
                        end_seconds=end,
                    )
                )

    return entries


def write_manifest(entries: list[ManifestEntry], counts: Counts) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "master_seed": MASTER_SEED,
        "timestep_seconds": TIMESTEP,
        "duration_seconds": DURATION,
        "networks": [n[1] for n in NETWORKS],
        "sampling": {
            "leak_areas_m2": LEAK_AREAS_M2,
            "gain_factors": GAIN_FACTORS,
            "bias_magnitudes_m": BIAS_MAGNITUDES_M,
            "drift_totals_m": DRIFT_TOTALS_M,
            "counts": counts.__dict__,
            "window": "onset U[6h,10h], duration U[10h,12h], clipped to 24h",
            "leak_pipe": "random (drawn from scenario seed)",
            "sensor_target": "random pressure node (drawn from scenario seed)",
        },
        "n_scenarios": len(entries),
        "scenarios": [e.__dict__ for e in entries],
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke set (few scenarios per type) for a fast end-to-end check",
    )
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="write configs and manifest but do not run the batch",
    )
    args = parser.parse_args()

    counts = (
        Counts(
            normal=6,
            leak_abrupt_reps=1,
            leak_incipient_reps=1,
            gain_reps=1,
            bias_reps=1,
            drift_reps=1,
        )
        if args.quick
        else Counts()
    )

    entries = build_configs(counts)
    write_manifest(entries, counts)
    print(f"Wrote {len(entries)} configs to {CONFIG_DIR} and manifest to {MANIFEST_PATH}")
    by_type: dict[str, int] = {}
    for e in entries:
        by_type[e.scenario_type] = by_type.get(e.scenario_type, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:14s} {n}")

    if args.configs_only:
        return

    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()
    print(f"\nRunning {len(config_paths)} scenarios with {args.workers} worker(s) -> {DUCKDB_PATH}")
    summary = run_batch(
        config_paths=config_paths,
        report_dir=REPORT_DIR,
        batch_id="ml_experiment",
        workers=args.workers,
        duckdb_path=DUCKDB_PATH,
        duckdb_wide_tables=False,  # long-only: the query layer / loader use the long tables
    )
    statuses: dict[str, int] = {}
    for r in summary.results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    print("Batch finished:", dict(sorted(statuses.items())))
    if summary.duckdb_import_errors:
        print(f"  WARNING: {len(summary.duckdb_import_errors)} DuckDB import errors")


if __name__ == "__main__":
    main()
