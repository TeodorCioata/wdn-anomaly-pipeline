"""Run every YAML config and organise CSV outputs into a Drive-friendly tree.

This script is the Phase 5 Week 7 deliverable for the supervisor's
Google Drive review. It runs every config under ``configs/`` and writes
each scenario's CSV outputs into a categorised directory tree under
``outputs/organised/``::

    outputs/organised/
      normal/<network>/{dda,pdd}/<csvs>
      leak/<network>/<variant>/<csvs>
      sensor_fault/<network>/<variant>/<csvs>
      cumulative/<network>/<variant>/<csvs>
      README.md

The script is idempotent: re-running it overwrites the per-scenario
folders. A top-level ``README.md`` is generated documenting the
directory layout, column conventions and the source config for each
folder.

CSV-only by design: the supervisor inspects CSVs directly. Parquet
outputs continue to be produced by the regular pipeline runs.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wdn_pipeline.config import load_config  # noqa: E402
from wdn_pipeline.runner import run  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("organise_outputs")


# Per-config mapping from filename stem to (category, network, variant).
# Configs that do not appear in this map fall through to a heuristic
# derived from the loaded ScenarioConfig.
EXPLICIT_TREE_MAP: dict[str, tuple[str, str, str]] = {
    # Normal scenarios (one DDA + one PDD per network from Week 5 onwards).
    "normal_net3":        ("normal", "net3",  "dda"),
    "normal_net3_pdd":    ("normal", "net3",  "pdd"),
    "normal_hanoi":       ("normal", "hanoi", "dda"),
    "normal_hanoi_pdd":   ("normal", "hanoi", "pdd"),
    "normal_fowm":        ("normal", "fowm",  "dda"),
    "normal_fowm_pdd":    ("normal", "fowm",  "pdd"),
    "normal_jilin":       ("normal", "jilin", "dda"),
    "normal_jilin_pdd":   ("normal", "jilin", "pdd"),
    # Leak scenarios (one per profile / location strategy).
    "leak_abrupt_net3":         ("leak", "net3",  "abrupt"),
    "leak_abrupt_net3_clean":   ("leak", "net3",  "abrupt_clean"),
    "leak_incipient_hanoi":     ("leak", "hanoi", "incipient"),
    "leak_multi_jilin":         ("leak", "jilin", "multi"),
    "leak_random_fowm":         ("leak", "fowm",  "random"),
    # Sensor fault scenarios (one per fault type).
    "sensor_bias_net3":     ("sensor_fault", "net3", "bias"),
    "sensor_drift_net3":    ("sensor_fault", "net3", "drift"),
    "sensor_stuck_net3":    ("sensor_fault", "net3", "stuck"),
    "sensor_dropout_net3":  ("sensor_fault", "net3", "dropout"),
    "sensor_noise_net3":    ("sensor_fault", "net3", "noise"),
    "sensor_gain_net3":     ("sensor_fault", "net3", "gain"),
    # Cumulative scenarios (one per network).
    "cumulative_leak_bias_net3":    ("cumulative", "net3",  "leak_bias"),
    "cumulative_leak_drift_hanoi":  ("cumulative", "hanoi", "leak_drift"),
    "cumulative_leak_dropout_jilin":("cumulative", "jilin", "leak_dropout"),
    "cumulative_leak_gain_net3":    ("cumulative", "net3",  "leak_gain"),
    "cumulative_leak_noise_fowm":   ("cumulative", "fowm",  "leak_noise"),
}


def categorise_config(config_path: Path) -> tuple[str, str, str]:
    """Resolve a config to ``(category, network, variant)`` tree coordinates.

    Falls back to the loaded config's scenario type and network name
    when the filename is not in :data:`EXPLICIT_TREE_MAP`.
    """

    stem = config_path.stem
    if stem in EXPLICIT_TREE_MAP:
        return EXPLICIT_TREE_MAP[stem]
    cfg = load_config(config_path)
    category = cfg.scenario.type
    network = (cfg.network.name or Path(cfg.network.inp_path).stem).lower()
    variant = cfg.scenario.label
    return category, network, variant


CATEGORY_TABLES: dict[str, tuple[str, ...]] = {
    "normal": ("pressure", "flowrate", "demand"),
    "leak": (
        "pressure",
        "pressure_clean",
        "flowrate",
        "flowrate_clean",
        "demand",
        "leak_demand",
    ),
    "sensor_fault": (
        "pressure",
        "pressure_clean",
        "flowrate",
        "flowrate_clean",
        "demand",
    ),
    "cumulative": (
        "pressure",
        "pressure_clean",
        "flowrate",
        "flowrate_clean",
        "demand",
        "leak_demand",
    ),
}


def organise_one(
    config_path: Path,
    organised_root: Path,
    scratch_root: Path,
) -> dict[str, object]:
    """Run one config and copy the relevant CSV outputs into the tree.

    The pipeline always writes the full table set; this function selects
    the subset relevant to the scenario category (e.g. normal scenarios
    do not need the ``*_clean`` siblings because no faults are present).
    """

    category, network, variant = categorise_config(config_path)
    scenario_dir = organised_root / category / network / variant
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Build a fresh PipelineConfig with the output redirected to a
    # scratch directory and CSV-only formats. Parquet output stays out
    # of the organised tree to keep the Drive upload focused.
    cfg = load_config(config_path)
    cfg_scratch_dir = scratch_root / config_path.stem
    cfg_scratch_dir.mkdir(parents=True, exist_ok=True)
    updated_output = cfg.output.model_copy(
        update={
            "directory": cfg_scratch_dir,
            "formats": ["csv"],
            "write_metadata_sidecar": True,
            "duckdb": False,
            "duckdb_path": None,
        }
    )
    cfg = cfg.model_copy(update={"output": updated_output})

    summary = run(cfg, config_path=str(config_path))
    severity = summary.validation.severity

    wanted_tables = CATEGORY_TABLES.get(category, CATEGORY_TABLES["normal"])
    copied: list[str] = []
    for src in cfg_scratch_dir.glob("*.csv"):
        for table_name in sorted(wanted_tables, key=len, reverse=True):
            suffix = f"_{table_name}.csv"
            if src.name.endswith(suffix):
                dest = scenario_dir / f"{table_name}.csv"
                shutil.copyfile(src, dest)
                copied.append(table_name)
                break

    if summary.metadata_path and summary.metadata_path.is_file():
        shutil.copyfile(summary.metadata_path, scenario_dir / "metadata.yaml")

    return {
        "config_path": str(config_path),
        "category": category,
        "network": network,
        "variant": variant,
        "tables": sorted(copied),
        "scenario_dir": str(scenario_dir.relative_to(organised_root.parent)),
        "severity": severity,
    }


README_TEMPLATE = """# Organised Pipeline Outputs

This directory holds CSV outputs from the WDN anomaly simulation
pipeline, organised by scenario type, network and variant so they can
be browsed in Google Drive without unpacking. The folder layout is::

    outputs/organised/
      normal/<network>/{{dda,pdd}}/
      leak/<network>/<variant>/
      sensor_fault/<network>/<variant>/
      cumulative/<network>/<variant>/

Each leaf folder is generated from one YAML config under
``configs/``. The mapping is documented at the bottom of this file.

## CSV file inventory

| File | Description | Columns |
|---|---|---|
| ``pressure.csv`` | Reported pressure at every junction (metres) | ``time_seconds``, one column per junction, ``label``, optional ``{{type}}_mask_{{target}}`` columns when sensor faults are present |
| ``pressure_clean.csv`` | Uncorrupted pressure (clean ground truth). Equal to ``pressure.csv`` when no pressure sensor faults are present. | Same columns as ``pressure.csv`` minus mask columns |
| ``flowrate.csv`` | Reported flow rate on every pipe (m^3/s) | ``time_seconds``, one column per pipe, ``label``, optional mask columns for flowrate faults |
| ``flowrate_clean.csv`` | Uncorrupted flow rate. Equal to ``flowrate.csv`` when no flow sensor faults are present. | Same columns as ``flowrate.csv`` minus mask columns |
| ``demand.csv`` | Realised consumer demand at every junction (m^3/s) | ``time_seconds``, one column per junction, ``label`` |
| ``leak_demand.csv`` | Leak outflow at every inserted leak node (m^3/s). Zero for normal scenarios. | ``time_seconds``, one column per leak node, ``label`` |
| ``metadata.yaml`` | Scenario metadata sidecar (network, seed, faults, validation severity) | YAML |

## Conventions

- **Time resolution**: 1 hour (3600 s) over a 24 hour simulation by default.
  Each row is one report timestep.
- **Time column**: ``time_seconds`` (seconds from simulation start, 0
  inclusive, ``duration_seconds`` exclusive).
- **Label column**: ``label`` is ``0`` for normal samples and ``1`` for
  any timestep covered by at least one fault window (union of all leak
  and sensor fault windows).
- **Half-open windows**: every fault window uses ``[start, end)`` so the
  exact ``end_time_seconds`` sample is already a normal frame.
- **Sensor fault mask columns**: ``{{fault_type}}_mask_{{target}}``
  (e.g. ``bias_mask_15``) is ``True`` exactly where that fault is active.
  Mask columns are appended to the corresponding corrupted table; the
  ``*_clean`` siblings stay free of mask columns to keep them suitable
  as ground truth.
- **Units**: pressure in metres, flow rate in m^3/s, demand in m^3/s,
  time in seconds.
- **Networks**: ``net3`` (Net3, 92 junctions, 117 pipes), ``hanoi``
  (Hanoi, 31 junctions, 34 pipes), ``fowm`` (FOWM, 36 junctions, 39
  pipes), ``jilin`` (Jilin, 27 junctions, 34 pipes).

## Reading the metadata YAML

Each ``metadata.yaml`` carries at least::

    network_name:       string
    network_inp:        path or bundled-network name
    scenario_type:      "normal" | "leak" | "sensor_fault" | "cumulative"
    scenario_label:     filename-safe label
    seed:               integer
    demand_model:       "DDA" | "PDD"
    duration_seconds:   integer
    fault_summary:
      leaks:            list of resolved leaks (pipe, area_m2, window, profile, ...)
      sensor_faults:    list of resolved sensor faults (type, target, window, ...)
      interactions:     informational list of sensor-fault-on-leak overlaps

## Source configs

Each leaf folder maps back to one YAML config under ``configs/``. The
table below lists every scenario produced by ``scripts/organise_outputs.py``:

{config_table}

---

Generated by ``scripts/organise_outputs.py``. Re-run the script after
adding or modifying configs.
"""


def render_readme(records: list[dict[str, object]]) -> str:
    """Render the organised-outputs README from the per-scenario records."""

    rows: list[str] = []
    rows.append("| Folder | Config | Severity |")
    rows.append("|---|---|---|")
    for r in sorted(records, key=lambda x: str(x["scenario_dir"])):
        rows.append(
            f"| `{r['scenario_dir']}` | `{r['config_path']}` | {r['severity']} |"
        )
    return README_TEMPLATE.format(config_table="\n".join(rows))


def iter_configs(config_dir: Path, only: Iterable[str] | None = None) -> list[Path]:
    """List configs to process, optionally filtered to a subset by stem."""

    paths = sorted(config_dir.glob("*.yaml"))
    if only:
        wanted = set(only)
        paths = [p for p in paths if p.stem in wanted]
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "configs",
        help="Directory of YAML configs.",
    )
    parser.add_argument(
        "--organised-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "organised",
        help="Destination tree.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "_organise_scratch",
        help="Scratch directory for per-config runs (overwritten).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of config stems to process (debug helper).",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep the per-config scratch directory after copying CSVs.",
    )
    args = parser.parse_args()

    organised_root: Path = args.organised_root
    scratch_root: Path = args.scratch_root
    if organised_root.exists():
        shutil.rmtree(organised_root)
    organised_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)

    configs = iter_configs(args.config_dir, args.only)
    if not configs:
        logger.error("No configs found in %s", args.config_dir)
        return 1

    records: list[dict[str, object]] = []
    failures: list[tuple[Path, str]] = []
    for i, cfg_path in enumerate(configs, start=1):
        logger.info("[%d/%d] %s", i, len(configs), cfg_path.name)
        try:
            record = organise_one(cfg_path, organised_root, scratch_root)
            records.append(record)
        except Exception as exc:  # noqa: BLE001 - keep going across configs
            logger.exception("Failed on %s", cfg_path)
            failures.append((cfg_path, str(exc)))

    readme_path = organised_root / "README.md"
    readme_path.write_text(render_readme(records), encoding="utf-8")
    logger.info("Wrote %s (%d scenarios)", readme_path, len(records))

    manifest_path = organised_root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(records, sort_keys=False), encoding="utf-8")

    if not args.keep_scratch and scratch_root.exists():
        shutil.rmtree(scratch_root)

    if failures:
        logger.error("%d config(s) failed:", len(failures))
        for p, err in failures:
            logger.error("  %s: %s", p, err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
