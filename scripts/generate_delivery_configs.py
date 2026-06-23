"""Generate the supervisor delivery-dataset scenario configs.

A focused, deterministic spread for the delivery dataset, modelled on
``scripts/generate_scale_configs.py`` but producing exactly one of each
fault type per network so the inventory is clean and documentable.

Networks: the four core networks (Net3, Hanoi, FOWM, Jilin) plus the
LeakG3PD sweep networks that converge under the delivery's 24 h PDD
config (Net1, PA1, modena, ky2). Two tiers:

- **Full spread** (leaks converge): normal PDD, the three leak profiles
  (abrupt / linear / step), all six sensor fault types and two cumulative
  scenarios. 12 scenarios each.
- **Normal + sensor only** (ky2): leaks under PDD do not converge on this
  large uncalibrated network, so it carries a normal scenario and one
  sensor fault. 2 each.

ky1, ky3 and ky5 are excluded: their full-day demand makes them
non-convergent under PDD (the convergence guard rejects them). The older
network sweep listed them as warnings only because it predated that guard
and accepted the non-converged partial result as data.

All timings are fixed (24 h / 1 h, fault window 06:00-18:00) and every
random field (leak pipe, sensor target) is drawn from the scenario seed,
so re-running this script reproduces byte-identical configs. Validation
tolerances follow the sweep convention (uncalibrated networks warn, never
fail, on pressure; structural and mass-balance checks still fail).

Configs land in ``configs/delivery/`` (gitignored; only this generator is
committed). Pipeline output goes to ``outputs/delivery/runs``.

Usage::

    python scripts/generate_delivery_configs.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import yaml

from wdn_pipeline.config import PipelineConfig

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "configs" / "delivery"
DEFAULT_OUTPUT_DIR = "outputs/delivery/runs"

HOUR = 3600
DURATION_SECONDS = 24 * HOUR
WINDOW = (6 * HOUR, 18 * HOUR)  # 06:00-18:00, half-open

LEAK_AREA_M2 = 1.0e-3
LEAK_PROFILES = ["abrupt", "linear", "step"]
SENSOR_TYPES = ["bias", "drift", "stuck", "dropout", "noise", "gain"]

# (name, inp_path, demand_mode, full_spread). Demand modes match the
# network sweep (the modes proven to run each network).
NETWORKS: list[tuple[str, str, str, bool]] = [
    ("net3", "Net3", "default", True),
    ("hanoi", "networks/Hanoi.inp", "fourier", True),
    ("fowm", "networks/FOWM.inp", "fourier", True),
    ("jilin", "networks/Jilin.inp", "default", True),
    ("net1", "networks/Net1.inp", "fourier", True),
    ("pa1", "networks/PA1.INP", "default", True),
    ("modena", "networks/modena.inp", "fourier", True),
    ("ky2", "networks/ky2.inp", "fourier", False),
]

# Uncalibrated networks warn (never fail) on pressure; structural and
# mass-balance checks still fail. Matches the sweep convention.
DELIVERY_VALIDATION = {
    "pressure_min_m": 0.0,
    "pressure_min_warning_tolerance_m": 1.0e6,
    "pressure_max_m": 1.0e6,
    "mass_balance_tol_m3s": 1.0e-3,
}


def _base_config(
    name: str, inp_path: str, mode: str, seed: int, label: str, scenario_type: str
) -> dict:
    simulation: dict = {
        "duration_seconds": DURATION_SECONDS,
        "hydraulic_timestep_seconds": HOUR,
        "report_timestep_seconds": HOUR,
        "demand_model": "PDD",
    }
    if mode == "fourier":
        simulation["pattern_timestep_seconds"] = HOUR
    return {
        "network": {"inp_path": inp_path, "name": name},
        "simulation": simulation,
        "seed": seed,
        "demand": {"mode": mode},
        "scenario": {"type": scenario_type, "label": label},
        "validation": dict(DELIVERY_VALIDATION),
        "output": {
            "directory": DEFAULT_OUTPUT_DIR,
            "formats": ["parquet"],
            "write_metadata_sidecar": True,
        },
    }


def _leak_spec(profile: str) -> dict:
    start, end = WINDOW
    spec: dict = {
        "area_m2": LEAK_AREA_M2,
        "start_time_seconds": start,
        "end_time_seconds": end,
        "profile": profile,
    }
    if profile != "abrupt":
        spec["profile_steps"] = 20
    return spec


def _sensor_spec(ftype: str) -> dict:
    start, end = WINDOW
    spec: dict = {"type": ftype, "start_time_seconds": start, "end_time_seconds": end}
    if ftype == "bias":
        spec.update(quantity="pressure", bias_value=5.0)
    elif ftype == "drift":
        spec.update(quantity="pressure", slope_per_second=5.0 / (end - start))
    elif ftype == "stuck":
        spec.update(quantity="pressure")
    elif ftype == "dropout":
        spec.update(quantity="flowrate", intervals=[[8 * HOUR, 14 * HOUR]])
    elif ftype == "noise":
        spec.update(quantity="flowrate", sigma=5.0e-3)
    elif ftype == "gain":
        spec.update(quantity="flowrate", gain_factor=1.2)
    return spec


def build_scenarios(out_dir: Path) -> dict[str, int]:
    """Generate and write every delivery config. Returns counts by type."""

    counts = {"normal": 0, "leak": 0, "sensor_fault": 0, "cumulative": 0}

    for net_idx, (name, inp_path, mode, full_spread) in enumerate(NETWORKS):
        k = 0

        def emit(label: str, scenario_type: str, faults: dict | None) -> None:
            nonlocal k
            seed = net_idx * 100 + k
            k += 1
            cfg = _base_config(name, inp_path, mode, seed, label, scenario_type)
            if faults:
                cfg["faults"] = faults
            _write(out_dir, name, label, cfg)
            counts[scenario_type] += 1

        # Normal PDD baseline (every network).
        emit("normal_pdd", "normal", None)

        if full_spread:
            for profile in LEAK_PROFILES:
                emit(f"leak_{profile}", "leak", {"leaks": [_leak_spec(profile)]})
            for ftype in SENSOR_TYPES:
                emit(f"sensor_{ftype}", "sensor_fault", {"sensor_faults": [_sensor_spec(ftype)]})
            emit(
                "cumulative_leak_bias",
                "cumulative",
                {"leaks": [_leak_spec("abrupt")], "sensor_faults": [_sensor_spec("bias")]},
            )
            emit(
                "cumulative_leak_noise",
                "cumulative",
                {"leaks": [_leak_spec("linear")], "sensor_faults": [_sensor_spec("noise")]},
            )
        else:
            # Leaks under PDD do not converge on these uncalibrated networks;
            # include a normal baseline and one sensor fault (post-simulation,
            # so the hydraulic solve is the same as the normal scenario).
            emit("sensor_bias", "sensor_fault", {"sensor_faults": [_sensor_spec("bias")]})

    return counts


def _write(out_dir: Path, name: str, label: str, cfg: dict) -> None:
    # Validate before writing so every delivery config is guaranteed loadable.
    PipelineConfig.model_validate(cfg)
    header = (
        "# Delivery-dataset config generated by "
        "scripts/generate_delivery_configs.py\n"
        "# 24h / 1h PDD; fault window 06:00-18:00; leak area 1e-3 m^2 on a "
        "seed-drawn pipe; sweep validation tolerances.\n"
    )
    path = out_dir / f"{name}_{label}.yaml"
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.yaml"):
        stale.unlink()

    counts = build_scenarios(args.out_dir)
    total = sum(counts.values())
    print(f"Wrote {total} delivery configs to {args.out_dir}")
    for scenario_type, n in counts.items():
        print(f"  {scenario_type:14s} {n}")


if __name__ == "__main__":
    main()
