"""Generate the Phase 5 Week 8 scale-run scenario configs.

Emits several hundred scenario YAMLs across the four core networks
(Net3, Hanoi, FOWM, Jilin) covering normal, leak, sensor-fault and
cumulative scenarios. All sampling is driven by one master seed so the
whole set is reproducible: re-running this script reproduces byte-
identical configs. The sampling distributions are recorded in each
config's header comment and summarised below.

Per network (defaults), 130 scenarios:

- 25 normal: demand_multiplier jittered U(0.9, 1.1) so even
  default-demand networks vary; PDD.
- 50 leak: random pipe (drawn from the scenario seed), area log-uniform
  [1e-4, 1e-2] m^2, onset/duration sampled on the hour grid, 50/50
  abrupt vs incipient (linear) mix.
- 40 sensor fault: the six fault types cycled, random target and
  quantity, magnitudes sampled per quantity.
- 15 cumulative: one leak plus one sensor fault.

4 networks x 130 = 520 scenarios by default (inside the 400-600 target).

Configs land in ``configs/scale/`` (gitignored; only this generator is
committed). Output goes to ``outputs/scale``.

Usage::

    python scripts/generate_scale_configs.py
    python scripts/generate_scale_configs.py --master-seed 7 --leak 30
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import yaml

from wdn_pipeline.config import PipelineConfig

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "configs" / "scale"
DEFAULT_MASTER_SEED = 20260609
DEFAULT_OUTPUT_DIR = "outputs/scale"

HOUR = 3600
DURATION_SECONDS = 24 * HOUR

# Core networks: (inp_path written into the config, demand mode).
NETWORKS: list[tuple[str, str, str]] = [
    ("net3", "Net3", "default"),
    ("hanoi", "networks/Hanoi.inp", "fourier"),
    ("fowm", "networks/FOWM.inp", "fourier"),
    ("jilin", "networks/Jilin.inp", "default"),
]

SENSOR_TYPES = ["bias", "drift", "stuck", "dropout", "noise", "gain"]

# Generous validation tolerances: across hundreds of leak/sensor
# scenarios we do not want documented physical artefacts (Net3 node "10",
# leak-induced dips, sensor corruption pushing the reported signal out of
# the nominal band) to register as hard failures.
SCALE_VALIDATION = {
    "pressure_min_m": 0.0,
    "pressure_min_warning_tolerance_m": 15.0,
    "pressure_max_m": 300.0,
    "mass_balance_tol_m3s": 1.0e-3,
}


def _window(rng: np.random.Generator) -> tuple[int, int]:
    """Sample a half-open fault window on the hour grid within 24h."""

    start_h = int(rng.integers(1, 13))
    dur_h = int(rng.integers(4, 19))
    end_h = min(start_h + dur_h, 24)
    return start_h * HOUR, end_h * HOUR


def _magnitude(rng: np.random.Generator, quantity: str, kind: str) -> float:
    """Sample a fault magnitude appropriate to the measured quantity."""

    if quantity == "pressure":
        ranges = {"bias": (2.0, 10.0), "noise": (0.5, 3.0), "drift": (2.0, 10.0)}
    else:  # flowrate, m^3/s
        ranges = {"bias": (2e-3, 2e-2), "noise": (1e-3, 1e-2), "drift": (2e-3, 2e-2)}
    lo, hi = ranges[kind]
    return float(rng.uniform(lo, hi))


def _base_config(
    network_name: str,
    inp_path: str,
    mode: str,
    seed: int,
    label: str,
    scenario_type: str,
) -> dict:
    simulation = {
        "duration_seconds": DURATION_SECONDS,
        "hydraulic_timestep_seconds": HOUR,
        "report_timestep_seconds": HOUR,
        "demand_model": "PDD",
    }
    demand: dict = {"mode": mode}
    if mode == "fourier":
        simulation["pattern_timestep_seconds"] = HOUR
    return {
        "network": {"inp_path": inp_path, "name": network_name},
        "simulation": simulation,
        "seed": seed,
        "demand": demand,
        "scenario": {"type": scenario_type, "label": label},
        "validation": dict(SCALE_VALIDATION),
        "output": {
            "directory": DEFAULT_OUTPUT_DIR,
            "formats": ["parquet", "csv"],
            "write_metadata_sidecar": True,
        },
    }


def _leak_spec(rng: np.random.Generator) -> dict:
    area = 10.0 ** rng.uniform(math.log10(1e-4), math.log10(1e-2))
    start, end = _window(rng)
    abrupt = bool(rng.integers(0, 2))
    spec = {
        "area_m2": float(area),
        "start_time_seconds": start,
        "end_time_seconds": end,
        "profile": "abrupt" if abrupt else "linear",
    }
    if not abrupt:
        spec["profile_steps"] = 30
    return spec


def _sensor_spec(rng: np.random.Generator, ftype: str) -> dict:
    quantity = "pressure" if rng.integers(0, 2) else "flowrate"
    start, end = _window(rng)
    spec: dict = {
        "type": ftype,
        "quantity": quantity,
        "start_time_seconds": start,
        "end_time_seconds": end,
    }
    sign = 1.0 if rng.integers(0, 2) else -1.0
    if ftype == "bias":
        spec["bias_value"] = sign * _magnitude(rng, quantity, "bias")
    elif ftype == "drift":
        total = sign * _magnitude(rng, quantity, "drift")
        spec["slope_per_second"] = total / (end - start)
    elif ftype == "stuck":
        pass
    elif ftype == "dropout":
        iv_start = start + HOUR
        iv_end = min(iv_start + int(rng.integers(1, 4)) * HOUR, end)
        spec["intervals"] = [[iv_start, iv_end]]
    elif ftype == "noise":
        spec["sigma"] = _magnitude(rng, quantity, "noise")
    elif ftype == "gain":
        # Avoid 1.0; draw from below or above unity.
        if rng.integers(0, 2):
            spec["gain_factor"] = float(rng.uniform(0.7, 0.95))
        else:
            spec["gain_factor"] = float(rng.uniform(1.05, 1.3))
    return spec


def build_scenarios(
    out_dir: Path,
    master_seed: int,
    n_normal: int,
    n_leak: int,
    n_sensor: int,
    n_cumulative: int,
) -> dict[str, int]:
    """Generate and write every scale config. Returns counts by type."""

    rng = np.random.default_rng(master_seed)
    counts = {"normal": 0, "leak": 0, "sensor_fault": 0, "cumulative": 0}

    for net_name, inp_path, mode in NETWORKS:
        # Normal seed sweep with demand-multiplier jitter for variety.
        for i in range(n_normal):
            cfg = _base_config(
                net_name,
                inp_path,
                mode,
                seed=i,
                label=f"normal_{i}",
                scenario_type="normal",
            )
            cfg["simulation"]["demand_multiplier"] = float(rng.uniform(0.9, 1.1))
            _write(out_dir, net_name, f"normal_{i}", cfg, master_seed)
            counts["normal"] += 1

        for i in range(n_leak):
            seed = int(rng.integers(0, 2**31 - 1))
            cfg = _base_config(
                net_name,
                inp_path,
                mode,
                seed=seed,
                label=f"leak_{i}",
                scenario_type="leak",
            )
            cfg["faults"] = {"leaks": [_leak_spec(rng)]}
            _write(out_dir, net_name, f"leak_{i}", cfg, master_seed)
            counts["leak"] += 1

        for i in range(n_sensor):
            ftype = SENSOR_TYPES[i % len(SENSOR_TYPES)]
            seed = int(rng.integers(0, 2**31 - 1))
            cfg = _base_config(
                net_name,
                inp_path,
                mode,
                seed=seed,
                label=f"sensor_{ftype}_{i}",
                scenario_type="sensor_fault",
            )
            cfg["faults"] = {"sensor_faults": [_sensor_spec(rng, ftype)]}
            _write(out_dir, net_name, f"sensor_{ftype}_{i}", cfg, master_seed)
            counts["sensor_fault"] += 1

        for i in range(n_cumulative):
            ftype = SENSOR_TYPES[i % len(SENSOR_TYPES)]
            seed = int(rng.integers(0, 2**31 - 1))
            cfg = _base_config(
                net_name,
                inp_path,
                mode,
                seed=seed,
                label=f"cumulative_{i}",
                scenario_type="cumulative",
            )
            cfg["faults"] = {
                "leaks": [_leak_spec(rng)],
                "sensor_faults": [_sensor_spec(rng, ftype)],
            }
            _write(out_dir, net_name, f"cumulative_{i}", cfg, master_seed)
            counts["cumulative"] += 1

    return counts


def _write(out_dir: Path, net_name: str, label: str, cfg: dict, master_seed: int) -> None:
    # Validate before writing so every scale config is guaranteed loadable.
    PipelineConfig.model_validate(cfg)
    header = (
        f"# Scale-run config (Phase 5 Week 8) generated by "
        f"scripts/generate_scale_configs.py\n"
        f"# master_seed={master_seed} | sampling: leak area log-uniform "
        f"[1e-4, 1e-2] m^2, windows on the hour grid in [1h, 24h], "
        f"abrupt/incipient 50/50, sensor magnitudes per quantity.\n"
    )
    path = out_dir / f"{net_name}_{label}.yaml"
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--normal", type=int, default=25)
    parser.add_argument("--leak", type=int, default=50)
    parser.add_argument("--sensor", type=int, default=40)
    parser.add_argument("--cumulative", type=int, default=15)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale configs so re-runs with different counts stay consistent.
    for stale in args.out_dir.glob("*.yaml"):
        stale.unlink()

    counts = build_scenarios(
        args.out_dir,
        args.master_seed,
        args.normal,
        args.leak,
        args.sensor,
        args.cumulative,
    )
    total = sum(counts.values())
    print(f"Wrote {total} scale configs to {args.out_dir} (master_seed={args.master_seed})")
    for scenario_type, n in counts.items():
        print(f"  {scenario_type:14s} {n}")


if __name__ == "__main__":
    main()
