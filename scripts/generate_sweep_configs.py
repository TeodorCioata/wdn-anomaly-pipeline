"""Generate one normal-scenario sweep config per LeakG3PD network.

Phase 5 Week 8 (WP3). Emits a 24h / 1h-timestep PDD normal-scenario YAML
for every ``.inp`` under ``networks/`` into ``configs/sweep/``. The demand
mode is detected programmatically (the Hanoi/FOWM lesson from Phase 3):
networks that ship demand patterns and a nonzero .inp duration use the
``default`` demand strategy; networks without are driven by the synthetic
``fourier`` diurnal pattern so the simulation has dynamics.

Validation tolerances are deliberately conservative: these networks are
explicitly not all calibrated, so sub-zero pressures are expected and
should register as warnings, not failures. Only genuine breakage (NaN /
inf or a broken mass balance) is allowed to ``fail``.

The pipeline always uses the pure-Python WNTRSimulator (decision D3),
whose per-timestep cost grows with network size. Networks above
``--max-junctions`` are skipped to keep the sweep tractable; the network
sweep report documents them as excluded with the reason. Lower / raise
the cap on the command line as the machine allows.

Usage::

    python scripts/generate_sweep_configs.py
    python scripts/generate_sweep_configs.py --max-junctions 3000
"""

from __future__ import annotations

import argparse
import glob
import warnings
from pathlib import Path

import wntr
import yaml

from wdn_pipeline.config import PipelineConfig

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETWORKS_DIR = REPO_ROOT / "networks"
DEFAULT_OUT_DIR = REPO_ROOT / "configs" / "sweep"
DEFAULT_SEED = 42
DEFAULT_DURATION_SECONDS = 24 * 3600
DEFAULT_TIMESTEP_SECONDS = 3600
# WNTRSimulator (pure Python) becomes impractically slow past a few
# thousand junctions for a 24h run. Networks larger than this are skipped
# and documented as excluded in the sweep report.
DEFAULT_MAX_JUNCTIONS = 2000

# Conservative validation tolerances for uncalibrated networks: pressure
# issues warn (never fail), so only NaN/inf or a broken mass balance can
# fail a sweep run.
SWEEP_VALIDATION = {
    "pressure_min_m": 0.0,
    "pressure_min_warning_tolerance_m": 1.0e6,
    "pressure_max_m": 1.0e6,
    "mass_balance_tol_m3s": 1.0e-3,
}


def discover_networks(networks_dir: Path) -> list[Path]:
    """Return every ``.inp`` / ``.INP`` file under ``networks_dir``, sorted."""

    patterns = ["*.inp", "*.INP"]
    found: set[Path] = set()
    for pat in patterns:
        found.update(Path(p) for p in glob.glob(str(networks_dir / pat)))
    return sorted(found, key=lambda p: p.name.lower())


def network_stats(path: Path) -> dict:
    """Load a network and return the stats needed to build a sweep config."""

    wn = wntr.network.WaterNetworkModel(str(path))
    return {
        "junctions": wn.num_junctions,
        "pipes": wn.num_pipes,
        "tanks": wn.num_tanks,
        "reservoirs": wn.num_reservoirs,
        "has_patterns": len(wn.pattern_name_list) > 0,
        "inp_duration_seconds": int(wn.options.time.duration),
        "headloss": wn.options.hydraulic.headloss,
    }


def choose_demand_mode(stats: dict) -> str:
    """``default`` when the .inp ships usable patterns, else ``fourier``.

    The .inp must carry at least one demand pattern *and* a nonzero
    native duration for ``default`` to produce dynamics; otherwise the
    synthetic Fourier diurnal pattern is used (the Hanoi/FOWM lesson).
    """

    if stats["has_patterns"] and stats["inp_duration_seconds"] > 0:
        return "default"
    return "fourier"


def safe_name(path: Path) -> str:
    """Filename-safe network identifier from an .inp path."""

    return path.stem.replace(" ", "_").replace("/", "-")


def skip_reason(stats: dict, max_junctions: int) -> str | None:
    """Why this network is excluded from the sweep, or ``None`` to include.

    Two pre-known WNTRSimulator (D3) incompatibilities are caught here:

    - Darcy-Weisbach / Chezy-Manning headloss in the .inp (WNTRSimulator
      is Hazen-Williams only; forcing H-W would corrupt the physics).
    - More junctions than the cap (the pure-Python solver is too slow).

    Networks with non-unit pump speeds also fail under WNTRSimulator but
    cannot be detected statically (the speed is applied via controls at
    simulation time); those surface as batch errors and are documented in
    the report from the captured error message.
    """

    if stats["headloss"] != "H-W":
        return (
            f"uses {stats['headloss']} headloss, unsupported by "
            "WNTRSimulator (D3/D34)"
        )
    if max_junctions and stats["junctions"] > max_junctions:
        return (
            f"{stats['junctions']} junctions > cap {max_junctions} "
            "(too slow for WNTRSimulator)"
        )
    return None


def build_config_dict(
    path: Path,
    stats: dict,
    seed: int,
    duration_seconds: int,
    timestep_seconds: int,
) -> dict:
    """Assemble the PipelineConfig body for one sweep network."""

    mode = choose_demand_mode(stats)
    simulation = {
        "duration_seconds": duration_seconds,
        "hydraulic_timestep_seconds": timestep_seconds,
        "report_timestep_seconds": timestep_seconds,
        "demand_model": "PDD",
    }
    demand: dict = {"mode": mode}
    if mode == "fourier":
        # Advance the synthetic pattern hourly so the diurnal shape resolves.
        simulation["pattern_timestep_seconds"] = timestep_seconds
    return {
        "network": {"inp_path": str(path.relative_to(REPO_ROOT)), "name": safe_name(path)},
        "simulation": simulation,
        "seed": seed,
        "demand": demand,
        "scenario": {"type": "normal", "label": "normal"},
        "validation": dict(SWEEP_VALIDATION),
        "output": {
            "directory": "outputs/sweep",
            "formats": ["parquet", "csv"],
            "write_metadata_sidecar": True,
        },
    }


def _config_header(path: Path, stats: dict, mode: str, seed: int) -> str:
    return (
        f"# Sweep config (Phase 5 Week 8, WP3) generated by "
        f"scripts/generate_sweep_configs.py\n"
        f"# network: {path.name} | junctions={stats['junctions']} "
        f"pipes={stats['pipes']} tanks={stats['tanks']} "
        f"reservoirs={stats['reservoirs']}\n"
        f"# demand mode '{mode}' chosen: has_patterns={stats['has_patterns']}, "
        f"inp_duration={stats['inp_duration_seconds']}s\n"
        f"# seed={seed} | 24h / 1h PDD | conservative validation tolerances "
        f"(uncalibrated networks warn, never fail, on pressure)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--networks-dir", type=Path, default=DEFAULT_NETWORKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--timestep-seconds", type=int, default=DEFAULT_TIMESTEP_SECONDS)
    parser.add_argument(
        "--max-junctions",
        type=int,
        default=DEFAULT_MAX_JUNCTIONS,
        help="Skip networks with more junctions than this (too slow for "
        "WNTRSimulator). Set 0 to disable the cap.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    networks = discover_networks(args.networks_dir)
    if not networks:
        print(f"No .inp files found under {args.networks_dir}")
        return

    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    for path in networks:
        stats = network_stats(path)
        reason = skip_reason(stats, args.max_junctions)
        if reason is not None:
            skipped.append((path.name, reason))
            continue
        cfg_dict = build_config_dict(
            path, stats, args.seed, args.duration_seconds, args.timestep_seconds
        )
        # Validate before writing so we never emit a broken sweep config.
        PipelineConfig.model_validate(cfg_dict)
        mode = choose_demand_mode(stats)
        out_path = args.out_dir / f"{safe_name(path)}.yaml"
        text = _config_header(path, stats, mode, args.seed) + yaml.safe_dump(
            cfg_dict, sort_keys=False
        )
        out_path.write_text(text, encoding="utf-8")
        written.append(out_path.name)

    print(f"Wrote {len(written)} sweep configs to {args.out_dir}:")
    for name in written:
        print(f"  {name}")
    if skipped:
        print(f"\nSkipped {len(skipped)} network(s) (documented as excluded):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
