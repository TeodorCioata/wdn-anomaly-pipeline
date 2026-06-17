"""Smoke tests for the LeakG3PD network sweep (Phase 5 Week 8).

These guard the sweep capability against regression: the config
generator helpers, and a handful of representative networks (small, mid,
larger, uncalibrated) run end-to-end through the pipeline producing
non-empty finite output. Runs use a short duration to stay fast while
exercising the same demand-mode detection and PDD code path as the full
24h sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_sweep_configs import (
    DEFAULT_MAX_JUNCTIONS,
    build_config_dict,
    choose_demand_mode,
    discover_networks,
    network_stats,
    safe_name,
    skip_reason,
)
from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORKS = REPO_ROOT / "networks"


def _network(name: str) -> Path:
    p = NETWORKS / name
    if not p.is_file():
        pytest.skip(f"{name} not present at {p}")
    return p


def test_discover_networks_includes_inp_and_INP() -> None:
    names = {p.name for p in discover_networks(NETWORKS)}
    assert "Net1.inp" in names
    # PA1 ships with an uppercase .INP extension.
    assert any(n.lower() == "pa1.inp" for n in names)


def test_choose_demand_mode() -> None:
    assert choose_demand_mode({"has_patterns": True, "inp_duration_seconds": 3600}) == "default"
    assert choose_demand_mode({"has_patterns": False, "inp_duration_seconds": 3600}) == "fourier"
    # Patterns but zero native duration (the Hanoi lesson) -> fourier.
    assert choose_demand_mode({"has_patterns": True, "inp_duration_seconds": 0}) == "fourier"


def test_skip_reason_dw_headloss() -> None:
    stats = network_stats(_network("Balerma.inp"))
    reason = skip_reason(stats, DEFAULT_MAX_JUNCTIONS)
    assert reason is not None
    assert "D-W" in reason


def test_skip_reason_large_network() -> None:
    stats = {"headloss": "H-W", "junctions": DEFAULT_MAX_JUNCTIONS + 1}
    reason = skip_reason(stats, DEFAULT_MAX_JUNCTIONS)
    assert reason is not None
    assert "junctions" in reason


def test_skip_reason_none_for_small_hw_network() -> None:
    stats = network_stats(_network("Net1.inp"))
    assert skip_reason(stats, DEFAULT_MAX_JUNCTIONS) is None


def test_safe_name_strips_spaces() -> None:
    assert safe_name(Path("networks/Water Sensor Network 2.inp")) == "Water_Sensor_Network_2"


def test_generated_config_validates() -> None:
    path = _network("modena.inp")
    stats = network_stats(path)
    cfg = build_config_dict(path, stats, seed=42, duration_seconds=86400, timestep_seconds=3600)
    # Must validate as a real PipelineConfig.
    PipelineConfig.model_validate(cfg)


def _run_network_short(path: Path, tmp_path: Path) -> object:
    """Run a sweep network end-to-end at a short (2h) duration."""

    stats = network_stats(path)
    cfg_dict = build_config_dict(
        path, stats, seed=42, duration_seconds=2 * 3600, timestep_seconds=3600
    )
    cfg_dict["output"] = {
        "directory": str(tmp_path / "outputs"),
        "formats": ["parquet"],
        "write_metadata_sidecar": False,
    }
    cfg = PipelineConfig.model_validate(cfg_dict)
    return run(cfg)


@pytest.mark.parametrize(
    "network",
    ["Net1.inp", "modena.inp", "PA1.INP", "ky2.inp"],
)
def test_representative_network_runs_end_to_end(network: str, tmp_path: Path) -> None:
    summary = _run_network_short(_network(network), tmp_path)
    # Non-empty, finite output and not a hard validation failure.
    assert summary.num_timesteps > 0
    assert summary.validation.severity in {"ok", "warning"}
    assert len(summary.output_paths) > 0


def test_dw_network_run_is_rejected(tmp_path: Path) -> None:
    # A Darcy-Weisbach network (Balerma) cannot run under WNTRSimulator;
    # the sweep excludes it up front, but a forced run must fail loudly
    # (WNTRSimulator raises NotImplementedError for D-W) rather than
    # silently produce garbage.
    with pytest.raises(NotImplementedError):
        _run_network_short(_network("Balerma.inp"), tmp_path)
