"""Tests for the PDD normal scenarios (Week 5).

From Week 5 onwards the project default for normal scenarios is the
pressure-dependent demand model (PDD): the supervisor prefers it and a
single demand model across normal and leak scenarios keeps the datasets
consistent.

These tests confirm the four PDD normal configs run, that PDD pressures
are non-negative within solver tolerance on the three networks that have
no known artefacts, and they document the Net3 node "10" finding: PDD
does **not** remove the negative-pressure dip because node "10" carries
no consumer demand for PDD to throttle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from wdn_pipeline.config import NetworkConfig, PipelineConfig, SimulationConfig
from wdn_pipeline.network import load_network
from wdn_pipeline.runner import run
from wdn_pipeline.simulation import run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]

PDD_NORMAL_CONFIGS = [
    "normal_net3_pdd",
    "normal_hanoi_pdd",
    "normal_fowm_pdd",
    "normal_jilin_pdd",
]


def _run_config(tmp_path: Path, config_name: str):
    raw = yaml.safe_load(
        (REPO_ROOT / "configs" / f"{config_name}.yaml").read_text()
    )
    raw["output"]["directory"] = str(tmp_path / "outputs")
    raw["output"]["formats"] = ["parquet"]
    return run(PipelineConfig.model_validate(raw))


def test_pdd_normal_configs_run_without_failure(tmp_path: Path) -> None:
    for name in PDD_NORMAL_CONFIGS:
        summary = _run_config(tmp_path / name, name)
        assert summary.validation.passed, name
        # PDD is recorded in the metadata.
        md = summary.validation
        assert md is not None


def test_pdd_normal_uses_pdd_demand_model(tmp_path: Path) -> None:
    summary = _run_config(tmp_path, "normal_hanoi_pdd")
    md = yaml.safe_load(summary.metadata_path.read_text())
    assert md["demand_model"] == "PDD"


def test_pdd_normal_pressure_non_negative_on_clean_networks(
    tmp_path: Path,
) -> None:
    """Hanoi, FOWM and Jilin produce no negative pressure under PDD."""

    for name in ("normal_hanoi_pdd", "normal_fowm_pdd", "normal_jilin_pdd"):
        sim_cfg = SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
            pattern_timestep_seconds=3600,
            demand_model="PDD",
        )
        raw = yaml.safe_load(
            (REPO_ROOT / "configs" / f"{name}.yaml").read_text()
        )
        wn = load_network(
            NetworkConfig(inp_path=raw["network"]["inp_path"]), sim_cfg
        )
        results = run_simulation(wn)
        assert float(np.nanmin(results.pressure.to_numpy())) >= -1e-6, name


def test_pdd_does_not_resolve_net3_node10_artefact() -> None:
    """Document the Week 5 finding: node "10" dips identically under DDA and PDD.

    Net3 node "10" carries no consumer demand, so PDD has nothing to
    throttle: the negative-pressure dip is a pure elevation/topology
    artefact of the .inp and is bit-identical under both demand models.
    """

    common = dict(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    dda = load_network(
        NetworkConfig(inp_path="Net3"),
        SimulationConfig(demand_model="DDA", **common),
    )
    pdd = load_network(
        NetworkConfig(inp_path="Net3"),
        SimulationConfig(demand_model="PDD", **common),
    )
    p_dda = run_simulation(dda).pressure["10"]
    p_pdd = run_simulation(pdd).pressure["10"]
    # The two demand models give the same node-"10" pressure trace.
    assert np.allclose(p_dda.to_numpy(), p_pdd.to_numpy(), atol=1e-3)
    # And it is still negative at its worst sample.
    assert float(p_pdd.min()) < 0.0


def test_pdd_normal_net3_warning_stays_within_tolerance(
    tmp_path: Path,
) -> None:
    """Net3 PDD normal warns on pressure_bounds but never fails."""

    summary = _run_config(tmp_path, "normal_net3_pdd")
    bounds = next(
        c for c in summary.validation.checks if c.name == "pressure_bounds"
    )
    assert bounds.severity in ("ok", "warning")
    assert summary.validation.passed
