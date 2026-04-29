"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from wdn_pipeline.config import (
    NetworkConfig,
    OutputConfig,
    PipelineConfig,
    ScenarioConfig,
    SimulationConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hanoi_inp() -> Path:
    """Path to the bundled Hanoi.inp under ``networks/``."""

    p = REPO_ROOT / "networks" / "Hanoi.inp"
    if not p.is_file():
        pytest.skip(f"Hanoi.inp not present at {p}")
    return p


@pytest.fixture
def net3_config(tmp_path: Path) -> PipelineConfig:
    """Minimal Net3 normal-scenario config with outputs in ``tmp_path``."""

    return PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
        ),
        seed=42,
        scenario=ScenarioConfig(type="normal", label="normal"),
        output=OutputConfig(directory=tmp_path / "outputs", formats=["parquet", "csv"]),
    )


@pytest.fixture
def hanoi_config(tmp_path: Path, hanoi_inp: Path) -> PipelineConfig:
    """Minimal Hanoi normal-scenario config with outputs in ``tmp_path``."""

    return PipelineConfig(
        network=NetworkConfig(inp_path=str(hanoi_inp), name="hanoi"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
            pattern_timestep_seconds=3600,
        ),
        seed=7,
        scenario=ScenarioConfig(type="normal", label="normal"),
        output=OutputConfig(directory=tmp_path / "outputs", formats=["parquet"]),
    )
