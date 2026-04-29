"""Tests for the network loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from wdn_pipeline.config import NetworkConfig, SimulationConfig
from wdn_pipeline.network import derive_network_name, load_network


@pytest.fixture
def sim_cfg() -> SimulationConfig:
    return SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
        pattern_timestep_seconds=3600,
    )


def test_load_bundled_net3(sim_cfg: SimulationConfig) -> None:
    wn = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    assert wn.num_junctions == 92
    assert wn.options.time.duration == 24 * 3600
    assert wn.options.time.hydraulic_timestep == 3600
    assert wn.options.time.pattern_timestep == 3600


def test_load_from_file(hanoi_inp: Path, sim_cfg: SimulationConfig) -> None:
    wn = load_network(NetworkConfig(inp_path=str(hanoi_inp), name="hanoi"), sim_cfg)
    assert wn.num_junctions == 31


def test_unknown_network_raises(sim_cfg: SimulationConfig) -> None:
    with pytest.raises(FileNotFoundError):
        load_network(NetworkConfig(inp_path="not_a_real_network_xyz"), sim_cfg)


def test_returns_fresh_model(sim_cfg: SimulationConfig) -> None:
    wn1 = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    wn2 = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    assert wn1 is not wn2
    # mutating one must not bleed into the other
    wn1.options.time.duration = 12345
    assert wn2.options.time.duration == 24 * 3600


def test_derive_network_name_uses_explicit_name() -> None:
    name = derive_network_name(NetworkConfig(inp_path="Net3", name="custom"))
    assert name == "custom"


def test_derive_network_name_strips_extension(tmp_path: Path) -> None:
    p = tmp_path / "My Net.inp"
    p.touch()
    assert derive_network_name(NetworkConfig(inp_path=str(p))) == "My_Net"


def test_pattern_timestep_optional_keeps_inp_default(hanoi_inp: Path) -> None:
    cfg = SimulationConfig(
        duration_seconds=3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path=str(hanoi_inp)), cfg)
    # Hanoi.inp's default pattern_timestep is 3600.
    assert wn.options.time.pattern_timestep == 3600
