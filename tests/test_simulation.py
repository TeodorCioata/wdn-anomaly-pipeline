"""Tests for the WNTR simulation wrapper."""

from __future__ import annotations

import pytest

from wdn_pipeline.config import NetworkConfig, SimulationConfig
from wdn_pipeline.network import load_network
from wdn_pipeline.simulation import SimulationResults, run_simulation


@pytest.fixture
def sim_results() -> SimulationResults:
    cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    return run_simulation(wn)


def test_returns_three_dataframes(sim_results: SimulationResults) -> None:
    assert sim_results.pressure.shape[0] == 25  # 0..24h inclusive
    assert sim_results.flowrate.shape[0] == sim_results.pressure.shape[0]
    assert sim_results.demand.shape[0] == sim_results.pressure.shape[0]


def test_indices_aligned(sim_results: SimulationResults) -> None:
    assert (sim_results.pressure.index == sim_results.flowrate.index).all()
    assert (sim_results.pressure.index == sim_results.demand.index).all()


def test_elapsed_seconds_populated(sim_results: SimulationResults) -> None:
    assert sim_results.elapsed_seconds > 0.0


def test_time_index_in_seconds(sim_results: SimulationResults) -> None:
    assert sim_results.pressure.index[0] == 0
    assert sim_results.pressure.index[-1] == 24 * 3600


def test_no_nan_in_outputs(sim_results: SimulationResults) -> None:
    assert not sim_results.pressure.isna().any().any()
    assert not sim_results.flowrate.isna().any().any()
    assert not sim_results.demand.isna().any().any()
