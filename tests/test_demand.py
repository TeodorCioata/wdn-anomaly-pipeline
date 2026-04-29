"""Tests for demand-pattern strategies."""

from __future__ import annotations

import numpy as np
import pytest

from wdn_pipeline.config import (
    DemandConfig,
    FourierDemandParams,
    NetworkConfig,
    ScaleDemandParams,
    SimulationConfig,
)
from wdn_pipeline.demand import (
    DefaultDemand,
    FourierDemand,
    ScaleDemand,
    apply_demand,
    build_strategy,
)
from wdn_pipeline.network import load_network


@pytest.fixture
def sim_cfg() -> SimulationConfig:
    return SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
        pattern_timestep_seconds=3600,
    )


@pytest.fixture
def net3(sim_cfg):
    return load_network(NetworkConfig(inp_path="Net3"), sim_cfg)


def _first_base_demand(wn) -> float:
    _, j = next(iter(wn.junctions()))
    return float(j.demand_timeseries_list[0].base_value)


def test_default_is_noop(net3) -> None:
    before = _first_base_demand(net3)
    DefaultDemand().apply(net3, seed=0)
    assert _first_base_demand(net3) == before


def test_scale_multiplies_base(net3) -> None:
    before = _first_base_demand(net3)
    ScaleDemand(ScaleDemandParams(multiplier=2.5)).apply(net3, seed=0)
    assert _first_base_demand(net3) == pytest.approx(before * 2.5)


def test_fourier_creates_pattern_with_expected_length(net3) -> None:
    params = FourierDemandParams(base=1.0, amplitude=0.3, period_hours=24.0, noise_std=0.0)
    FourierDemand(params).apply(net3, seed=0)
    assert "synthetic_fourier" in net3.pattern_name_list
    pattern = net3.get_pattern("synthetic_fourier")
    expected_len = (net3.options.time.duration // net3.options.time.pattern_timestep) + 1
    assert len(pattern.multipliers) == expected_len


def test_fourier_assigns_pattern_to_every_junction(net3) -> None:
    params = FourierDemandParams(base=1.0, amplitude=0.0, noise_std=0.0)
    FourierDemand(params).apply(net3, seed=0)
    for _, j in net3.junctions():
        for ts in j.demand_timeseries_list:
            assert ts.pattern_name == "synthetic_fourier"


def test_fourier_deterministic_for_same_seed(sim_cfg) -> None:
    params = FourierDemandParams(base=1.0, amplitude=0.5, noise_std=0.1)
    wn1 = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    wn2 = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    FourierDemand(params).apply(wn1, seed=123)
    FourierDemand(params).apply(wn2, seed=123)
    p1 = np.asarray(wn1.get_pattern("synthetic_fourier").multipliers)
    p2 = np.asarray(wn2.get_pattern("synthetic_fourier").multipliers)
    np.testing.assert_array_equal(p1, p2)


def test_fourier_different_seeds_give_different_patterns(sim_cfg) -> None:
    params = FourierDemandParams(base=1.0, amplitude=0.1, noise_std=0.5)
    wn1 = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    wn2 = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    FourierDemand(params).apply(wn1, seed=1)
    FourierDemand(params).apply(wn2, seed=2)
    p1 = np.asarray(wn1.get_pattern("synthetic_fourier").multipliers)
    p2 = np.asarray(wn2.get_pattern("synthetic_fourier").multipliers)
    assert not np.array_equal(p1, p2)


def test_fourier_multipliers_non_negative(sim_cfg) -> None:
    # Force the noise to dominate; the strategy should clip below zero.
    params = FourierDemandParams(base=0.1, amplitude=0.05, noise_std=2.0)
    wn = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    FourierDemand(params).apply(wn, seed=0)
    multipliers = np.asarray(wn.get_pattern("synthetic_fourier").multipliers)
    assert (multipliers >= 0.0).all()


def test_build_strategy_dispatch() -> None:
    assert isinstance(build_strategy(DemandConfig(mode="default")), DefaultDemand)
    assert isinstance(build_strategy(DemandConfig(mode="scale")), ScaleDemand)
    assert isinstance(build_strategy(DemandConfig(mode="fourier")), FourierDemand)


def test_apply_demand_dispatches(net3) -> None:
    before = _first_base_demand(net3)
    apply_demand(net3, DemandConfig(mode="scale", scale=ScaleDemandParams(multiplier=3.0)), seed=0)
    assert _first_base_demand(net3) == pytest.approx(before * 3.0)
