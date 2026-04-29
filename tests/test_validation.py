"""Tests for the physics-based validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wdn_pipeline.config import NetworkConfig, SimulationConfig, ValidationConfig
from wdn_pipeline.network import load_network
from wdn_pipeline.simulation import SimulationResults, run_simulation
from wdn_pipeline.validation import (
    ValidationCheck,
    ValidationReport,
    residuals,
    validate_normal_scenario,
)


@pytest.fixture
def net3_results() -> tuple:
    cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    results = run_simulation(wn)
    return wn, results


def _fake_results(pressure: pd.DataFrame) -> SimulationResults:
    flow = pd.DataFrame(0.0, index=pressure.index, columns=["L1"])
    demand = pd.DataFrame(0.0, index=pressure.index, columns=pressure.columns)
    return SimulationResults(pressure=pressure, flowrate=flow, demand=demand, elapsed_seconds=0.0)


def test_check_finite_fails_on_nan(net3_results) -> None:
    wn, results = net3_results
    p = results.pressure.copy()
    p.iloc[0, 0] = np.nan
    bad = SimulationResults(p, results.flowrate, results.demand, results.elapsed_seconds)
    report = validate_normal_scenario(wn, bad, ValidationConfig())
    fin = next(c for c in report.checks if c.name == "finite_values")
    assert fin.severity == "fail"
    assert report.severity == "fail"
    assert not report.passed


def test_pressure_bounds_warning(net3_results) -> None:
    """Net3 produces a small negative pressure at node '10' under DDA.

    With the default tolerance of 1.0 m it must be a warning, not a fail.
    """

    wn, results = net3_results
    report = validate_normal_scenario(wn, results, ValidationConfig())
    pb = next(c for c in report.checks if c.name == "pressure_bounds")
    assert pb.severity == "warning"
    assert report.passed  # warning does not fail


def test_pressure_bounds_fail_below_tolerance(net3_results) -> None:
    wn, results = net3_results
    cfg = ValidationConfig(pressure_min_m=0.0, pressure_min_warning_tolerance_m=0.1)
    report = validate_normal_scenario(wn, results, cfg)
    pb = next(c for c in report.checks if c.name == "pressure_bounds")
    assert pb.severity == "fail"


def test_pressure_bounds_strict_ok_for_hanoi(hanoi_config) -> None:
    from wdn_pipeline.demand import apply_demand

    wn = load_network(hanoi_config.network, hanoi_config.simulation)
    apply_demand(wn, hanoi_config.demand, hanoi_config.seed)
    results = run_simulation(wn)
    report = validate_normal_scenario(wn, results, ValidationConfig())
    assert report.severity == "ok"
    assert report.passed


def test_mass_balance_passes(net3_results) -> None:
    wn, results = net3_results
    report = validate_normal_scenario(wn, results, ValidationConfig())
    mb = next(c for c in report.checks if c.name == "mass_balance")
    assert mb.severity == "ok"


def test_report_severity_aggregation() -> None:
    rep = ValidationReport(
        checks=[
            ValidationCheck("a", "ok", ""),
            ValidationCheck("b", "warning", ""),
        ]
    )
    assert rep.severity == "warning"
    assert rep.passed is True

    rep2 = ValidationReport(
        checks=[
            ValidationCheck("a", "warning", ""),
            ValidationCheck("b", "fail", ""),
        ]
    )
    assert rep2.severity == "fail"
    assert rep2.passed is False


def test_residuals_near_zero_for_identical_runs(net3_results) -> None:
    """Two simulations with identical inputs must agree to numerical noise.

    WNTRSimulator uses an iterative Newton solver and is not bit-exact
    across calls. A residual on the order of double-precision epsilon
    (~1e-14 m) is normal and expected; anything larger would indicate a
    real source of nondeterminism.
    """

    cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn1 = load_network(NetworkConfig(inp_path="Net3"), cfg)
    wn2 = load_network(NetworkConfig(inp_path="Net3"), cfg)
    r1 = run_simulation(wn1)
    r2 = run_simulation(wn2)
    diff = residuals(r1.pressure, r2.pressure)
    assert float(np.abs(diff.to_numpy()).max()) < 1e-12
