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


def test_flowrate_sensor_fault_mass_balance_uses_clean(tmp_path) -> None:
    """A flowrate sensor fault must not break the mass-balance check.

    Regression: mass balance is a physics check and must run on the
    clean (uncorrupted) flowrate. A flowrate bias corrupts
    ``results.flowrate``; validating mass balance on the corrupted frame
    spuriously fails it. The fix routes mass balance to the clean frames
    in the sensor-fault validator, matching the cumulative validator.
    """

    from wdn_pipeline.config import PipelineConfig
    from wdn_pipeline.runner import run

    cfg = PipelineConfig.model_validate(
        {
            "network": {"inp_path": "Net3", "name": "net3"},
            "simulation": {
                "duration_seconds": 6 * 3600,
                "hydraulic_timestep_seconds": 3600,
                "report_timestep_seconds": 3600,
            },
            "seed": 3,
            "scenario": {"type": "sensor_fault", "label": "flow_bias"},
            "faults": {
                "sensor_faults": [
                    {
                        "type": "bias",
                        "quantity": "flowrate",
                        "bias_value": 0.5,  # large enough to wreck a corrupted balance
                        "start_time_seconds": 3600,
                        "end_time_seconds": 18000,
                    }
                ]
            },
            "output": {
                "directory": str(tmp_path / "out"),
                "formats": ["parquet"],
                "write_metadata_sidecar": False,
            },
        }
    )
    summary = run(cfg)
    mass = next(c for c in summary.validation.checks if c.name == "mass_balance")
    assert mass.severity == "ok", mass.detail
    assert summary.validation.severity != "fail"


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
    leak = pd.DataFrame(0.0, index=pressure.index, columns=pressure.columns)
    return SimulationResults(
        pressure=pressure,
        flowrate=flow,
        demand=demand,
        leak_demand=leak,
        elapsed_seconds=0.0,
        pressure_clean=pressure.copy(),
        flowrate_clean=flow.copy(),
    )


def test_check_finite_fails_on_nan(net3_results) -> None:
    wn, results = net3_results
    p = results.pressure.copy()
    p.iloc[0, 0] = np.nan
    bad = SimulationResults(
        pressure=p,
        flowrate=results.flowrate,
        demand=results.demand,
        leak_demand=results.leak_demand,
        elapsed_seconds=results.elapsed_seconds,
        pressure_clean=p.copy(),
        flowrate_clean=results.flowrate_clean,
    )
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


# ---------------------------------------------------------------------------
# n-aware noise std tolerance (Week 6 polish)
# ---------------------------------------------------------------------------


def _build_noise_fault(target: str, start: int, end: int, sigma: float):
    """Build a ResolvedSensorFault with every required field present."""

    from wdn_pipeline.faults.sensor import ResolvedSensorFault

    return ResolvedSensorFault(
        type="noise",
        quantity="pressure",
        target=target,
        start_time_seconds=start,
        end_time_seconds=end,
        bias_value=None,
        slope_per_second=None,
        intervals=None,
        fill_value=None,
        sigma=sigma,
        rng_offset=0,
        gain_factor=None,
        name=None,
    )


def _make_noise_results(
    n_steps: int, dt: int, sigma: float, seed: int
) -> tuple[SimulationResults, SimulationResults]:
    """Construct (clean, corrupted) result pairs differing only by additive noise."""

    import pandas as _pd

    times = [i * dt for i in range(n_steps)]
    clean_vals = np.full(n_steps, 50.0)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=n_steps)
    clean = _pd.DataFrame({"a": clean_vals}, index=_pd.Index(times, name="time_seconds"))
    corrupted = _pd.DataFrame({"a": clean_vals + noise}, index=clean.index)
    flow = _pd.DataFrame(0.0, index=clean.index, columns=["L1"])
    demand = _pd.DataFrame(0.0, index=clean.index, columns=["a"])
    leak = _pd.DataFrame(0.0, index=clean.index, columns=["a"])
    results = SimulationResults(
        pressure=corrupted,
        flowrate=flow,
        demand=demand,
        leak_demand=leak,
        elapsed_seconds=0.0,
        pressure_clean=clean,
        flowrate_clean=flow.copy(),
    )
    return results, results  # caller only needs the first


def test_noise_tolerance_widens_for_small_n() -> None:
    """A 24-sample run whose sample std lands ~25-30% off sigma still
    passes the n-aware tolerance band."""

    from wdn_pipeline.validation import _check_sensor_fault_signal_applied

    results, _ = _make_noise_results(n_steps=24, dt=3600, sigma=1.0, seed=99)
    fault = _build_noise_fault(target="a", start=0, end=24 * 3600, sigma=1.0)
    check = _check_sensor_fault_signal_applied(results, [fault], ValidationConfig())
    # The seed=99 draw has sample std ~0.74; the legacy 30% band would
    # fail it. The n-aware tolerance for n=24 widens to ~4/sqrt(46) =
    # ~0.59 sigma, so the check is OK.
    assert check.severity == "ok"


def test_noise_tolerance_remains_tight_for_large_n() -> None:
    """For large n the n-aware tolerance converges to the 30% floor."""

    from wdn_pipeline.validation import _check_sensor_fault_signal_applied

    # Use a large clean-sample-and-noise run; std should be close to sigma.
    results, _ = _make_noise_results(n_steps=2000, dt=60, sigma=2.0, seed=11)
    fault = _build_noise_fault(target="a", start=0, end=2000 * 60, sigma=2.0)
    check = _check_sensor_fault_signal_applied(results, [fault], ValidationConfig())
    assert check.severity == "ok"


def test_noise_tolerance_rejects_gross_sigma_mismatch() -> None:
    """A sample std that is way off the spec sigma still fails."""

    from wdn_pipeline.validation import _check_sensor_fault_signal_applied

    # Build a noise residual with sigma=5 but claim sigma=1.
    results, _ = _make_noise_results(n_steps=24, dt=3600, sigma=5.0, seed=0)
    fault = _build_noise_fault(target="a", start=0, end=24 * 3600, sigma=1.0)
    check = _check_sensor_fault_signal_applied(results, [fault], ValidationConfig())
    assert check.severity == "fail"


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
