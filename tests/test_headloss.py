"""Tests for the Hazen-Williams head loss energy-conservation validator.

The validator recomputes per-pipe head loss from network geometry and the
simulated flow using WNTR's own constants, then compares it to the simulated
head difference. These tests assert it agrees with the simulator on converged
networks, catches deliberate corruptions, uses the signed form (so direction
matters), handles minor losses and zero flow, excludes pumps/valves/closed
pipes and runs on the clean frames under a flowrate sensor fault.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import wntr

from wdn_pipeline.config import (
    NetworkConfig,
    PipelineConfig,
    SimulationConfig,
    ValidationConfig,
)
from wdn_pipeline.network import load_network
from wdn_pipeline.runner import run
from wdn_pipeline.simulation import SimulationResults, run_simulation
from wdn_pipeline.validation import (
    _HW_DIAMETER_EXP,
    _HW_EXP,
    _HW_G,
    _HW_K,
    _HW_MINOR_EXP,
    _check_hazen_williams_headloss,
    _hw_minor_k,
    _hw_resistance,
    _predict_headloss,
)

CFG = ValidationConfig()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def net3() -> tuple[wntr.network.WaterNetworkModel, SimulationResults]:
    cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    results = run_simulation(wn)
    return wn, results


def _minor_loss_network() -> wntr.network.WaterNetworkModel:
    """Tiny reservoir -> pipe (with minor loss) -> junction network."""

    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R", base_head=100.0)
    wn.add_junction("J", base_demand=0.02, elevation=0.0)
    wn.add_pipe("P", "R", "J", length=1500.0, diameter=0.3, roughness=110, minor_loss=8.0)
    wn.options.time.duration = 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    return wn


# --------------------------------------------------------------------------- #
# Core: validator agrees with the simulator on converged networks
# --------------------------------------------------------------------------- #
def test_net3_normal_passes_tight(net3) -> None:
    wn, results = net3
    check = _check_hazen_williams_headloss(wn, results, CFG)
    assert check.name == "hazen_williams_headloss"
    assert check.severity == "ok", check.detail


def test_hanoi_normal_passes_tight() -> None:
    from pathlib import Path

    inp = Path(__file__).resolve().parents[1] / "networks" / "Hanoi.inp"
    if not inp.is_file():
        pytest.skip("Hanoi.inp not present")
    cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
        pattern_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path=str(inp)), cfg)
    results = run_simulation(wn)
    check = _check_hazen_williams_headloss(wn, results, cfg=CFG)
    assert check.severity == "ok", check.detail


# --------------------------------------------------------------------------- #
# Hand calculation guards the constants and exponents
# --------------------------------------------------------------------------- #
def test_hand_calculation_matches_predictor() -> None:
    # Independent long-hand calculation with the documented constants.
    L, D, C, ml = 1500.0, 0.3, 110.0, 8.0
    q = 0.0234
    k_manual = 10.666829500036352 * C ** (-1.852) * D ** (-4.871) * L
    minor_manual = 8.0 * ml / (9.81 * np.pi**2 * D**4)
    dh_manual = np.sign(q) * (k_manual * abs(q) ** 1.852 + minor_manual * abs(q) ** 2)

    k = _hw_resistance(np.array([C]), np.array([D]), np.array([L]))
    minor_k = _hw_minor_k(np.array([ml]), np.array([D]))
    dh_pred = _predict_headloss(k, minor_k, np.array([q]))
    assert dh_pred[0] == pytest.approx(dh_manual, rel=1e-12)


def test_predictor_matches_simulator_one_pipe(net3) -> None:
    """Hand-recompute one real pipe/timestep and match the simulated head drop."""

    wn, results = net3
    pipe = "329"  # an always-open Net3 pipe with appreciable flow
    link = wn.get_link(pipe)
    t = results.head.index[0]
    q = float(results.flowrate_clean.loc[t, pipe])
    k = _HW_K * link.roughness ** (-_HW_EXP) * link.diameter ** (-_HW_DIAMETER_EXP) * link.length
    minor_k = 8.0 * link.minor_loss / (_HW_G * np.pi**2 * link.diameter**4)
    dh_pred = np.sign(q) * (k * abs(q) ** _HW_EXP + minor_k * abs(q) ** _HW_MINOR_EXP)
    dh_obs = float(results.head.loc[t, link.start_node_name]) - float(
        results.head.loc[t, link.end_node_name]
    )
    assert dh_obs == pytest.approx(dh_pred, abs=CFG.headloss_tol_m)


# --------------------------------------------------------------------------- #
# Minor losses are read from the (post-simulation) model
# --------------------------------------------------------------------------- #
def test_minor_loss_pipe_passes_and_is_used() -> None:
    wn = _minor_loss_network()
    results = run_simulation(wn)
    # With the real minor_loss the prediction matches the simulator.
    check = _check_hazen_williams_headloss(wn, results, CFG)
    assert check.severity == "ok", check.detail
    # The minor-loss term is non-trivial here, so dropping it must break the
    # agreement: proves the check actually reads minor_loss from the model.
    wn.get_link("P").minor_loss = 0.0
    check_no_minor = _check_hazen_williams_headloss(wn, results, CFG)
    assert check_no_minor.severity != "ok", check_no_minor.detail


# --------------------------------------------------------------------------- #
# Corrupted head fails
# --------------------------------------------------------------------------- #
def test_corrupted_head_fails(net3) -> None:
    wn, results = net3
    bad_head = results.head.copy()
    # Offset one node's head by a large, physically impossible amount.
    node = bad_head.columns[5]
    bad_head[node] = bad_head[node] + 5.0
    corrupted = dataclasses.replace(results, head=bad_head)
    check = _check_hazen_williams_headloss(wn, corrupted, CFG)
    assert check.severity == "fail", check.detail


# --------------------------------------------------------------------------- #
# Zero-flow open pipes contribute zero residual, no NaN
# --------------------------------------------------------------------------- #
def test_predictor_zero_flow_no_nan() -> None:
    q = np.array([0.0, 0.0, 1e-12, -0.0])
    out = _predict_headloss(np.array([5.0]), np.array([0.0]), q)
    assert not np.any(np.isnan(out))
    assert out[0] == 0.0 and out[1] == 0.0


def test_zero_flow_open_pipe_zero_residual() -> None:
    """An open pipe carrying exactly zero flow with zero head drop is OK."""

    idx = pd.Index([0, 3600], name="time")
    wn = _minor_loss_network()
    run_simulation(wn)  # consume model so geometry is finalised
    # Synthetic results: zero flow, equal heads -> residual 0.
    flow = pd.DataFrame({"P": [0.0, 0.0]}, index=idx)
    head = pd.DataFrame({"R": [100.0, 100.0], "J": [100.0, 100.0]}, index=idx)
    status = pd.DataFrame({"P": [1, 1]}, index=idx)  # open
    results = SimulationResults(
        pressure=head[["J"]],
        flowrate=flow,
        demand=head[["J"]] * 0,
        leak_demand=head[["J"]] * 0,
        elapsed_seconds=0.0,
        pressure_clean=head[["J"]],
        flowrate_clean=flow,
        head=head,
        link_status=status,
    )
    check = _check_hazen_williams_headloss(wn, results, CFG)
    assert check.severity == "ok", check.detail
    assert "nan" not in check.detail.lower()


# --------------------------------------------------------------------------- #
# Closed pipes are excluded (Net3 pipe 330 is initially closed)
# --------------------------------------------------------------------------- #
def test_closed_pipe_excluded(net3) -> None:
    wn, results = net3
    # Net3 pipe "330" has initial_status Closed and is opened by a control;
    # its closed timesteps carry a large head difference at zero flow that
    # WNTR does not constrain. The check must still pass.
    assert wn.get_link("330").initial_status.name == "Closed"
    closed_steps = int((results.link_status["330"].to_numpy() == 0).sum())
    assert closed_steps > 0
    check = _check_hazen_williams_headloss(wn, results, CFG)
    assert check.severity == "ok", check.detail
    assert "excluded" in check.detail


def test_status_fallback_excludes_zero_flow(net3) -> None:
    """When link_status is absent, exactly-zero-flow pipes are excluded."""

    wn, results = net3
    no_status = dataclasses.replace(results, link_status=None)
    check = _check_hazen_williams_headloss(wn, no_status, CFG)
    assert check.severity == "ok", check.detail


# --------------------------------------------------------------------------- #
# Pumps and valves are excluded
# --------------------------------------------------------------------------- #
def test_pumps_excluded(net3) -> None:
    wn, results = net3
    # Net3 has pumps; if they were checked as pipes the head *gain* would
    # produce an enormous residual. The check passes because it iterates
    # pipe_name_list only.
    assert len(wn.pump_name_list) > 0
    for pump in wn.pump_name_list:
        assert pump not in wn.pipe_name_list
    check = _check_hazen_williams_headloss(wn, results, CFG)
    assert check.severity == "ok", check.detail


# --------------------------------------------------------------------------- #
# Signed form: reversing flow without reversing head difference fails
# --------------------------------------------------------------------------- #
def test_sign_sensitivity(net3) -> None:
    wn, results = net3
    # Flip the sign of every flow but leave heads untouched: the predicted
    # head loss reverses sign, so the residual roughly doubles everywhere a
    # pipe carries flow. The signed form must catch this.
    reversed_flow = -results.flowrate_clean
    corrupted = dataclasses.replace(results, flowrate=reversed_flow, flowrate_clean=reversed_flow)
    check = _check_hazen_williams_headloss(wn, corrupted, CFG)
    assert check.severity == "fail", check.detail


# --------------------------------------------------------------------------- #
# Clean-frame behaviour under a flowrate sensor fault
# --------------------------------------------------------------------------- #
def test_flowrate_sensor_fault_headloss_ok(tmp_path) -> None:
    cfg = PipelineConfig.model_validate(
        {
            "network": {"inp_path": "Net3", "name": "net3"},
            "simulation": {
                "duration_seconds": 6 * 3600,
                "hydraulic_timestep_seconds": 3600,
                "report_timestep_seconds": 3600,
            },
            "seed": 5,
            "scenario": {"type": "sensor_fault", "label": "flow_bias"},
            "faults": {
                "sensor_faults": [
                    {
                        "type": "bias",
                        "quantity": "flowrate",
                        "bias_value": 0.9,
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
    hl = next(c for c in summary.validation.checks if c.name == "hazen_williams_headloss")
    assert hl.severity == "ok", hl.detail


# --------------------------------------------------------------------------- #
# Config band validation
# --------------------------------------------------------------------------- #
def test_warning_band_must_exceed_ok_band() -> None:
    with pytest.raises(ValueError, match="headloss_warning_tol_m"):
        ValidationConfig(headloss_tol_m=1e-1, headloss_warning_tol_m=1e-3)


def test_warning_band_between_ok_and_fail(net3) -> None:
    wn, results = net3
    bad_head = results.head.copy()
    node = bad_head.columns[5]
    bad_head[node] = bad_head[node] + 0.01  # 1 cm: above ok (1e-3), below fail
    corrupted = dataclasses.replace(results, head=bad_head)
    tight = ValidationConfig(headloss_tol_m=1e-3, headloss_warning_tol_m=1.0)
    check = _check_hazen_williams_headloss(wn, corrupted, tight)
    assert check.severity == "warning", check.detail
