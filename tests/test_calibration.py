"""Tests for the WNTR hydraulic calibration options (Week 8)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wdn_pipeline.config import (
    NetworkConfig,
    OutputConfig,
    PipelineConfig,
    ScenarioConfig,
    SimulationConfig,
)
from wdn_pipeline.network import apply_hydraulic_options, load_network
from wdn_pipeline.runner import run
from wdn_pipeline.simulation import run_simulation


def _sim_cfg(**overrides: object) -> SimulationConfig:
    base = {
        "duration_seconds": 2 * 3600,
        "hydraulic_timestep_seconds": 3600,
        "report_timestep_seconds": 3600,
    }
    base.update(overrides)
    return SimulationConfig(**base)


def test_allowlist_lands_on_hydraulic_options() -> None:
    cfg = _sim_cfg(
        viscosity=1.3,
        accuracy=0.0005,
        trials=77,
        demand_multiplier=1.2,
        specific_gravity=1.01,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    overrides = apply_hydraulic_options(wn, cfg)

    assert wn.options.hydraulic.viscosity == 1.3
    assert wn.options.hydraulic.accuracy == 0.0005
    assert wn.options.hydraulic.trials == 77
    assert wn.options.hydraulic.demand_multiplier == 1.2
    assert wn.options.hydraulic.specific_gravity == 1.01
    # One record per changed option, in allowlist order.
    changed = {o.option for o in overrides}
    assert changed == {
        "viscosity",
        "accuracy",
        "trials",
        "demand_multiplier",
        "specific_gravity",
    }


def test_none_leaves_inp_value_untouched() -> None:
    cfg = _sim_cfg()  # no calibration fields set
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    overrides = apply_hydraulic_options(wn, cfg)
    assert overrides == []
    # Net3 ships with the WNTR/EPANET defaults; nothing was changed.
    assert wn.options.hydraulic.accuracy == 0.001
    assert wn.options.hydraulic.trials == 40
    assert wn.options.hydraulic.demand_multiplier == 1.0


def test_override_record_carries_inp_and_config_values() -> None:
    cfg = _sim_cfg(trials=120)
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    (override,) = apply_hydraulic_options(wn, cfg)
    assert override.option == "trials"
    assert override.inp_value == 40
    assert override.config_value == 120
    assert override.to_dict() == {
        "option": "trials",
        "inp_value": 40,
        "config_value": 120,
    }


def _junction_demand_sum(cfg: SimulationConfig) -> float:
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    apply_hydraulic_options(wn, cfg)
    results = run_simulation(wn)
    junctions = [j for j in wn.junction_name_list if j in results.demand.columns]
    return float(np.abs(results.demand[junctions].to_numpy()).sum())


def test_demand_multiplier_changes_simulation_output() -> None:
    base = _junction_demand_sum(_sim_cfg())  # multiplier left at .inp default 1.0
    scaled = _junction_demand_sum(_sim_cfg(demand_multiplier=1.5))
    # Under DDA delivered junction demand is base * pattern * multiplier,
    # so the total scales exactly with the multiplier ratio (1.5 / 1.0).
    assert scaled == pytest.approx(1.5 * base, rel=1e-6)


def test_required_pressure_changes_pdd_output() -> None:
    # Under PDD a higher required_pressure throttles delivered demand at
    # nodes that cannot reach it, so the pressure field must change.
    low = _sim_cfg(demand_model="PDD", required_pressure=0.07)
    high = _sim_cfg(demand_model="PDD", required_pressure=40.0)

    def _pressure(cfg: SimulationConfig) -> np.ndarray:
        wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
        apply_hydraulic_options(wn, cfg)
        return run_simulation(wn).pressure.to_numpy()

    assert not np.allclose(_pressure(low), _pressure(high))


def test_unknown_passthrough_key_raises_at_network_prep() -> None:
    cfg = _sim_cfg(extra_hydraulic_options={"not_a_real_option": 1})
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    with pytest.raises(ValueError, match="Unknown WNTR hydraulic option"):
        apply_hydraulic_options(wn, cfg)


def test_valid_passthrough_key_applied_and_recorded() -> None:
    cfg = _sim_cfg(extra_hydraulic_options={"checkfreq": 4})
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    (override,) = apply_hydraulic_options(wn, cfg)
    assert wn.options.hydraulic.checkfreq == 4
    assert override.option == "checkfreq"
    assert override.config_value == 4


def test_pdd_only_option_rejected_under_dda() -> None:
    with pytest.raises(ValidationError, match="pressure-dependent"):
        _sim_cfg(demand_model="DDA", required_pressure=0.1)
    with pytest.raises(ValidationError, match="pressure-dependent"):
        _sim_cfg(demand_model="DDA", minimum_pressure=1.0)
    with pytest.raises(ValidationError, match="pressure-dependent"):
        _sim_cfg(demand_model="DDA", pressure_exponent=0.6)


def test_pdd_only_option_allowed_under_pdd() -> None:
    cfg = _sim_cfg(demand_model="PDD", required_pressure=0.1, minimum_pressure=0.0)
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    apply_hydraulic_options(wn, cfg)
    assert wn.options.hydraulic.required_pressure == 0.1


def test_dw_headloss_rejected_under_wntr_simulator() -> None:
    # Config load accepts D-W (it is a valid EPANET formula); network
    # prep rejects it because WNTRSimulator is H-W only.
    cfg = _sim_cfg(headloss="D-W")
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    with pytest.raises(ValueError, match="WNTRSimulator"):
        apply_hydraulic_options(wn, cfg)


def test_hw_headloss_accepted() -> None:
    cfg = _sim_cfg(headloss="H-W")
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    overrides = apply_hydraulic_options(wn, cfg)
    assert wn.options.hydraulic.headloss == "H-W"
    assert any(o.option == "headloss" for o in overrides)


def test_overrides_recorded_in_run_summary_and_sidecar(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=_sim_cfg(demand_multiplier=1.1, trials=60),
        seed=1,
        scenario=ScenarioConfig(type="normal", label="calib"),
        output=OutputConfig(
            directory=tmp_path / "outputs",
            formats=["parquet"],
            write_metadata_sidecar=True,
        ),
    )
    summary = run(cfg)
    options = {o.option for o in summary.option_overrides}
    assert {"demand_multiplier", "trials"} <= options

    import yaml

    meta = yaml.safe_load(summary.metadata_path.read_text())
    recorded = {o["option"]: o["config_value"] for o in meta["calibration_overrides"]}
    assert recorded["demand_multiplier"] == 1.1
    assert recorded["trials"] == 60
