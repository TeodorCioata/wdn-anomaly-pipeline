"""Tests for config schema loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from wdn_pipeline.config import (
    DemandConfig,
    PipelineConfig,
    SimulationConfig,
    ValidationConfig,
    load_config,
)


def _minimal_yaml(**override: object) -> dict:
    base = {
        "network": {"inp_path": "Net3", "name": "net3"},
        "simulation": {
            "duration_seconds": 86400,
            "hydraulic_timestep_seconds": 3600,
            "report_timestep_seconds": 3600,
        },
        "seed": 1,
    }
    base.update(override)  # type: ignore[arg-type]
    return base


def test_load_minimal_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(_minimal_yaml()))
    cfg = load_config(cfg_path)
    assert cfg.network.inp_path == "Net3"
    assert cfg.simulation.duration_seconds == 86400
    assert cfg.demand.mode == "default"
    assert cfg.scenario.type == "normal"
    assert "parquet" in cfg.output.formats


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_extra_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(
            {
                "network": {"inp_path": "Net3"},
                "simulation": {"duration_seconds": 3600},
                "extra": "not allowed",
            }
        )


def test_report_timestep_must_divide_hydraulic() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(
            duration_seconds=3600,
            hydraulic_timestep_seconds=600,
            report_timestep_seconds=900,
        )


def test_duration_must_divide_hydraulic() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(duration_seconds=4000, hydraulic_timestep_seconds=3600)


def test_config_is_frozen() -> None:
    cfg = PipelineConfig(
        network={"inp_path": "Net3"},
        simulation={"duration_seconds": 3600, "hydraulic_timestep_seconds": 3600},
    )
    with pytest.raises(ValidationError):
        cfg.seed = 99  # type: ignore[misc]


def test_demand_modes_typed() -> None:
    with pytest.raises(ValidationError):
        DemandConfig(mode="invalid_mode")  # type: ignore[arg-type]


def test_validation_defaults() -> None:
    v = ValidationConfig()
    assert v.pressure_min_m == 0.0
    assert v.pressure_min_warning_tolerance_m == 1.0
    assert v.pressure_max_m == 150.0


def test_output_formats_must_be_unique() -> None:
    from wdn_pipeline.config import OutputConfig

    with pytest.raises(ValidationError):
        OutputConfig(formats=["parquet", "parquet"])


# ---------------------------------------------------------------------------
# Sensor-fault discriminated union
# ---------------------------------------------------------------------------


def _sensor_cfg(fault: dict) -> dict:
    """Minimal config dict carrying a single sensor fault."""

    return {
        "network": {"inp_path": "Net3", "name": "net3"},
        "simulation": {
            "duration_seconds": 86400,
            "hydraulic_timestep_seconds": 3600,
            "report_timestep_seconds": 3600,
        },
        "scenario": {"type": "sensor_fault", "label": "x"},
        "faults": {"sensor_faults": [fault]},
    }


def test_sensor_fault_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(
            _sensor_cfg(
                {
                    "type": "wibble",
                    "target": "15",
                    "start_time_seconds": 0,
                    "end_time_seconds": 3600,
                }
            )
        )


def test_sensor_fault_discriminator_routes_to_concrete_class() -> None:
    from wdn_pipeline.config import BiasFault, GainFault

    cfg = PipelineConfig.model_validate(
        _sensor_cfg(
            {
                "type": "gain",
                "target": "15",
                "gain_factor": 1.2,
                "start_time_seconds": 0,
                "end_time_seconds": 3600,
            }
        )
    )
    fault = cfg.faults.sensor_faults[0]
    assert isinstance(fault, GainFault)
    assert not isinstance(fault, BiasFault)
    assert fault.gain_factor == 1.2


@pytest.mark.parametrize(
    ("fault", "foreign_field"),
    [
        ({"type": "bias", "bias_value": 1.0}, "sigma"),
        ({"type": "drift", "slope_per_second": 0.1}, "gain_factor"),
        ({"type": "stuck"}, "bias_value"),
        ({"type": "dropout", "intervals": [[0, 1800]]}, "sigma"),
        ({"type": "noise", "sigma": 1.0}, "bias_value"),
        ({"type": "gain", "gain_factor": 1.2}, "slope_per_second"),
    ],
)
def test_sensor_subtype_rejects_foreign_field(fault: dict, foreign_field: str) -> None:
    """Each subtype forbids fields belonging to other fault types.

    Before the discriminated-union migration the single all-optional model
    silently accepted (and ignored) a foreign field; now the routed
    subtype rejects it via the inherited ``extra='forbid'``.
    """

    fault = {
        **fault,
        "target": "15",
        "start_time_seconds": 0,
        "end_time_seconds": 3600,
        foreign_field: 1.0,
    }
    with pytest.raises(ValidationError, match="not permitted"):
        PipelineConfig.model_validate(_sensor_cfg(fault))
