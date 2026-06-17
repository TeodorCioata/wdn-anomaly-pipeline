"""Tests for cumulative scenarios (Week 5).

A cumulative scenario carries both leaks and sensor faults. The two
fault families compose through the existing runner flow with no
special-case orchestration: leaks modify the hydraulic model before
simulation, sensor faults corrupt the DataFrames afterwards.

Covers:

- End-to-end runs of the five shipped cumulative configs.
- The per-timestep label is the union of every leak and sensor-fault
  window; per-fault-type masks stay separate.
- The cumulative validator runs both the leak-specific and the
  sensor-fault structural checks.
- The informational interaction record when a sensor fault sits on a
  leak node.
- Determinism across both fault stages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from wdn_pipeline.config import (
    BiasFault,
    FaultsConfig,
    LeakSpec,
    NetworkConfig,
    OutputConfig,
    PipelineConfig,
    ScenarioConfig,
    SimulationConfig,
    ValidationConfig,
)
from wdn_pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]

CUMULATIVE_CONFIGS = [
    "cumulative_leak_bias_net3",
    "cumulative_leak_drift_hanoi",
    "cumulative_leak_dropout_jilin",
    "cumulative_leak_noise_fowm",
    "cumulative_leak_gain_net3",
]


def _net3_cumulative(
    tmp_path: Path,
    *,
    sensor_target: str = "15",
    leak_window: tuple[int, int] = (21600, 64800),
    sensor_window: tuple[int, int] = (3600, 14400),
) -> PipelineConfig:
    """A Net3 cumulative config: abrupt leak on pipe 40 plus a bias sensor."""

    return PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
            demand_model="PDD",
        ),
        seed=42,
        scenario=ScenarioConfig(type="cumulative", label="cumulative_test"),
        faults=FaultsConfig(
            leaks=[
                LeakSpec(
                    pipe="40",
                    split_fraction=0.5,
                    area_m2=0.005,
                    start_time_seconds=leak_window[0],
                    end_time_seconds=leak_window[1],
                    profile="abrupt",
                )
            ],
            sensor_faults=[
                BiasFault(
                    type="bias",
                    quantity="pressure",
                    target=sensor_target,
                    bias_value=2.0,
                    start_time_seconds=sensor_window[0],
                    end_time_seconds=sensor_window[1],
                )
            ],
        ),
        validation=ValidationConfig(pressure_min_warning_tolerance_m=2.0),
        output=OutputConfig(directory=tmp_path / "out", formats=["parquet"]),
    )


# ---------------------------------------------------------------------------
# End-to-end shipped configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", CUMULATIVE_CONFIGS)
def test_cumulative_config_runs_end_to_end(tmp_path: Path, config_name: str) -> None:
    yaml_path = REPO_ROOT / "configs" / f"{config_name}.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    raw["output"]["directory"] = str(tmp_path / "outputs")
    cfg = PipelineConfig.model_validate(raw)
    summary = run(cfg)
    # Both fault families resolved.
    assert summary.resolved_leaks
    assert summary.resolved_sensor_faults
    # Severity is ok or warning, never fail.
    assert summary.validation.passed


@pytest.mark.parametrize("config_name", CUMULATIVE_CONFIGS)
def test_cumulative_validator_runs_both_check_families(tmp_path: Path, config_name: str) -> None:
    yaml_path = REPO_ROOT / "configs" / f"{config_name}.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    raw["output"]["directory"] = str(tmp_path / "outputs")
    summary = run(PipelineConfig.model_validate(raw))
    names = {c.name for c in summary.validation.checks}
    assert "leak_demand_active" in names
    assert "sensor_fault_mask_consistent" in names
    assert "sensor_fault_signal_applied" in names
    # The structural leak and sensor checks must not fail.
    for check in summary.validation.checks:
        if check.name in (
            "leak_demand_active",
            "sensor_fault_mask_consistent",
        ):
            assert check.severity == "ok"


# ---------------------------------------------------------------------------
# Label union
# ---------------------------------------------------------------------------


def test_cumulative_label_is_union_of_windows(tmp_path: Path) -> None:
    cfg = _net3_cumulative(tmp_path, leak_window=(21600, 64800), sensor_window=(3600, 14400))
    summary = run(cfg)
    pressure_path = next(p for p in summary.output_paths if p.name.endswith("pressure.parquet"))
    df = pq.read_table(pressure_path).to_pandas().set_index("time_seconds")
    times = df.index.to_numpy()
    leak_mask = (times >= 21600) & (times < 64800)
    sensor_mask = (times >= 3600) & (times < 14400)
    expected = (leak_mask | sensor_mask).astype(int)
    assert list(df["label"].astype(int)) == list(expected)
    # The union genuinely spans both disjoint windows.
    assert df["label"].sum() > leak_mask.sum()


def test_cumulative_masks_are_separate_columns(tmp_path: Path) -> None:
    cfg = _net3_cumulative(tmp_path)
    summary = run(cfg)
    pressure_path = next(p for p in summary.output_paths if p.name.endswith("pressure.parquet"))
    df = pq.read_table(pressure_path).to_pandas().set_index("time_seconds")
    # The per-channel sensor mask is its own column, distinct from the
    # union label.
    assert "bias_mask_15" in df.columns
    assert "label" in df.columns
    times = df.index.to_numpy()
    sensor_mask = (times >= 3600) & (times < 14400)
    assert list(df["bias_mask_15"].astype(bool)) == list(sensor_mask)


# ---------------------------------------------------------------------------
# Interaction recording
# ---------------------------------------------------------------------------


def test_cumulative_interaction_recorded_when_sensor_on_leak_node(
    tmp_path: Path,
) -> None:
    """A pressure fault on the inserted leak node is recorded in metadata."""

    cfg = _net3_cumulative(tmp_path, sensor_target="leak_0_40")
    summary = run(cfg)
    assert len(summary.interactions) == 1
    interaction = summary.interactions[0]
    assert interaction["kind"] == "pressure_sensor_on_leak_node"
    assert interaction["leak_node"] == "leak_0_40"
    assert interaction["sensor_target"] == "leak_0_40"
    # The interaction is informational: it does not gate the run.
    assert summary.validation.passed
    md = yaml.safe_load(summary.metadata_path.read_text())
    assert md["fault_summary"]["interactions"]


def test_cumulative_no_interaction_when_channels_disjoint(
    tmp_path: Path,
) -> None:
    cfg = _net3_cumulative(tmp_path, sensor_target="15")
    summary = run(cfg)
    assert summary.interactions == []
    md = yaml.safe_load(summary.metadata_path.read_text())
    assert md["fault_summary"]["interactions"] == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_cumulative_is_deterministic(tmp_path: Path) -> None:
    cfg1 = _net3_cumulative(tmp_path / "a")
    cfg2 = _net3_cumulative(tmp_path / "b")
    s1 = run(cfg1)
    s2 = run(cfg2)
    p1 = next(p for p in s1.output_paths if p.name.endswith("pressure.parquet"))
    p2 = next(p for p in s2.output_paths if p.name.endswith("pressure.parquet"))
    df1 = pq.read_table(p1).to_pandas().set_index("time_seconds")
    df2 = pq.read_table(p2).to_pandas().set_index("time_seconds")
    pd.testing.assert_frame_equal(df1, df2)


def test_cumulative_random_leak_config_is_deterministic(
    tmp_path: Path,
) -> None:
    """The FOWM config draws a random leak; the same seed reproduces it."""

    raw = yaml.safe_load((REPO_ROOT / "configs" / "cumulative_leak_noise_fowm.yaml").read_text())
    raw["output"]["formats"] = ["parquet"]
    raw1 = dict(raw)
    raw1["output"] = {**raw["output"], "directory": str(tmp_path / "a")}
    raw2 = dict(raw)
    raw2["output"] = {**raw["output"], "directory": str(tmp_path / "b")}
    s1 = run(PipelineConfig.model_validate(raw1))
    s2 = run(PipelineConfig.model_validate(raw2))
    assert s1.resolved_leaks[0].pipe == s2.resolved_leaks[0].pipe
    assert s1.resolved_leaks[0].split_fraction == pytest.approx(s2.resolved_leaks[0].split_fraction)
