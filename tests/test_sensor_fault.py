"""Tests for the sensor-fault injection module.

Covers:

- Pydantic validation of the typed ``SensorFaultSpec`` model.
- Each fault type's primitive transformation (bias, drift, stuck,
  dropout, noise) at the injector level.
- Deterministic seeding for the noise fault and random target
  selection.
- End-to-end runs through the runner for each of the five YAML configs.
- Structural validators (``sensor_fault_mask_consistent``,
  ``sensor_fault_signal_applied``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from wdn_pipeline.config import (
    BiasFault,
    DriftFault,
    DropoutFault,
    FaultsConfig,
    GainFault,
    NetworkConfig,
    NoiseFault,
    OutputConfig,
    PipelineConfig,
    ScenarioConfig,
    SimulationConfig,
    StuckFault,
)
from wdn_pipeline.faults.sensor import (
    ResolvedSensorFault,
    SensorFaultInjector,
)
from wdn_pipeline.network import load_network
from wdn_pipeline.runner import run, run_from_config_file
from wdn_pipeline.simulation import SimulationResults, run_simulation
from wdn_pipeline.validation import validate_sensor_fault_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config-level validation
# ---------------------------------------------------------------------------


def test_bias_spec_requires_bias_value() -> None:
    with pytest.raises(ValidationError, match="bias_value"):
        BiasFault(
            type="bias",
            quantity="pressure",
            target="15",
            start_time_seconds=0,
            end_time_seconds=3600,
        )


def test_bias_spec_rejects_zero_bias() -> None:
    with pytest.raises(ValidationError, match="non-zero"):
        BiasFault(
            type="bias",
            target="15",
            bias_value=0.0,
            start_time_seconds=0,
            end_time_seconds=3600,
        )


def test_drift_spec_requires_slope() -> None:
    with pytest.raises(ValidationError, match="slope_per_second"):
        DriftFault(
            type="drift",
            target="15",
            start_time_seconds=0,
            end_time_seconds=3600,
        )


def test_noise_spec_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValidationError, match="sigma"):
        NoiseFault(
            type="noise",
            target="15",
            sigma=0.0,
            start_time_seconds=0,
            end_time_seconds=3600,
        )


def test_dropout_spec_rejects_interval_outside_window() -> None:
    with pytest.raises(ValidationError, match="outside the fault window"):
        DropoutFault(
            type="dropout",
            target="15",
            start_time_seconds=3600,
            end_time_seconds=7200,
            intervals=[(0, 1800)],
        )


def test_dropout_spec_requires_nonempty_intervals() -> None:
    with pytest.raises(ValidationError, match="intervals"):
        DropoutFault(
            type="dropout",
            target="15",
            start_time_seconds=0,
            end_time_seconds=3600,
            intervals=[],
        )


def test_sensor_spec_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="end_time_seconds"):
        BiasFault(
            type="bias",
            target="15",
            bias_value=1.0,
            start_time_seconds=7200,
            end_time_seconds=3600,
        )


def test_pipeline_rejects_sensor_fault_past_duration() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        PipelineConfig(
            network=NetworkConfig(inp_path="Net3"),
            simulation=SimulationConfig(
                duration_seconds=3600,
                hydraulic_timestep_seconds=3600,
                report_timestep_seconds=3600,
            ),
            faults=FaultsConfig(
                sensor_faults=[
                    BiasFault(
                        type="bias",
                        target="15",
                        bias_value=1.0,
                        start_time_seconds=0,
                        end_time_seconds=7200,
                    )
                ]
            ),
        )


# ---------------------------------------------------------------------------
# Injector primitives — fixtures
# ---------------------------------------------------------------------------


def _build_fake_results(n_steps: int = 24, dt: int = 3600) -> SimulationResults:
    """Build a small synthetic SimulationResults with deterministic signals.

    Pressure at node ``a`` is a clean diurnal-like ramp; node ``b`` is
    flat at 50 m. Flow on link ``L1`` is a small constant. The shape
    matches the conventions used downstream by the injector.
    """

    idx = pd.Index([i * dt for i in range(n_steps)], name="time_seconds")
    pressure = pd.DataFrame(
        {
            "a": np.linspace(40.0, 60.0, n_steps),
            "b": np.full(n_steps, 50.0),
        },
        index=idx,
    )
    flow = pd.DataFrame({"L1": np.full(n_steps, 0.1)}, index=idx)
    demand = pd.DataFrame(0.0, index=idx, columns=pressure.columns)
    leak = pd.DataFrame(0.0, index=idx, columns=pressure.columns)
    return SimulationResults(
        pressure=pressure,
        flowrate=flow,
        demand=demand,
        leak_demand=leak,
        elapsed_seconds=0.0,
        pressure_clean=pressure.copy(),
        flowrate_clean=flow.copy(),
    )


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------


def test_bias_applies_constant_offset_in_window() -> None:
    results = _build_fake_results()
    spec = BiasFault(
        type="bias",
        quantity="pressure",
        target="a",
        bias_value=3.5,
        start_time_seconds=7200,
        end_time_seconds=21600,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    inside = (times >= 7200) & (times < 21600)
    outside = ~inside
    diff = out.results.pressure["a"].to_numpy() - results.pressure_clean["a"].to_numpy()
    assert np.allclose(diff[inside], 3.5)
    assert np.allclose(diff[outside], 0.0)


def test_bias_does_not_touch_other_columns() -> None:
    results = _build_fake_results()
    spec = BiasFault(
        type="bias",
        target="a",
        bias_value=2.0,
        start_time_seconds=0,
        end_time_seconds=86400,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    pd.testing.assert_series_equal(out.results.pressure["b"], results.pressure["b"])
    pd.testing.assert_frame_equal(out.results.flowrate, results.flowrate)


def test_bias_preserves_clean_signal() -> None:
    results = _build_fake_results()
    spec = BiasFault(
        type="bias",
        target="a",
        bias_value=5.0,
        start_time_seconds=3600,
        end_time_seconds=10800,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    pd.testing.assert_frame_equal(out.results.pressure_clean, results.pressure_clean)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def test_drift_applies_linear_ramp_in_window() -> None:
    results = _build_fake_results()
    spec = DriftFault(
        type="drift",
        target="a",
        slope_per_second=0.001,
        start_time_seconds=3600,
        end_time_seconds=14400,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    inside = (times >= 3600) & (times < 14400)
    expected = 0.001 * (times[inside] - 3600)
    diff = (
        out.results.pressure["a"].to_numpy()[inside]
        - results.pressure_clean["a"].to_numpy()[inside]
    )
    assert np.allclose(diff, expected, atol=1e-12)


def test_drift_zero_outside_window() -> None:
    results = _build_fake_results()
    spec = DriftFault(
        type="drift",
        target="a",
        slope_per_second=0.005,
        start_time_seconds=10800,
        end_time_seconds=21600,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    outside = (times < 10800) | (times >= 21600)
    diff = (
        out.results.pressure["a"].to_numpy()[outside]
        - results.pressure_clean["a"].to_numpy()[outside]
    )
    assert np.allclose(diff, 0.0)


# ---------------------------------------------------------------------------
# Stuck
# ---------------------------------------------------------------------------


def test_stuck_freezes_at_start_value() -> None:
    results = _build_fake_results()
    spec = StuckFault(
        type="stuck",
        target="a",
        start_time_seconds=10800,
        end_time_seconds=36000,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    active = (times >= 10800) & (times < 36000)
    active_idx = np.where(active)[0]
    stuck_value = float(results.pressure_clean["a"].to_numpy()[active_idx[0]])
    inside = out.results.pressure["a"].to_numpy()[active]
    assert np.allclose(inside, stuck_value)


def test_stuck_leaves_outside_untouched() -> None:
    results = _build_fake_results()
    spec = StuckFault(
        type="stuck",
        target="a",
        start_time_seconds=7200,
        end_time_seconds=14400,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    outside = (times < 7200) | (times >= 14400)
    pd.testing.assert_series_equal(
        pd.Series(out.results.pressure["a"].to_numpy()[outside]),
        pd.Series(results.pressure_clean["a"].to_numpy()[outside]),
    )


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


def test_dropout_fills_nan_in_subintervals() -> None:
    results = _build_fake_results()
    spec = DropoutFault(
        type="dropout",
        target="a",
        start_time_seconds=0,
        end_time_seconds=86400,
        intervals=[(3600, 10800), (21600, 25200)],
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    in_sub = ((times >= 3600) & (times < 10800)) | ((times >= 21600) & (times < 25200))
    inside_vals = out.results.pressure["a"].to_numpy()[in_sub]
    outside_vals = out.results.pressure["a"].to_numpy()[~in_sub]
    assert np.all(np.isnan(inside_vals))
    assert not np.any(np.isnan(outside_vals))


def test_dropout_respects_custom_fill_value() -> None:
    results = _build_fake_results()
    spec = DropoutFault(
        type="dropout",
        target="a",
        fill_value=-1.0,
        start_time_seconds=0,
        end_time_seconds=86400,
        intervals=[(3600, 10800)],
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    in_sub = (times >= 3600) & (times < 10800)
    assert np.allclose(out.results.pressure["a"].to_numpy()[in_sub], -1.0)


def test_dropout_mask_matches_subintervals_only() -> None:
    """The mask exposed by the injector covers sub-intervals, not the outer window."""

    results = _build_fake_results()
    spec = DropoutFault(
        type="dropout",
        target="a",
        start_time_seconds=0,
        end_time_seconds=86400,
        intervals=[(3600, 7200)],
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    mask = out.masks["dropout_mask_a"]
    expected_active = int(((mask.index >= 3600) & (mask.index < 7200)).sum())
    assert int(mask.sum()) == expected_active


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def test_noise_corrupts_only_window() -> None:
    results = _build_fake_results(n_steps=100, dt=600)
    spec = NoiseFault(
        type="noise",
        target="a",
        sigma=0.5,
        start_time_seconds=12000,
        end_time_seconds=48000,
        rng_offset=7,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(123))
    times = out.results.pressure.index.to_numpy()
    inside = (times >= 12000) & (times < 48000)
    outside = ~inside
    diff = out.results.pressure["a"].to_numpy() - results.pressure_clean["a"].to_numpy()
    assert not np.allclose(diff[inside], 0.0)
    assert np.allclose(diff[outside], 0.0)


def test_noise_residual_statistics_match_sigma() -> None:
    """Large-sample mean ~ 0, std ~ sigma."""

    results = _build_fake_results(n_steps=2000, dt=60)
    spec = NoiseFault(
        type="noise",
        target="a",
        sigma=2.0,
        start_time_seconds=0,
        end_time_seconds=120000,
        rng_offset=3,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(11))
    diff = out.results.pressure["a"].to_numpy() - results.pressure_clean["a"].to_numpy()
    assert abs(float(diff.mean())) < 0.2
    assert abs(float(diff.std()) - 2.0) < 0.2


def test_noise_deterministic_for_same_seed() -> None:
    results1 = _build_fake_results(n_steps=200, dt=300)
    results2 = _build_fake_results(n_steps=200, dt=300)
    spec = NoiseFault(
        type="noise",
        target="a",
        sigma=1.0,
        start_time_seconds=0,
        end_time_seconds=60000,
        rng_offset=5,
    )
    out1 = SensorFaultInjector().apply(results1, [spec], np.random.default_rng(42))
    out2 = SensorFaultInjector().apply(results2, [spec], np.random.default_rng(42))
    np.testing.assert_array_equal(
        out1.results.pressure["a"].to_numpy(),
        out2.results.pressure["a"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Random target selection and end-to-end determinism
# ---------------------------------------------------------------------------


def test_random_target_selection_is_seeded(tmp_path: Path) -> None:
    sim_cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    results = run_simulation(wn)
    spec = BiasFault(
        type="bias",
        target=None,
        bias_value=1.5,
        start_time_seconds=0,
        end_time_seconds=86400,
    )
    out1 = SensorFaultInjector().apply(results, [spec], np.random.default_rng(99), wn=wn)
    out2 = SensorFaultInjector().apply(results, [spec], np.random.default_rng(99), wn=wn)
    assert out1.resolved[0].target == out2.resolved[0].target


def test_random_target_pressure_in_junctions(tmp_path: Path) -> None:
    sim_cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    results = run_simulation(wn)
    spec = BiasFault(
        type="bias",
        target=None,
        bias_value=1.5,
        start_time_seconds=0,
        end_time_seconds=86400,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(7), wn=wn)
    assert out.resolved[0].target in wn.junction_name_list


def test_invalid_target_raises_on_apply() -> None:
    results = _build_fake_results()
    spec = BiasFault(
        type="bias",
        target="not_a_real_node",
        bias_value=1.0,
        start_time_seconds=0,
        end_time_seconds=86400,
    )
    with pytest.raises(ValueError, match="not a column"):
        SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_mask_consistent_validator_passes_for_well_formed_run() -> None:
    sim_cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    results = run_simulation(wn)
    spec = BiasFault(
        type="bias",
        target="15",
        bias_value=1.0,
        start_time_seconds=21600,
        end_time_seconds=64800,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0), wn=wn)
    from wdn_pipeline.config import ValidationConfig

    report = validate_sensor_fault_scenario(
        wn, out.results, out.resolved, out.masks, ValidationConfig()
    )
    mc = next(c for c in report.checks if c.name == "sensor_fault_mask_consistent")
    sa = next(c for c in report.checks if c.name == "sensor_fault_signal_applied")
    assert mc.severity == "ok"
    assert sa.severity == "ok"


def test_signal_applied_validator_detects_tampering() -> None:
    """Hand-rolled corrupted frame should fail signal_applied."""

    sim_cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    results = run_simulation(wn)
    # Build the "resolved" object the injector would have produced...
    fault = ResolvedSensorFault(
        type="bias",
        quantity="pressure",
        target="15",
        start_time_seconds=21600,
        end_time_seconds=64800,
        bias_value=2.0,
        slope_per_second=None,
        intervals=None,
        fill_value=None,
        sigma=None,
        rng_offset=0,
        name=None,
    )
    # ... but corrupt with a different offset to force a mismatch.
    corrupted = results.pressure.copy()
    times = corrupted.index.to_numpy()
    active = (times >= 21600) & (times < 64800)
    corrupted.loc[active, "15"] = results.pressure_clean["15"].to_numpy()[active] + 9.9
    tampered_results = SimulationResults(
        pressure=corrupted,
        flowrate=results.flowrate,
        demand=results.demand,
        leak_demand=results.leak_demand,
        elapsed_seconds=results.elapsed_seconds,
        pressure_clean=results.pressure_clean,
        flowrate_clean=results.flowrate_clean,
    )
    mask = pd.Series(active, index=corrupted.index, dtype=bool)
    from wdn_pipeline.config import ValidationConfig

    report = validate_sensor_fault_scenario(
        wn,
        tampered_results,
        [fault],
        {"bias_mask_15": mask},
        ValidationConfig(),
    )
    sa = next(c for c in report.checks if c.name == "sensor_fault_signal_applied")
    assert sa.severity == "fail"


# ---------------------------------------------------------------------------
# End-to-end via YAML configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_name",
    [
        "sensor_bias_net3",
        "sensor_drift_net3",
        "sensor_stuck_net3",
        "sensor_dropout_net3",
        "sensor_noise_net3",
    ],
)
def test_sensor_config_end_to_end(tmp_path: Path, config_name: str) -> None:
    """Each shipped config runs and structural validators pass."""

    yaml_path = REPO_ROOT / "configs" / f"{config_name}.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    raw["output"]["directory"] = str(tmp_path / "outputs")
    cfg = PipelineConfig.model_validate(raw)
    summary = run(cfg)
    mc = next(c for c in summary.validation.checks if c.name == "sensor_fault_mask_consistent")
    sa = next(c for c in summary.validation.checks if c.name == "sensor_fault_signal_applied")
    assert mc.severity == "ok"
    assert sa.severity == "ok"


def test_end_to_end_label_window_matches_fault(tmp_path: Path) -> None:
    """Per-timestep label is 1 exactly inside the fault window."""

    cfg = PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
        ),
        seed=42,
        scenario=ScenarioConfig(type="sensor_fault", label="sensor_bias_test"),
        faults=FaultsConfig(
            sensor_faults=[
                BiasFault(
                    type="bias",
                    target="15",
                    bias_value=2.0,
                    start_time_seconds=21600,
                    end_time_seconds=64800,
                )
            ]
        ),
        output=OutputConfig(directory=tmp_path / "out", formats=["parquet"]),
    )
    summary = run(cfg)
    pressure_path = next(p for p in summary.output_paths if "pressure.parquet" in p.name)
    df = pq.read_table(pressure_path).to_pandas().set_index("time_seconds")
    labels = df["label"].astype(int)
    expected = ((labels.index >= 21600) & (labels.index < 64800)).astype(int)
    pd.testing.assert_series_equal(
        labels.rename("x"),
        pd.Series(expected, index=labels.index, name="x", dtype=int),
        check_dtype=False,
    )


def test_clean_table_preserved_and_differs_from_corrupted(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
        ),
        seed=42,
        scenario=ScenarioConfig(type="sensor_fault", label="sensor_bias_clean_test"),
        faults=FaultsConfig(
            sensor_faults=[
                BiasFault(
                    type="bias",
                    target="15",
                    bias_value=3.0,
                    start_time_seconds=21600,
                    end_time_seconds=64800,
                )
            ]
        ),
        output=OutputConfig(directory=tmp_path / "out", formats=["parquet"]),
    )
    summary = run(cfg)
    pressure_path = next(p for p in summary.output_paths if p.name.endswith("pressure.parquet"))
    pressure_clean_path = next(
        p for p in summary.output_paths if p.name.endswith("pressure_clean.parquet")
    )
    p = pq.read_table(pressure_path).to_pandas().set_index("time_seconds")
    pc = pq.read_table(pressure_clean_path).to_pandas().set_index("time_seconds")
    diff = (p["15"] - pc["15"]).to_numpy()
    times = p.index.to_numpy()
    inside = (times >= 21600) & (times < 64800)
    assert np.allclose(diff[inside], 3.0)
    assert np.allclose(diff[~inside], 0.0)


def test_end_to_end_is_deterministic(tmp_path: Path) -> None:
    """Same config run twice produces bit-identical corrupted output."""

    base = PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
        ),
        seed=42,
        scenario=ScenarioConfig(type="sensor_fault", label="sensor_noise_det"),
        faults=FaultsConfig(
            sensor_faults=[
                NoiseFault(
                    type="noise",
                    target="15",
                    sigma=0.5,
                    start_time_seconds=0,
                    end_time_seconds=86400,
                    rng_offset=2,
                )
            ]
        ),
        output=OutputConfig(directory=tmp_path / "first", formats=["parquet"]),
    )
    other = base.model_copy(
        update={"output": base.output.model_copy(update={"directory": tmp_path / "second"})}
    )
    s1 = run(base)
    s2 = run(other)
    path1 = next(p for p in s1.output_paths if p.name.endswith("pressure.parquet"))
    path2 = next(p for p in s2.output_paths if p.name.endswith("pressure.parquet"))
    df1 = pq.read_table(path1).to_pandas().set_index("time_seconds")
    df2 = pq.read_table(path2).to_pandas().set_index("time_seconds")
    pd.testing.assert_frame_equal(df1, df2)


def test_metadata_records_resolved_sensor_fault(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
        ),
        seed=42,
        scenario=ScenarioConfig(type="sensor_fault", label="sensor_drift_meta"),
        faults=FaultsConfig(
            sensor_faults=[
                DriftFault(
                    type="drift",
                    target="15",
                    slope_per_second=0.001,
                    start_time_seconds=3600,
                    end_time_seconds=86400,
                    name="meta_test",
                )
            ]
        ),
        output=OutputConfig(directory=tmp_path / "out", formats=["parquet"]),
    )
    summary = run(cfg)
    md = yaml.safe_load(summary.metadata_path.read_text())
    sensor_meta = md["fault_summary"]["sensor_faults"]
    assert len(sensor_meta) == 1
    entry = sensor_meta[0]
    assert entry["type"] == "drift"
    assert entry["target"] == "15"
    assert entry["slope_per_second"] == pytest.approx(0.001)
    assert entry["name"] == "meta_test"


def test_mask_column_emitted_in_pressure_table(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
        ),
        seed=42,
        scenario=ScenarioConfig(type="sensor_fault", label="sensor_bias_mask"),
        faults=FaultsConfig(
            sensor_faults=[
                BiasFault(
                    type="bias",
                    target="15",
                    bias_value=4.0,
                    start_time_seconds=21600,
                    end_time_seconds=64800,
                )
            ]
        ),
        output=OutputConfig(directory=tmp_path / "out", formats=["parquet"]),
    )
    summary = run(cfg)
    pressure_path = next(p for p in summary.output_paths if p.name.endswith("pressure.parquet"))
    df = pq.read_table(pressure_path).to_pandas().set_index("time_seconds")
    assert "bias_mask_15" in df.columns
    mask = df["bias_mask_15"].astype(bool).to_numpy()
    expected = (df.index.to_numpy() >= 21600) & (df.index.to_numpy() < 64800)
    assert np.array_equal(mask, expected)


def test_run_from_yaml_config(tmp_path: Path) -> None:
    """Use the CLI / YAML entry point with a shipped config."""

    src = REPO_ROOT / "configs" / "sensor_bias_net3.yaml"
    raw = yaml.safe_load(src.read_text())
    raw["output"]["directory"] = str(tmp_path / "outputs")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    summary = run_from_config_file(cfg_path)
    assert summary.validation.passed
    assert len(summary.resolved_sensor_faults) == 1


# ---------------------------------------------------------------------------
# Gain fault (Week 5)
# ---------------------------------------------------------------------------


def test_gain_spec_requires_gain_factor() -> None:
    with pytest.raises(ValidationError, match="gain_factor"):
        GainFault(
            type="gain",
            target="15",
            start_time_seconds=0,
            end_time_seconds=3600,
        )


def test_gain_spec_rejects_unit_gain() -> None:
    with pytest.raises(ValidationError, match="no-op"):
        GainFault(
            type="gain",
            target="15",
            gain_factor=1.0,
            start_time_seconds=0,
            end_time_seconds=3600,
        )


def test_gain_spec_rejects_zero_gain() -> None:
    with pytest.raises(ValidationError, match="zero gain"):
        GainFault(
            type="gain",
            target="15",
            gain_factor=0.0,
            start_time_seconds=0,
            end_time_seconds=3600,
        )


def test_gain_applies_multiplicative_factor_in_window() -> None:
    results = _build_fake_results()
    spec = GainFault(
        type="gain",
        target="a",
        gain_factor=1.2,
        start_time_seconds=7200,
        end_time_seconds=21600,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    inside = (times >= 7200) & (times < 21600)
    clean = results.pressure_clean["a"].to_numpy()
    corrupted = out.results.pressure["a"].to_numpy()
    assert np.allclose(corrupted[inside], 1.2 * clean[inside])
    assert np.allclose(corrupted[~inside], clean[~inside])


def test_gain_deterministic_for_same_seed() -> None:
    results = _build_fake_results()
    spec = GainFault(
        type="gain",
        target="a",
        gain_factor=0.85,
        start_time_seconds=0,
        end_time_seconds=86400,
    )
    out1 = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    out2 = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    pd.testing.assert_frame_equal(out1.results.pressure, out2.results.pressure)


def test_gain_resolved_records_factor() -> None:
    results = _build_fake_results()
    spec = GainFault(
        type="gain",
        target="a",
        gain_factor=1.15,
        start_time_seconds=0,
        end_time_seconds=86400,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0))
    assert out.resolved[0].gain_factor == pytest.approx(1.15)


def test_gain_config_end_to_end(tmp_path: Path) -> None:
    """The shipped sensor_gain_net3 config runs and validators pass."""

    yaml_path = REPO_ROOT / "configs" / "sensor_gain_net3.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    raw["output"]["directory"] = str(tmp_path / "outputs")
    cfg = PipelineConfig.model_validate(raw)
    summary = run(cfg)
    mc = next(c for c in summary.validation.checks if c.name == "sensor_fault_mask_consistent")
    sa = next(c for c in summary.validation.checks if c.name == "sensor_fault_signal_applied")
    assert mc.severity == "ok"
    # A 10% gain is well above the 5% detectability floor.
    assert sa.severity == "ok"


def test_gain_detectability_warning_for_small_gain() -> None:
    """A gain factor very close to 1.0 triggers the detectability warning."""

    from wdn_pipeline.config import ValidationConfig

    sim_cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    results = run_simulation(wn)
    spec = GainFault(
        type="gain",
        target="15",
        gain_factor=1.001,
        start_time_seconds=0,
        end_time_seconds=86400,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0), wn=wn)
    report = validate_sensor_fault_scenario(
        wn, out.results, out.resolved, out.masks, ValidationConfig()
    )
    sa = next(c for c in report.checks if c.name == "sensor_fault_signal_applied")
    # Structural part still holds (corrupted == gain * clean), so this is
    # a warning, not a fail.
    assert sa.severity == "warning"
    assert "undetectable" in sa.detail


def test_gain_detectability_ok_for_large_gain() -> None:
    """A clearly detectable gain leaves signal_applied at ok."""

    from wdn_pipeline.config import ValidationConfig

    sim_cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    wn = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    results = run_simulation(wn)
    spec = GainFault(
        type="gain",
        target="15",
        gain_factor=1.25,
        start_time_seconds=0,
        end_time_seconds=86400,
    )
    out = SensorFaultInjector().apply(results, [spec], np.random.default_rng(0), wn=wn)
    report = validate_sensor_fault_scenario(
        wn, out.results, out.resolved, out.masks, ValidationConfig()
    )
    sa = next(c for c in report.checks if c.name == "sensor_fault_signal_applied")
    assert sa.severity == "ok"


# ---------------------------------------------------------------------------
# Multiple faults on the same channel (hardening pass R1)
# ---------------------------------------------------------------------------


def test_two_faults_on_same_channel_compose() -> None:
    """Two different-type faults on one channel both appear in the output.

    Regression: previously each fault rewrote the whole column from the
    clean baseline, so only the last-applied fault survived. The injector
    now writes only each fault's active region, preserving earlier faults.
    """

    results = _build_fake_results(n_steps=24, dt=3600)
    bias = BiasFault(
        type="bias",
        target="a",
        bias_value=3.0,
        start_time_seconds=0,
        end_time_seconds=7200,  # first two steps
    )
    stuck = StuckFault(
        type="stuck",
        target="a",
        start_time_seconds=36000,  # later, disjoint window
        end_time_seconds=72000,
    )
    out = SensorFaultInjector().apply(results, [bias, stuck], np.random.default_rng(0))
    times = out.results.pressure.index.to_numpy()
    clean = results.pressure_clean["a"].to_numpy()
    corrupted = out.results.pressure["a"].to_numpy()
    bias_win = (times >= 0) & (times < 7200)
    stuck_win = (times >= 36000) & (times < 72000)
    untouched = ~bias_win & ~stuck_win
    # Bias effect survives even though it was applied first.
    assert np.allclose(corrupted[bias_win], clean[bias_win] + 3.0)
    # Stuck effect present (frozen at its first in-window clean value).
    stuck_value = clean[np.where(stuck_win)[0][0]]
    assert np.allclose(corrupted[stuck_win], stuck_value)
    # Everything outside both windows is the untouched clean signal.
    assert np.allclose(corrupted[untouched], clean[untouched])


def test_duplicate_same_type_target_raises() -> None:
    """Two faults of the same type on the same channel are rejected."""

    results = _build_fake_results()
    f1 = BiasFault(
        type="bias", target="a", bias_value=1.0, start_time_seconds=0, end_time_seconds=7200
    )
    f2 = BiasFault(
        type="bias", target="a", bias_value=2.0, start_time_seconds=7200, end_time_seconds=14400
    )
    with pytest.raises(ValueError, match="Duplicate sensor-fault mask"):
        SensorFaultInjector().apply(results, [f1, f2], np.random.default_rng(0))
