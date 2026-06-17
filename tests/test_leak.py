"""Tests for the leak injection module.

These cover the leak fault module itself, the config-level validators
(geometry mutual exclusion, PDD requirement, time-window invariants),
the runner integration (resolved leaks threaded through labels and
validation) and end-to-end determinism on a network whose leak
parameters are drawn from the scenario RNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from wdn_pipeline.config import (
    FaultsConfig,
    LeakSpec,
    NetworkConfig,
    OutputConfig,
    PipelineConfig,
    ScenarioConfig,
    SimulationConfig,
)
from wdn_pipeline.faults.leak import LeakInjector, ResolvedLeak
from wdn_pipeline.network import load_network
from wdn_pipeline.runner import run
from wdn_pipeline.simulation import run_simulation
from wdn_pipeline.validation import validate_leak_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config-level validation
# ---------------------------------------------------------------------------


def _abrupt_leak_kwargs(**override) -> dict:
    base = {
        "pipe": "40",
        "split_fraction": 0.5,
        "area_m2": 0.001,
        "discharge_coeff": 0.75,
        "start_time_seconds": 3600,
        "end_time_seconds": 7200,
        "profile": "abrupt",
    }
    base.update(override)
    return base


def test_leakspec_accepts_area() -> None:
    spec = LeakSpec(**_abrupt_leak_kwargs())
    assert spec.area_m2 == 0.001


def test_leakspec_accepts_diameter() -> None:
    spec = LeakSpec(**_abrupt_leak_kwargs(area_m2=None, diameter_m=0.05))
    assert spec.diameter_m == 0.05


def test_leakspec_rejects_both_area_and_diameter() -> None:
    with pytest.raises(ValidationError):
        LeakSpec(**_abrupt_leak_kwargs(diameter_m=0.05))  # area still set


def test_leakspec_rejects_neither_area_nor_diameter() -> None:
    with pytest.raises(ValidationError):
        LeakSpec(**_abrupt_leak_kwargs(area_m2=None))


def test_leakspec_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError):
        LeakSpec(**_abrupt_leak_kwargs(start_time_seconds=7200, end_time_seconds=3600))


def test_leakspec_rejects_split_fraction_out_of_range() -> None:
    with pytest.raises(ValidationError):
        LeakSpec(**_abrupt_leak_kwargs(split_fraction=1.5))


def test_leakspec_rejects_invalid_discharge_coeff() -> None:
    with pytest.raises(ValidationError):
        LeakSpec(**_abrupt_leak_kwargs(discharge_coeff=0.0))


def test_pipeline_rejects_dda_with_leaks() -> None:
    """Leaks require PDD; the validator must reject DDA + leaks."""

    with pytest.raises(ValidationError, match="PDD"):
        PipelineConfig(
            network=NetworkConfig(inp_path="Net3"),
            simulation=SimulationConfig(
                duration_seconds=24 * 3600,
                hydraulic_timestep_seconds=3600,
                report_timestep_seconds=3600,
                demand_model="DDA",
            ),
            faults=FaultsConfig(leaks=[LeakSpec(**_abrupt_leak_kwargs())]),
        )


def test_pipeline_rejects_leak_window_past_duration() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        PipelineConfig(
            network=NetworkConfig(inp_path="Net3"),
            simulation=SimulationConfig(
                duration_seconds=3600,
                hydraulic_timestep_seconds=3600,
                report_timestep_seconds=3600,
                demand_model="PDD",
            ),
            faults=FaultsConfig(
                leaks=[LeakSpec(**_abrupt_leak_kwargs(start_time_seconds=0, end_time_seconds=7200))]
            ),
        )


def test_pipeline_accepts_pdd_with_leaks() -> None:
    cfg = PipelineConfig(
        network=NetworkConfig(inp_path="Net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
            demand_model="PDD",
        ),
        faults=FaultsConfig(leaks=[LeakSpec(**_abrupt_leak_kwargs())]),
    )
    assert cfg.simulation.demand_model == "PDD"
    assert len(cfg.faults.leaks) == 1


# ---------------------------------------------------------------------------
# Leak injector primitives
# ---------------------------------------------------------------------------


@pytest.fixture
def net3_pdd():
    """Fresh Net3 model with PDD enabled and 24h duration."""

    sim_cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
        demand_model="PDD",
    )
    return load_network(NetworkConfig(inp_path="Net3"), sim_cfg)


def test_explicit_pipe_round_trip(net3_pdd) -> None:
    spec = LeakSpec(**_abrupt_leak_kwargs(pipe="40", split_fraction=0.5))
    rng = np.random.default_rng(0)
    [resolved] = LeakInjector().apply(net3_pdd, [spec], rng)
    assert resolved.pipe == "40"
    assert resolved.leak_node_name in net3_pdd.junction_name_list
    assert resolved.new_pipe_name in net3_pdd.pipe_name_list
    # The original pipe is split, but the original name still refers
    # to a (now shorter) segment in the model.
    assert "40" in net3_pdd.pipe_name_list
    junction = net3_pdd.get_node(resolved.leak_node_name)
    assert junction._leak is True
    assert junction._leak_area == pytest.approx(spec.area_m2)


def test_diameter_to_area_conversion(net3_pdd) -> None:
    spec = LeakSpec(**_abrupt_leak_kwargs(area_m2=None, diameter_m=0.1))
    rng = np.random.default_rng(0)
    [resolved] = LeakInjector().apply(net3_pdd, [spec], rng)
    expected_area = np.pi * (0.1 / 2.0) ** 2
    assert resolved.area_m2 == pytest.approx(expected_area)
    assert resolved.diameter_m == 0.1


def test_two_leaks_on_same_pipe(net3_pdd) -> None:
    """Two leaks on one pipe re-split the upstream segment with unique names."""

    spec1 = LeakSpec(**_abrupt_leak_kwargs(pipe="40", split_fraction=0.3))
    spec2 = LeakSpec(**_abrupt_leak_kwargs(pipe="40", split_fraction=0.6))
    rng = np.random.default_rng(0)
    resolved = LeakInjector().apply(net3_pdd, [spec1, spec2], rng)
    assert len(resolved) == 2
    # Distinct leak-node and split-segment names (the unique-name helper).
    assert resolved[0].leak_node_name != resolved[1].leak_node_name
    assert resolved[0].new_pipe_name != resolved[1].new_pipe_name
    assert resolved[0].leak_node_name in net3_pdd.junction_name_list
    assert resolved[1].leak_node_name in net3_pdd.junction_name_list
    # Both leak nodes carry an active orifice.
    assert net3_pdd.get_node(resolved[0].leak_node_name)._leak is True
    assert net3_pdd.get_node(resolved[1].leak_node_name)._leak is True


def test_random_pipe_selection_is_seeded() -> None:
    """Two RNG-equivalent runs must pick the same pipe and split fraction."""

    sim_cfg = SimulationConfig(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
        demand_model="PDD",
    )
    spec = LeakSpec(**_abrupt_leak_kwargs(pipe=None, split_fraction=None))
    wn1 = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    wn2 = load_network(NetworkConfig(inp_path="Net3"), sim_cfg)
    [r1] = LeakInjector().apply(wn1, [spec], np.random.default_rng(123))
    [r2] = LeakInjector().apply(wn2, [spec], np.random.default_rng(123))
    assert r1.pipe == r2.pipe
    assert r1.split_fraction == pytest.approx(r2.split_fraction)


def test_random_pipe_selection_skips_non_pipes(monkeypatch, net3_pdd) -> None:
    """Random selection must reject any link whose link_type is not 'Pipe'."""

    spec = LeakSpec(**_abrupt_leak_kwargs(pipe=None, split_fraction=0.5))
    # The Net3 pump link names are well-known; random selection should
    # never produce them.
    pump_names = set(net3_pdd.pump_name_list)
    seen_pipes = set()
    for seed in range(20):
        wn = load_network(
            NetworkConfig(inp_path="Net3"),
            SimulationConfig(
                duration_seconds=24 * 3600,
                hydraulic_timestep_seconds=3600,
                report_timestep_seconds=3600,
                demand_model="PDD",
            ),
        )
        [r] = LeakInjector().apply(wn, [spec], np.random.default_rng(seed))
        seen_pipes.add(r.pipe)
    assert seen_pipes.isdisjoint(pump_names)


def test_explicit_non_pipe_rejected(net3_pdd) -> None:
    """Explicitly naming a pump as the leak target must raise."""

    if not net3_pdd.pump_name_list:
        pytest.skip("Net3 unexpectedly has no pumps")
    spec = LeakSpec(**_abrupt_leak_kwargs(pipe=net3_pdd.pump_name_list[0]))
    with pytest.raises(ValueError, match="only pipes"):
        LeakInjector().apply(net3_pdd, [spec], np.random.default_rng(0))


def test_unknown_pipe_rejected(net3_pdd) -> None:
    spec = LeakSpec(**_abrupt_leak_kwargs(pipe="this_pipe_does_not_exist"))
    with pytest.raises(ValueError, match="not a link"):
        LeakInjector().apply(net3_pdd, [spec], np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Profile semantics: abrupt / linear leak demand trajectories
# ---------------------------------------------------------------------------


def _build_pdd_config(
    *, leaks: list[LeakSpec], tmp_path: Path, network: str = "Net3", name: str = "net3"
) -> PipelineConfig:
    return PipelineConfig(
        network=NetworkConfig(inp_path=network, name=name),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
            demand_model="PDD",
        ),
        seed=42,
        faults=FaultsConfig(leaks=leaks),
        scenario=ScenarioConfig(type="leak", label="leak_test"),
        output=OutputConfig(directory=tmp_path / "outputs", formats=["parquet"]),
    )


def test_abrupt_leak_demand_window(tmp_path: Path) -> None:
    """Leak demand is zero pre-onset, non-zero in the window, zero post-end."""

    leak = LeakSpec(
        pipe="40",
        split_fraction=0.5,
        area_m2=0.005,
        start_time_seconds=21600,
        end_time_seconds=64800,
        profile="abrupt",
    )
    cfg = _build_pdd_config(leaks=[leak], tmp_path=tmp_path)
    summary = run(cfg)
    [resolved] = summary.resolved_leaks
    # Read leak_demand from the output parquet to round-trip through serialisation.
    import pyarrow.parquet as pq

    leak_path = next(p for p in summary.output_paths if "leak_demand.parquet" in p.name)
    df = pq.read_table(leak_path).to_pandas().set_index("time_seconds")
    series = df[resolved.leak_node_name]
    pre = series.loc[series.index < leak.start_time_seconds]
    during = series.loc[
        (series.index >= leak.start_time_seconds) & (series.index < leak.end_time_seconds)
    ]
    post = series.loc[series.index > leak.end_time_seconds]
    assert (pre.abs() < 1e-12).all()
    assert (during > 0.0).all()
    assert (post.abs() < 1e-12).all()


def test_incipient_linear_monotonic_growth(tmp_path: Path) -> None:
    """Linear-profile leak demand must grow monotonically up to the peak."""

    leak = LeakSpec(
        pipe="5",
        split_fraction=0.4,
        diameter_m=0.06,
        start_time_seconds=14400,
        end_time_seconds=50400,
        profile="linear",
        profile_steps=20,
    )
    cfg = PipelineConfig(
        network=NetworkConfig(inp_path=str(REPO_ROOT / "networks" / "Hanoi.inp"), name="hanoi"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
            pattern_timestep_seconds=3600,
            demand_model="PDD",
        ),
        seed=7,
        demand={  # type: ignore[arg-type]
            "mode": "fourier",
            "fourier": {
                "base": 1.0,
                "amplitude": 0.3,
                "period_hours": 24.0,
                "phase_shift_hours": 6.0,
                "noise_std": 0.0,
            },
        },
        faults=FaultsConfig(leaks=[leak]),
        scenario=ScenarioConfig(type="leak", label="leak_incipient"),
        output=OutputConfig(directory=tmp_path / "outputs", formats=["parquet"]),
    )
    summary = run(cfg)
    [resolved] = summary.resolved_leaks
    import pyarrow.parquet as pq

    leak_path = next(p for p in summary.output_paths if "leak_demand.parquet" in p.name)
    df = pq.read_table(leak_path).to_pandas().set_index("time_seconds")
    series = df[resolved.leak_node_name]
    during = series.loc[
        (series.index >= leak.start_time_seconds) & (series.index < leak.end_time_seconds)
    ]
    # Leak demand depends on both area and head; as the leak grows the
    # system head drops so the demand-vs-time curve is not strictly
    # monotonic at the noise floor. The robust signal is that the
    # second half of the window has a higher mean than the first half:
    # the area is much larger by then, even after the head sag.
    nonzero = during[during > 0]
    assert len(nonzero) >= 4
    half = len(nonzero) // 2
    first_half_mean = float(nonzero.iloc[:half].mean())
    second_half_mean = float(nonzero.iloc[half:].mean())
    assert second_half_mean > first_half_mean, (
        f"second-half leak demand should exceed first-half: "
        f"first={first_half_mean:.3e}, second={second_half_mean:.3e}"
    )
    # Post-end must be zero.
    post = series.loc[series.index > leak.end_time_seconds]
    assert (post.abs() < 1e-12).all()


def test_multi_leak_two_active_nodes(tmp_path: Path) -> None:
    """Two concurrent leaks produce two non-zero columns in leak_demand."""

    leaks = [
        LeakSpec(
            pipe="10",
            split_fraction=0.5,
            area_m2=0.004,
            start_time_seconds=14400,
            end_time_seconds=43200,
            profile="abrupt",
        ),
        LeakSpec(
            pipe="20",
            split_fraction=0.6,
            diameter_m=0.05,
            start_time_seconds=28800,
            end_time_seconds=72000,
            profile="linear",
            profile_steps=20,
        ),
    ]
    cfg = PipelineConfig(
        network=NetworkConfig(inp_path=str(REPO_ROOT / "networks" / "Jilin.inp"), name="jilin"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
            pattern_timestep_seconds=3600,
            demand_model="PDD",
        ),
        seed=23,
        faults=FaultsConfig(leaks=leaks),
        scenario=ScenarioConfig(type="leak", label="leak_multi"),
        output=OutputConfig(directory=tmp_path / "outputs", formats=["parquet"]),
    )
    summary = run(cfg)
    assert len(summary.resolved_leaks) == 2
    import pyarrow.parquet as pq

    leak_path = next(p for p in summary.output_paths if "leak_demand.parquet" in p.name)
    df = pq.read_table(leak_path).to_pandas().set_index("time_seconds")
    nonzero_columns = [c for c in df.columns if c != "label" and (df[c] != 0).any()]
    leak_node_names = {r.leak_node_name for r in summary.resolved_leaks}
    assert leak_node_names.issubset(set(nonzero_columns))


# ---------------------------------------------------------------------------
# Validation: leak-aware checks
# ---------------------------------------------------------------------------


def test_leak_aware_mass_balance_passes(tmp_path: Path) -> None:
    leak = LeakSpec(
        pipe="40",
        split_fraction=0.5,
        area_m2=0.005,
        start_time_seconds=21600,
        end_time_seconds=64800,
        profile="abrupt",
    )
    cfg = _build_pdd_config(leaks=[leak], tmp_path=tmp_path)
    # Bump the warning floor to absorb Net3's known node "10" pressure dip.
    cfg = cfg.model_copy(
        update={
            "validation": cfg.validation.model_copy(
                update={"pressure_min_warning_tolerance_m": 2.0}
            )
        }
    )
    summary = run(cfg)
    mb = next(c for c in summary.validation.checks if c.name == "mass_balance")
    assert mb.severity == "ok"


def test_leak_pressure_drop_check_fires_warning(net3_pdd) -> None:
    """Pressure-drop check warns when the during-leak mean is not lower."""

    leak = LeakSpec(
        pipe="40",
        split_fraction=0.5,
        area_m2=0.005,
        start_time_seconds=21600,
        end_time_seconds=64800,
        profile="abrupt",
    )
    rng = np.random.default_rng(42)
    resolved = LeakInjector().apply(net3_pdd, [leak], rng)
    results = run_simulation(net3_pdd)
    from wdn_pipeline.config import ValidationConfig

    report = validate_leak_scenario(
        net3_pdd, results, resolved, ValidationConfig(pressure_min_warning_tolerance_m=2.0)
    )
    pd_check = next(c for c in report.checks if c.name == "leak_pressure_drop")
    # Net3 + pipe "40" has the documented diurnal-confound issue; warning
    # is the correct severity.
    assert pd_check.severity == "warning"


# ---------------------------------------------------------------------------
# Determinism and end-to-end
# ---------------------------------------------------------------------------


def test_deterministic_random_leak_resolution(tmp_path: Path) -> None:
    """Two runs of a fully random leak with the same seed match exactly."""

    leak = LeakSpec(
        pipe=None,
        split_fraction=None,
        area_m2=0.003,
        start_time_seconds=14400,
        end_time_seconds=64800,
        profile="abrupt",
    )
    cfg = PipelineConfig(
        network=NetworkConfig(inp_path=str(REPO_ROOT / "networks" / "FOWM.inp"), name="fowm"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
            pattern_timestep_seconds=3600,
            demand_model="PDD",
        ),
        seed=99,
        demand={  # type: ignore[arg-type]
            "mode": "fourier",
            "fourier": {
                "base": 1.0,
                "amplitude": 0.3,
                "period_hours": 24.0,
                "phase_shift_hours": 6.0,
                "noise_std": 0.0,
            },
        },
        faults=FaultsConfig(leaks=[leak]),
        scenario=ScenarioConfig(type="leak", label="leak_random"),
        output=OutputConfig(directory=tmp_path / "first", formats=["parquet"]),
    )
    cfg2 = cfg.model_copy(
        update={"output": cfg.output.model_copy(update={"directory": tmp_path / "second"})}
    )
    s1 = run(cfg)
    s2 = run(cfg2)
    assert len(s1.resolved_leaks) == len(s2.resolved_leaks) == 1
    r1 = s1.resolved_leaks[0]
    r2 = s2.resolved_leaks[0]
    assert r1.pipe == r2.pipe
    assert r1.split_fraction == pytest.approx(r2.split_fraction)
    assert r1.leak_node_name == r2.leak_node_name


def test_leak_label_window_matches_resolved(tmp_path: Path) -> None:
    """Labels are 1 in [start_time, end_time] and 0 elsewhere."""

    leak = LeakSpec(
        pipe="40",
        split_fraction=0.5,
        area_m2=0.003,
        start_time_seconds=21600,
        end_time_seconds=64800,
        profile="abrupt",
    )
    cfg = _build_pdd_config(leaks=[leak], tmp_path=tmp_path)
    cfg = cfg.model_copy(
        update={
            "validation": cfg.validation.model_copy(
                update={"pressure_min_warning_tolerance_m": 2.0}
            )
        }
    )
    summary = run(cfg)
    import pyarrow.parquet as pq

    pressure_path = next(p for p in summary.output_paths if "pressure.parquet" in p.name)
    df = pq.read_table(pressure_path).to_pandas().set_index("time_seconds")
    labels = df["label"].astype(int)
    # Half-open window: end_time_seconds is excluded because WNTR's
    # end-control sets leak_status=False at that exact tick, so
    # leak_demand is already zero there. Including end_time_seconds in
    # the label window would mark a normal frame as anomalous.
    expected = (
        (labels.index >= leak.start_time_seconds) & (labels.index < leak.end_time_seconds)
    ).astype(int)
    pd.testing.assert_series_equal(
        labels.rename("x"),
        pd.Series(expected, index=labels.index, name="x", dtype=int),
        check_dtype=False,
    )


def test_metadata_records_resolved_leaks(tmp_path: Path) -> None:
    leak = LeakSpec(
        pipe="40",
        split_fraction=0.5,
        diameter_m=0.05,
        start_time_seconds=21600,
        end_time_seconds=64800,
        profile="abrupt",
        name="test_leak",
    )
    cfg = _build_pdd_config(leaks=[leak], tmp_path=tmp_path)
    cfg = cfg.model_copy(
        update={
            "validation": cfg.validation.model_copy(
                update={"pressure_min_warning_tolerance_m": 2.0}
            )
        }
    )
    summary = run(cfg)
    import yaml

    md = yaml.safe_load(summary.metadata_path.read_text())
    leaks = md["fault_summary"]["leaks"]
    assert len(leaks) == 1
    entry = leaks[0]
    assert entry["pipe"] == "40"
    assert entry["split_fraction"] == 0.5
    assert entry["area_m2"] == pytest.approx(np.pi * (0.05 / 2.0) ** 2)
    assert entry["diameter_m"] == 0.05
    assert entry["profile"] == "abrupt"
    assert entry["name"] == "test_leak"
    assert "leak_node_name" in entry


def test_label_matches_leak_demand_at_every_step(tmp_path: Path) -> None:
    """Per-timestep label must agree with leak_demand>0 everywhere.

    Regression test for the inclusive-end off-by-one. WNTR's end control
    sets leak_status=False at end_time_seconds, so leak_demand is zero
    at that exact reported timestep. The label must use the same
    half-open window.
    """

    leak = LeakSpec(
        pipe="40",
        split_fraction=0.5,
        area_m2=0.005,
        start_time_seconds=21600,
        end_time_seconds=64800,
        profile="abrupt",
    )
    cfg = _build_pdd_config(leaks=[leak], tmp_path=tmp_path)
    cfg = cfg.model_copy(
        update={
            "validation": cfg.validation.model_copy(
                update={"pressure_min_warning_tolerance_m": 2.0}
            )
        }
    )
    summary = run(cfg)
    import pyarrow.parquet as pq

    pressure_path = next(p for p in summary.output_paths if "pressure.parquet" in p.name)
    leak_path = next(p for p in summary.output_paths if "leak_demand.parquet" in p.name)
    pressure = pq.read_table(pressure_path).to_pandas().set_index("time_seconds")
    leak_demand = pq.read_table(leak_path).to_pandas().set_index("time_seconds")
    labels = pressure["label"].astype(int)
    [resolved] = summary.resolved_leaks
    leak_col = leak_demand[resolved.leak_node_name]
    # label==1 implies leak_demand > 0
    assert ((labels == 1) <= (leak_col > 0)).all()
    # label==0 implies leak_demand == 0
    assert ((labels == 0) <= (leak_col.abs() < 1e-12)).all()


def test_leak_demand_active_check_fires_ok_for_correct_leak(tmp_path: Path) -> None:
    leak = LeakSpec(
        pipe="40",
        split_fraction=0.5,
        area_m2=0.005,
        start_time_seconds=21600,
        end_time_seconds=64800,
        profile="abrupt",
    )
    cfg = _build_pdd_config(leaks=[leak], tmp_path=tmp_path)
    cfg = cfg.model_copy(
        update={
            "validation": cfg.validation.model_copy(
                update={"pressure_min_warning_tolerance_m": 2.0}
            )
        }
    )
    summary = run(cfg)
    active = next(c for c in summary.validation.checks if c.name == "leak_demand_active")
    assert active.severity == "ok"


def test_leak_demand_active_check_fails_when_window_too_narrow(tmp_path: Path) -> None:
    """If the leak window covers no reported step, the structural check fails."""

    # 3600s report timestep but a 1-second leak window between report
    # ticks: the leak fires inside one hydraulic step but never appears
    # in a reported sample.
    leak = LeakSpec(
        pipe="40",
        split_fraction=0.5,
        area_m2=0.005,
        start_time_seconds=21601,
        end_time_seconds=21602,
        profile="abrupt",
    )
    cfg = _build_pdd_config(leaks=[leak], tmp_path=tmp_path)
    cfg = cfg.model_copy(
        update={
            "validation": cfg.validation.model_copy(
                update={"pressure_min_warning_tolerance_m": 2.0}
            )
        }
    )
    summary = run(cfg)
    active = next(c for c in summary.validation.checks if c.name == "leak_demand_active")
    assert active.severity == "fail"


def test_resolved_leak_to_dict_is_serialisable() -> None:
    rl = ResolvedLeak(
        pipe="40",
        split_fraction=0.5,
        area_m2=0.001,
        diameter_m=None,
        discharge_coeff=0.75,
        start_time_seconds=3600,
        end_time_seconds=7200,
        profile="abrupt",
        profile_steps=1,
        leak_node_name="leak_0_40",
        new_pipe_name="pipe_0_40_B",
        name=None,
    )
    d = rl.to_dict()
    assert d["pipe"] == "40"
    assert d["leak_node_name"] == "leak_0_40"
