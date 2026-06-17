"""Tests for the optional ML loader (wdn_pipeline.ml)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wdn_pipeline.config import (
    BiasFault,
    FaultsConfig,
    NetworkConfig,
    OutputConfig,
    PipelineConfig,
    ScenarioConfig,
    SimulationConfig,
)
from wdn_pipeline.ml import WDNWindowDataset, scenario_train_test_split
from wdn_pipeline.runner import run

# ml.py imports without torch; only WDNWindowDataset needs it. The split
# helper is always tested; the Dataset tests skip without the ml extra.
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch (ml extra) not installed")


# -- scenario_train_test_split (no torch needed) --------------------------


def test_split_is_disjoint_and_covers_all() -> None:
    scenarios = [f"net3_normal_{i}" for i in range(50)]
    train, test = scenario_train_test_split(scenarios, test_fraction=0.3, seed=0)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(scenarios)


def test_split_is_deterministic_and_order_independent() -> None:
    scenarios = [f"net3_leak_{i}" for i in range(40)]
    a = scenario_train_test_split(scenarios, test_fraction=0.25, seed=7)
    b = scenario_train_test_split(list(reversed(scenarios)), test_fraction=0.25, seed=7)
    # Same membership regardless of input order (order within lists differs).
    assert set(a[0]) == set(b[0])
    assert set(a[1]) == set(b[1])


def test_split_fraction_is_approximately_respected() -> None:
    scenarios = [f"s_{i}" for i in range(400)]
    _train, test = scenario_train_test_split(scenarios, test_fraction=0.2, seed=1)
    assert 0.15 < len(test) / len(scenarios) < 0.25


def test_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        scenario_train_test_split(["a", "b"], test_fraction=1.5)


# -- WDNWindowDataset (torch) ---------------------------------------------


def _build_small_db(tmp_path: Path) -> Path:
    """Run a normal and a sensor scenario into a shared DuckDB file."""

    db_path = tmp_path / "ml.duckdb"
    normal = PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=SimulationConfig(
            duration_seconds=24 * 3600,
            hydraulic_timestep_seconds=3600,
            report_timestep_seconds=3600,
        ),
        seed=1,
        scenario=ScenarioConfig(type="normal", label="normal"),
        output=OutputConfig(
            directory=tmp_path / "out", formats=["parquet"], duckdb=True, duckdb_path=db_path
        ),
    )
    sensor = normal.model_copy(
        update={
            "scenario": ScenarioConfig(type="sensor_fault", label="sensor_bias"),
            "faults": FaultsConfig(
                sensor_faults=[
                    BiasFault(
                        type="bias",
                        quantity="pressure",
                        target="15",
                        start_time_seconds=21600,
                        end_time_seconds=64800,
                        bias_value=5.0,
                    )
                ]
            ),
        }
    )
    run(normal)
    run(sensor)
    return db_path


@requires_torch
def test_window_dataset_shapes_and_count(tmp_path: Path) -> None:
    db_path = _build_small_db(tmp_path)
    ds = WDNWindowDataset(
        db_path,
        scenarios=["net3_normal_1"],
        quantity="pressure",
        channels=None,
        window_length=6,
        stride=1,
    )
    # Net3 24h at 1h = 25 timesteps -> 25 - 6 + 1 = 20 windows for one scenario.
    assert len(ds) == 20
    window, label = ds[0]
    assert window.shape[0] == 6
    assert window.dtype == torch.float32
    assert label.dtype == torch.float32
    # Normal scenario: every window label is 0.
    assert all(float(ds[i][1]) == 0.0 for i in range(len(ds)))


@requires_torch
def test_window_dataset_flags_anomalous_windows(tmp_path: Path) -> None:
    db_path = _build_small_db(tmp_path)
    ds = WDNWindowDataset(
        db_path, scenarios=["net3_sensor_bias_1"], quantity="pressure", window_length=4
    )
    labels = [float(ds[i][1]) for i in range(len(ds))]
    # The bias fault spans midday, so at least one window is anomalous and
    # at least one (early morning) is not.
    assert any(v == 1.0 for v in labels)
    assert any(v == 0.0 for v in labels)


@requires_torch
def test_window_dataset_does_not_cross_scenarios(tmp_path: Path) -> None:
    db_path = _build_small_db(tmp_path)
    ds = WDNWindowDataset(
        db_path,
        scenarios=["net3_normal_1", "net3_sensor_bias_1"],
        quantity="pressure",
        window_length=6,
    )
    # 20 windows per scenario, two scenarios, no window spans the boundary.
    assert len(ds) == 40


@requires_torch
def test_window_dataset_requires_positive_window(tmp_path: Path) -> None:
    db_path = _build_small_db(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        WDNWindowDataset(db_path, scenarios=["net3_normal_1"], window_length=0)


@requires_torch
def test_window_dataset_respects_max_scenarios(tmp_path: Path) -> None:
    db_path = _build_small_db(tmp_path)
    with pytest.raises(ValueError, match="max_scenarios"):
        WDNWindowDataset(
            db_path,
            scenarios=["net3_normal_1", "net3_sensor_bias_1"],
            window_length=6,
            max_scenarios=1,
        )


@requires_torch
def test_window_dataset_cap_allows_within_limit(tmp_path: Path) -> None:
    db_path = _build_small_db(tmp_path)
    ds = WDNWindowDataset(db_path, scenarios=["net3_normal_1"], window_length=6, max_scenarios=4)
    assert len(ds) == 20
