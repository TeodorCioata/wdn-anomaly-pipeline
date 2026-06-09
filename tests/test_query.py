"""Tests for the DuckDB query layer (Phase 5 Week 8, WP1, D32)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.query import DatasetQuery
from wdn_pipeline.runner import app, run

runner = CliRunner()


def _cfg(out_dir: Path, db: Path, label: str, scenario_type: str, seed: int, faults: dict | None) -> PipelineConfig:
    body: dict = {
        "network": {"inp_path": "Net3", "name": "net3"},
        "simulation": {
            "duration_seconds": 2 * 3600,
            "hydraulic_timestep_seconds": 3600,
            "report_timestep_seconds": 3600,
            "demand_model": "PDD",
        },
        "seed": seed,
        "scenario": {"type": scenario_type, "label": label},
        "validation": {"pressure_min_warning_tolerance_m": 50.0, "pressure_max_m": 500.0},
        "output": {
            "directory": str(out_dir),
            "formats": ["parquet"],
            "write_metadata_sidecar": False,
            "duckdb": True,
            "duckdb_path": str(db),
        },
    }
    if faults:
        body["faults"] = faults
    return PipelineConfig.model_validate(body)


@pytest.fixture(scope="module")
def query_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a small consolidated DuckDB with a mix of scenarios."""

    out_dir = tmp_path_factory.mktemp("qout")
    db = out_dir / "test.duckdb"
    leak = {"leaks": [{"area_m2": 0.01, "start_time_seconds": 3600, "end_time_seconds": 7200}]}
    sensor = {
        "sensor_faults": [
            {
                "type": "bias",
                "quantity": "pressure",
                "target": "10",
                "bias_value": 5.0,
                "start_time_seconds": 3600,
                "end_time_seconds": 7200,
            }
        ]
    }
    run(_cfg(out_dir, db, "normal", "normal", 0, None))
    run(_cfg(out_dir, db, "leak_a", "leak", 1, leak))
    run(_cfg(out_dir, db, "leak_b", "leak", 2, leak))
    run(_cfg(out_dir, db, "sensor", "sensor_fault", 3, sensor))
    return db


def test_scenarios_lists_all(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        df = q.scenarios()
    assert len(df) == 4
    assert set(df["scenario_basename"]) == {
        "net3_normal_0",
        "net3_leak_a_1",
        "net3_leak_b_2",
        "net3_sensor_3",
    }


def test_scenarios_filter_by_type(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        leaks = q.scenarios(scenario_type="leak")
    assert len(leaks) == 2
    assert all(t == "leak" for t in leaks["scenario_type"])


def test_pressure_scenario_filter(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        df = q.pressure(scenario="net3_normal_0")
    assert set(df["scenario_basename"]) == {"net3_normal_0"}
    assert len(df) > 0


def test_pressure_node_filter(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        df = q.pressure(scenario="net3_normal_0", nodes=["10", "15"])
    assert set(df["name"]) == {"10", "15"}


def test_time_window_is_half_open(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        df = q.pressure(scenario="net3_normal_0", nodes=["10"], t_start=3600, t_end=7200)
    # [3600, 7200): includes 3600, excludes 7200.
    assert sorted(df["time_seconds"].unique()) == [3600]


def test_composed_filters(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        df = q.pressure(
            scenario="net3_normal_0", nodes=["10"], t_start=0, t_end=3600
        )
    assert set(df["name"]) == {"10"}
    assert sorted(df["time_seconds"].unique()) == [0]


def test_cross_scenario_query(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        df = q.pressure(scenarios=["net3_leak_a_1", "net3_leak_b_2"], nodes=["10"])
    assert set(df["scenario_basename"]) == {"net3_leak_a_1", "net3_leak_b_2"}


def test_wide_roundtrips_against_wide_table(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        wide = q.pressure(scenario="net3_normal_0", nodes=["10", "15"], wide=True)
        # Compare against the per-scenario wide table directly.
        ref = q.sql(
            'SELECT time_seconds, "10", "15" FROM net3_normal_0_pressure '
            "ORDER BY time_seconds"
        ).set_index("time_seconds")
    assert wide.index.name == "time_seconds"
    assert list(wide.columns) == ["10", "15"]
    for node in ("10", "15"):
        assert wide[node].to_list() == pytest.approx(ref[node].to_list())


def test_read_only_cannot_write(query_db: Path) -> None:
    with DatasetQuery(query_db) as q, pytest.raises(duckdb.Error):
        q.sql("CREATE TABLE should_fail (a INTEGER)")


def test_parametrised_binding_blocks_injection(query_db: Path) -> None:
    malicious = "'; DROP TABLE pressure_long;--"
    with DatasetQuery(query_db) as q:
        df = q.pressure(scenario="net3_normal_0", nodes=[malicious])
        # Treated as data: no match, and the table survives.
        assert len(df) == 0
        n = q.sql("SELECT COUNT(*) AS c FROM pressure_long")["c"][0]
    assert n > 0


def test_labels_anomalous_only(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        anomalous = q.labels(scenario="net3_leak_a_1", label=1)
        allrows = q.labels(scenario="net3_leak_a_1")
    assert len(anomalous) > 0
    assert all(v == 1 for v in anomalous["label"])
    assert len(allrows) >= len(anomalous)


def test_masks_returns_sensor_mask(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        df = q.masks(scenario="net3_sensor_3")
    assert "bias_mask_10" in set(df["mask"])


def test_flowrate_link_filter(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        # Pick any link present in the flowrate long table.
        link = q.sql(
            "SELECT name FROM flowrate_long WHERE scenario_basename='net3_normal_0' LIMIT 1"
        )["name"][0]
        df = q.flowrate(scenario="net3_normal_0", links=[link])
    assert set(df["name"]) == {link}


def test_leak_demand_nonzero_for_leak(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        df = q.leak_demand(scenario="net3_leak_a_1")
    assert df["value"].abs().max() > 1e-9


def test_long_table_agrees_with_wide_sampled(query_db: Path) -> None:
    with DatasetQuery(query_db) as q:
        long_val = q.sql(
            "SELECT value FROM pressure_long WHERE scenario_basename='net3_leak_a_1' "
            "AND name='10' AND time_seconds=3600"
        )["value"][0]
        wide_val = q.sql('SELECT "10" FROM net3_leak_a_1_pressure WHERE time_seconds=3600')[
            "10"
        ][0]
    assert long_val == pytest.approx(wide_val)


def test_cli_list_scenarios(query_db: Path) -> None:
    result = runner.invoke(app, ["query", str(query_db), "--list-scenarios"])
    assert result.exit_code == 0
    assert "net3_leak_a_1" in result.stdout


def test_cli_slice_to_csv(query_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "slice.csv"
    result = runner.invoke(
        app,
        [
            "query", str(query_db), "--quantity", "pressure",
            "--scenario", "net3_normal_0", "--nodes", "10,15",
            "--t-start", "0", "--t-end", "3600", "--out", str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.is_file()
    import pandas as pd

    df = pd.read_csv(out)
    # CSV round-trip can infer numeric-looking node names as ints.
    assert set(df["name"].astype(str)) == {"10", "15"}
    assert sorted(df["time_seconds"].unique()) == [0]


def test_cli_scenario_type_slice_to_parquet(query_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "leaks.parquet"
    result = runner.invoke(
        app,
        [
            "query", str(query_db), "--quantity", "pressure",
            "--scenario-type", "leak", "--nodes", "10", "--out", str(out),
        ],
    )
    assert result.exit_code == 0
    import pandas as pd

    df = pd.read_parquet(out)
    assert set(df["scenario_basename"]) == {"net3_leak_a_1", "net3_leak_b_2"}
