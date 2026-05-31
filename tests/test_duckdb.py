"""Tests for the DuckDB writer (decision D30, Phase 5 Week 7)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from wdn_pipeline.batch import resolve_config_paths, run_batch
from wdn_pipeline.config import load_config
from wdn_pipeline.output import DuckDBWriter
from wdn_pipeline.runner import run


def _normal_net3_config(
    tmp_path: Path,
    label: str,
    seed: int = 42,
    duckdb_path: Path | None = None,
) -> Path:
    """Write a Net3 normal config, optionally with DuckDB output enabled."""

    out_dir = tmp_path / "outputs"
    cfg: dict = {
        "network": {"inp_path": "Net3", "name": "net3"},
        "simulation": {
            "duration_seconds": 24 * 3600,
            "hydraulic_timestep_seconds": 3600,
            "report_timestep_seconds": 3600,
        },
        "seed": seed,
        "scenario": {"type": "normal", "label": label},
        "output": {
            "directory": str(out_dir),
            "formats": ["parquet"],
            "write_metadata_sidecar": False,
        },
    }
    if duckdb_path is not None:
        cfg["output"]["duckdb"] = True
        cfg["output"]["duckdb_path"] = str(duckdb_path)
    path = tmp_path / f"{label}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_duckdb_writer_requires_path_when_enabled(tmp_path: Path) -> None:
    cfg_yaml = {
        "network": {"inp_path": "Net3", "name": "net3"},
        "simulation": {
            "duration_seconds": 24 * 3600,
            "hydraulic_timestep_seconds": 3600,
            "report_timestep_seconds": 3600,
        },
        "seed": 1,
        "scenario": {"type": "normal", "label": "n"},
        "output": {
            "directory": str(tmp_path / "out"),
            "formats": ["parquet"],
            "duckdb": True,
        },
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg_yaml))
    with pytest.raises(ValidationError):
        load_config(path)


def test_single_scenario_writes_duckdb(tmp_path: Path) -> None:
    db_path = tmp_path / "single.duckdb"
    cfg_path = _normal_net3_config(tmp_path, "n", seed=11, duckdb_path=db_path)
    cfg = load_config(cfg_path)
    summary = run(cfg, config_path=str(cfg_path))
    assert summary.validation.severity in {"ok", "warning"}
    assert db_path.is_file()
    assert db_path in summary.output_paths

    con = duckdb.connect(str(db_path))
    tables = sorted(r[0] for r in con.execute("SHOW TABLES").fetchall())
    expected = [
        "net3_n_11_demand",
        "net3_n_11_flowrate",
        "net3_n_11_flowrate_clean",
        "net3_n_11_leak_demand",
        "net3_n_11_pressure",
        "net3_n_11_pressure_clean",
        "scenarios",
    ]
    assert tables == expected
    count = con.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0]
    assert count == 1


def test_scenarios_metadata_table_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.duckdb"
    cfg_path = _normal_net3_config(tmp_path, "n", seed=7, duckdb_path=db_path)
    cfg = load_config(cfg_path)
    run(cfg, config_path=str(cfg_path))

    con = duckdb.connect(str(db_path))
    cols = [r[0] for r in con.execute("DESCRIBE scenarios").fetchall()]
    required = {
        "scenario_basename",
        "config_path",
        "scenario_type",
        "scenario_label",
        "network_name",
        "seed",
        "demand_model",
        "duration_seconds",
        "num_leaks",
        "num_sensor_faults",
        "num_interactions",
        "validation_severity",
        "fault_summary_json",
    }
    assert required.issubset(set(cols))
    row = con.execute(
        "SELECT scenario_basename, network_name, seed, num_leaks, "
        "num_sensor_faults, validation_severity FROM scenarios"
    ).fetchone()
    assert row[0] == "net3_n_7"
    assert row[1] == "net3"
    assert row[2] == 7
    assert row[3] == 0
    assert row[4] == 0
    assert row[5] in {"ok", "warning"}


def test_duckdb_table_roundtrip_matches_parquet(tmp_path: Path) -> None:
    """Reading a table from DuckDB must equal reading the parquet sibling."""

    db_path = tmp_path / "roundtrip.duckdb"
    cfg_path = _normal_net3_config(tmp_path, "rt", seed=3, duckdb_path=db_path)
    cfg = load_config(cfg_path)
    summary = run(cfg, config_path=str(cfg_path))

    pressure_parquet = next(
        p for p in summary.output_paths if p.name.endswith("pressure.parquet")
    )
    df_parquet = pq.read_table(pressure_parquet).to_pandas()

    con = duckdb.connect(str(db_path))
    df_duck = con.execute(
        'SELECT * FROM "net3_rt_3_pressure" ORDER BY time_seconds'
    ).fetchdf()

    assert set(df_parquet.columns) == set(df_duck.columns)
    df_parquet = df_parquet.sort_values("time_seconds").reset_index(drop=True)
    df_duck = df_duck.sort_values("time_seconds").reset_index(drop=True)
    # Pressure columns are floats; compare numerically.
    numeric_cols = [
        c for c in df_parquet.columns if df_parquet[c].dtype.kind in {"f", "i"}
    ]
    for col in numeric_cols:
        assert np.allclose(
            df_parquet[col].to_numpy(),
            df_duck[col].to_numpy(),
            atol=1e-12,
            equal_nan=True,
        ), f"mismatch in column {col}"


def test_batch_consolidates_three_scenarios_into_one_duckdb(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.duckdb"
    configs = [
        _normal_net3_config(tmp_path, f"b{i}", seed=20 + i) for i in range(3)
    ]
    summary = run_batch(
        configs,
        tmp_path / "report",
        batch_id="three",
        duckdb_path=db_path,
    )
    assert summary.n_error == 0
    assert summary.duckdb_path == db_path
    assert db_path.is_file()

    con = duckdb.connect(str(db_path))
    n_scenarios = con.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0]
    assert n_scenarios == 3
    basenames = sorted(
        r[0] for r in con.execute("SELECT scenario_basename FROM scenarios").fetchall()
    )
    assert basenames == ["net3_b0_20", "net3_b1_21", "net3_b2_22"]
    # One table per scenario per table_name should be present.
    table_names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    for base in basenames:
        for tbl in (
            "pressure",
            "flowrate",
            "demand",
            "leak_demand",
            "pressure_clean",
            "flowrate_clean",
        ):
            assert f"{base}_{tbl}" in table_names


def test_batch_duckdb_parallel_produces_same_tables_as_sequential(
    tmp_path: Path,
) -> None:
    configs = [
        _normal_net3_config(tmp_path, f"p{i}", seed=30 + i) for i in range(3)
    ]
    seq_db = tmp_path / "seq.duckdb"
    par_db = tmp_path / "par.duckdb"
    run_batch(configs, tmp_path / "seq_report", batch_id="seq", duckdb_path=seq_db)
    run_batch(
        configs,
        tmp_path / "par_report",
        batch_id="par",
        workers=2,
        duckdb_path=par_db,
    )

    con_seq = duckdb.connect(str(seq_db))
    con_par = duckdb.connect(str(par_db))
    seq_tables = sorted(r[0] for r in con_seq.execute("SHOW TABLES").fetchall())
    par_tables = sorted(r[0] for r in con_par.execute("SHOW TABLES").fetchall())
    assert seq_tables == par_tables


def test_duckdb_writer_idempotent_on_rerun(tmp_path: Path) -> None:
    """Running the same scenario twice keeps one row in 'scenarios'."""

    db_path = tmp_path / "idem.duckdb"
    cfg_path = _normal_net3_config(tmp_path, "x", seed=99, duckdb_path=db_path)
    cfg = load_config(cfg_path)
    run(cfg, config_path=str(cfg_path))
    run(cfg, config_path=str(cfg_path))

    con = duckdb.connect(str(db_path))
    count = con.execute(
        "SELECT COUNT(*) FROM scenarios WHERE scenario_basename = 'net3_x_99'"
    ).fetchone()[0]
    assert count == 1


def test_duckdb_writer_direct_api(tmp_path: Path) -> None:
    """Smoke test for :meth:`DuckDBWriter.write` outside the runner."""

    import pandas as pd

    db_path = tmp_path / "direct.duckdb"
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df.index.name = "time_seconds"
    DuckDBWriter.write(
        db_path=db_path,
        basename="manual",
        tables={"pressure": df},
        metadata={
            "scenario_type": "normal",
            "network_name": "x",
            "seed": 1,
            "fault_summary": {"leaks": [], "sensor_faults": [], "interactions": []},
        },
        validation_severity="ok",
        config_path=None,
    )
    con = duckdb.connect(str(db_path))
    assert {"manual_pressure", "scenarios"} == {
        r[0] for r in con.execute("SHOW TABLES").fetchall()
    }
    row_count = con.execute("SELECT COUNT(*) FROM manual_pressure").fetchone()[0]
    assert row_count == 3


def test_batch_cli_paths_resolution_works_with_glob(tmp_path: Path) -> None:
    """Coverage for the typical batch entry point with --duckdb."""

    _normal_net3_config(tmp_path, "g1", seed=1)
    _normal_net3_config(tmp_path, "g2", seed=2)
    paths = resolve_config_paths(paths=[str(tmp_path / "*.yaml")])
    assert len(paths) == 2
