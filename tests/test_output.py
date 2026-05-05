"""Tests for the output writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from wdn_pipeline.labelling import Labels
from wdn_pipeline.output import (
    CsvWriter,
    ParquetWriter,
    assemble_tables,
    build_basename,
    write_outputs,
)
from wdn_pipeline.simulation import SimulationResults


def _fake_results() -> SimulationResults:
    idx = pd.Index([0, 3600, 7200], name="time_seconds")
    pressure = pd.DataFrame({"n1": [10.0, 11.0, 12.0], "n2": [20.0, 21.0, 22.0]}, index=idx)
    flow = pd.DataFrame({"l1": [0.1, 0.2, 0.3]}, index=idx)
    demand = pd.DataFrame({"n1": [0.01, 0.01, 0.01], "n2": [0.0, 0.0, 0.0]}, index=idx)
    leak = pd.DataFrame({"n1": [0.0, 0.0, 0.0], "n2": [0.0, 0.0, 0.0]}, index=idx)
    return SimulationResults(pressure, flow, demand, leak, elapsed_seconds=0.5)


def _fake_labels(n: int) -> Labels:
    idx = pd.Index([0, 3600, 7200])
    return Labels(
        timestep_labels=pd.Series([0] * n, index=idx, dtype="int8", name="label"),
        metadata={"scenario_type": "normal", "scenario_label": "normal", "seed": 1},
    )


def test_build_basename() -> None:
    assert build_basename("net3", "normal", 42) == "net3_normal_42"
    assert build_basename("My Net", "leak abrupt", 7) == "My_Net_leak_abrupt_7"


def test_assemble_tables_appends_label_column() -> None:
    tables = assemble_tables(_fake_results(), _fake_labels(3))
    for name, df in tables.items():
        assert "label" in df.columns, name
        assert (df["label"] == 0).all()


def test_parquet_writer_round_trip(tmp_path: Path) -> None:
    tables = assemble_tables(_fake_results(), _fake_labels(3))
    metadata = {"scenario_type": "normal", "seed": 1}
    paths = ParquetWriter().write(tmp_path, "test", tables, metadata)
    assert len(paths) == 4
    for p in paths:
        assert p.exists()
        table = pq.read_table(p)
        # The metadata sidecar string is stored in pyarrow file metadata.
        kv = table.schema.metadata or {}
        assert b"wdn_pipeline" in kv
        embedded = yaml.safe_load(kv[b"wdn_pipeline"].decode("utf-8"))
        assert embedded["scenario_type"] == "normal"


def test_csv_writer_round_trip(tmp_path: Path) -> None:
    tables = assemble_tables(_fake_results(), _fake_labels(3))
    paths = CsvWriter().write(tmp_path, "test", tables, {})
    assert len(paths) == 4
    df = pd.read_csv(paths[0])
    assert "time_seconds" in df.columns
    assert "label" in df.columns


def test_write_outputs_both_formats_and_sidecar(tmp_path: Path) -> None:
    tables = assemble_tables(_fake_results(), _fake_labels(3))
    result = write_outputs(
        directory=tmp_path,
        basename="x",
        formats=["parquet", "csv"],
        tables=tables,
        metadata={"scenario_label": "normal"},
        write_metadata_sidecar_flag=True,
    )
    # Four tables x two formats = 8 data files.
    assert len(result.data_paths) == 8
    assert result.metadata_path is not None
    assert result.metadata_path.is_file()
    sidecar = yaml.safe_load(result.metadata_path.read_text())
    assert sidecar["scenario_label"] == "normal"


def test_unknown_format_raises(tmp_path: Path) -> None:
    tables = assemble_tables(_fake_results(), _fake_labels(3))
    with pytest.raises(ValueError):
        write_outputs(
            directory=tmp_path,
            basename="x",
            formats=["pickle"],  # unsupported
            tables=tables,
            metadata={},
            write_metadata_sidecar_flag=False,
        )
