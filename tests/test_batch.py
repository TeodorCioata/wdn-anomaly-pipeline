"""Tests for the sequential batch driver."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from wdn_pipeline.batch import (
    BatchSummary,
    resolve_config_paths,
    run_batch,
)


def _normal_net3_config(
    tmp_path: Path,
    label: str,
    seed: int = 42,
    duration_seconds: int = 24 * 3600,
) -> Path:
    """Write a minimal Net3 normal-scenario YAML config."""

    out_dir = tmp_path / "outputs"
    cfg = {
        "network": {"inp_path": "Net3", "name": "net3"},
        "simulation": {
            "duration_seconds": duration_seconds,
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
    path = tmp_path / f"{label}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_resolve_config_paths_handles_explicit_list_and_glob(tmp_path: Path) -> None:
    a = _normal_net3_config(tmp_path, "alpha")
    b = _normal_net3_config(tmp_path, "beta")
    explicit = resolve_config_paths(paths=[str(a)])
    assert explicit == [a]

    glob_paths = resolve_config_paths(paths=[str(tmp_path / "*.yaml")])
    assert set(glob_paths) == {a, b}


def test_resolve_config_paths_directory_scan(tmp_path: Path) -> None:
    _normal_net3_config(tmp_path, "alpha")
    _normal_net3_config(tmp_path, "beta")
    paths = resolve_config_paths(config_dir=tmp_path)
    assert len(paths) == 2
    assert all(p.suffix == ".yaml" for p in paths)


def test_resolve_config_paths_deduplicates(tmp_path: Path) -> None:
    a = _normal_net3_config(tmp_path, "alpha")
    paths = resolve_config_paths(paths=[str(a), str(a)], config_dir=tmp_path)
    assert paths == [a]


def test_batch_rejects_duplicate_basenames(tmp_path: Path) -> None:
    """Two configs that resolve to the same output basename are rejected.

    They would otherwise overwrite each other's outputs and DuckDB rows.
    The pre-flight runs before any simulation.
    """

    cfg = {
        "network": {"inp_path": "Net3", "name": "net3"},
        "simulation": {
            "duration_seconds": 3600,
            "hydraulic_timestep_seconds": 3600,
            "report_timestep_seconds": 3600,
        },
        "seed": 7,
        "scenario": {"type": "normal", "label": "dup"},
        "output": {
            "directory": str(tmp_path / "outputs"),
            "formats": ["parquet"],
            "write_metadata_sidecar": False,
        },
    }
    p1 = tmp_path / "a.yaml"
    p2 = tmp_path / "b.yaml"
    p1.write_text(yaml.safe_dump(cfg))
    p2.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Duplicate output basenames"):
        run_batch([p1, p2], tmp_path / "report")


def test_run_batch_writes_summary_artefacts(tmp_path: Path) -> None:
    configs = [_normal_net3_config(tmp_path, f"normal_{i}", seed=10 + i) for i in range(3)]
    report_dir = tmp_path / "report"
    summary = run_batch(configs, report_dir, batch_id="test_batch")

    assert isinstance(summary, BatchSummary)
    assert summary.n_total == 3
    assert summary.n_error == 0
    assert summary.batch_id == "test_batch"

    json_path = report_dir / "batch_summary.json"
    text_path = report_dir / "batch_summary.txt"
    assert json_path.is_file()
    assert text_path.is_file()

    payload = json.loads(json_path.read_text())
    assert payload["batch_id"] == "test_batch"
    assert payload["totals"]["total"] == 3
    assert len(payload["runs"]) == 3
    for run_entry in payload["runs"]:
        assert run_entry["status"] in {"ok", "warning"}
        assert run_entry["network"] == "net3"
        assert run_entry["validation"]["severity"] in {"ok", "warning"}

    text = text_path.read_text()
    assert "Batch summary (test_batch)" in text
    assert "[normal]" in text


def test_run_batch_continues_on_broken_config(tmp_path: Path) -> None:
    good = _normal_net3_config(tmp_path, "good")
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        "network:\n  inp_path: /nonexistent/path/that/does/not.inp\n"
        "simulation:\n  duration_seconds: 3600\n"
    )
    report_dir = tmp_path / "report"
    summary = run_batch([good, broken], report_dir, batch_id="error_batch")
    assert summary.n_total == 2
    assert summary.n_error == 1
    assert summary.n_ok + summary.n_warning == 1

    payload = json.loads((report_dir / "batch_summary.json").read_text())
    statuses = [r["status"] for r in payload["runs"]]
    assert "error" in statuses
    error_entry = next(r for r in payload["runs"] if r["status"] == "error")
    assert "error" in error_entry


def test_run_batch_classifies_scenario_types(tmp_path: Path) -> None:
    # Build one normal config; the batch summary's text output should
    # have the normal section populated.
    cfg = _normal_net3_config(tmp_path, "normal")
    report_dir = tmp_path / "report"
    summary = run_batch([cfg], report_dir, batch_id="classify")
    assert summary.results[0].scenario_type == "normal"
    text = (report_dir / "batch_summary.txt").read_text()
    assert "[normal]" in text


def test_batch_two_runs_produce_identical_outputs(tmp_path: Path) -> None:
    """Reproducibility spot-check: a config run twice produces equal data."""

    cfg = _normal_net3_config(tmp_path, "repro", seed=11)
    report_a = tmp_path / "report_a"
    report_b = tmp_path / "report_b"
    summary_a = run_batch([cfg], report_a, batch_id="run_a")
    summary_b = run_batch([cfg], report_b, batch_id="run_b")

    assert summary_a.results[0].summary is not None
    assert summary_b.results[0].summary is not None
    path_a = next(
        p for p in summary_a.results[0].summary.output_paths if p.name.endswith("pressure.parquet")
    )
    path_b = next(
        p for p in summary_b.results[0].summary.output_paths if p.name.endswith("pressure.parquet")
    )
    df_a = pq.read_table(path_a).to_pandas()
    df_b = pq.read_table(path_b).to_pandas()
    # Pressure should be bit-identical for the same seed; the WNTR
    # Newton residual is well under the 1e-12 m threshold documented in
    # CLAUDE.md.
    diff = (df_a.drop(columns=["time_seconds"]) - df_b.drop(columns=["time_seconds"])).abs()
    assert diff.max().max() < 1e-12


@pytest.mark.parametrize("missing", [True, False])
def test_resolve_config_paths_passes_through_missing(tmp_path: Path, missing: bool) -> None:
    """A non-matching glob entry is kept as a literal path so the batch
    sees the failure and reports it; it is not silently dropped."""

    real = _normal_net3_config(tmp_path, "real")
    entries = [str(real)]
    if missing:
        entries.append(str(tmp_path / "does_not_exist.yaml"))
    resolved = resolve_config_paths(paths=entries)
    expected_count = 2 if missing else 1
    assert len(resolved) == expected_count


# ---------------------------------------------------------------------------
# Phase 5 Week 7: parallel batch tests.
# ---------------------------------------------------------------------------


def _read_pressure_parquet(path: Path):
    return pq.read_table(path).to_pandas().sort_values("time_seconds").reset_index(drop=True)


def test_parallel_batch_matches_sequential_outputs(tmp_path: Path) -> None:
    """workers=2 must produce numerically identical pressure to workers=1."""

    configs = [_normal_net3_config(tmp_path, f"par_{i}", seed=40 + i) for i in range(3)]
    seq_summary = run_batch(configs, tmp_path / "seq", batch_id="seq", workers=1)
    par_summary = run_batch(configs, tmp_path / "par", batch_id="par", workers=2)
    assert seq_summary.n_error == 0
    assert par_summary.n_error == 0
    assert seq_summary.n_total == par_summary.n_total == 3
    assert par_summary.workers == 2

    # The runs may complete in any order under parallel execution; pair
    # by config_path before comparing parquet outputs.
    seq_by_cfg = {r.config_path: r for r in seq_summary.results}
    par_by_cfg = {r.config_path: r for r in par_summary.results}
    for cfg_path in seq_by_cfg:
        seq_r = seq_by_cfg[cfg_path]
        par_r = par_by_cfg[cfg_path]
        assert seq_r.summary is not None and par_r.summary is not None
        seq_p = next(p for p in seq_r.summary.output_paths if p.name.endswith("pressure.parquet"))
        par_p = next(p for p in par_r.summary.output_paths if p.name.endswith("pressure.parquet"))
        seq_df = _read_pressure_parquet(seq_p)
        par_df = _read_pressure_parquet(par_p)
        diff = (seq_df.drop(columns=["time_seconds"]) - par_df.drop(columns=["time_seconds"])).abs()
        assert diff.max().max() < 1e-12


def test_parallel_batch_isolates_broken_config(tmp_path: Path) -> None:
    """A broken config inside a parallel pool must not crash the batch."""

    good = _normal_net3_config(tmp_path, "good", seed=51)
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        "network:\n  inp_path: /nonexistent/path/that/does/not.inp\n"
        "simulation:\n  duration_seconds: 3600\n"
    )
    summary = run_batch(
        [good, broken],
        tmp_path / "report",
        batch_id="par_broken",
        workers=2,
    )
    assert summary.n_total == 2
    assert summary.n_error == 1
    assert summary.n_ok + summary.n_warning == 1


def test_batch_summary_records_workers_and_duckdb(tmp_path: Path) -> None:
    cfg = _normal_net3_config(tmp_path, "wflag", seed=61)
    summary = run_batch([cfg], tmp_path / "report", batch_id="meta", workers=1, duckdb_path=None)
    assert summary.workers == 1
    assert summary.duckdb_path is None
    text = (tmp_path / "report" / "batch_summary.txt").read_text()
    assert "workers=1" in text


def _quality_net3_config(tmp_path: Path, label: str, seed: int = 1) -> Path:
    """Write a water-quality (age) Net3 config for the parallel-import test."""

    cfg = {
        "network": {"inp_path": "Net3", "name": "net3"},
        "simulation": {
            "duration_seconds": 24 * 3600,
            "hydraulic_timestep_seconds": 3600,
            "report_timestep_seconds": 3600,
            "demand_model": "DDA",
            "quality": {"parameter": "age"},
        },
        "seed": seed,
        "scenario": {"type": "normal", "label": label},
        "output": {
            "directory": str(tmp_path / "outputs"),
            "formats": ["parquet"],
            "write_metadata_sidecar": True,
        },
    }
    path = tmp_path / f"{label}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_parallel_duckdb_import_includes_quality_table(tmp_path: Path) -> None:
    """A parallel --duckdb batch must consolidate the optional quality table.

    Workers write parquet and the main process re-imports it; the import
    must recognise the quality table suffix (it is not in the base table
    set) or water-quality scenarios would silently lose their quality
    data in the consolidated DuckDB file.
    """

    from wdn_pipeline.query import DatasetQuery

    cfg_quality = _quality_net3_config(tmp_path, "quality_age", seed=1)
    cfg_normal = _normal_net3_config(tmp_path, "plain", seed=2)
    db_path = tmp_path / "batch.duckdb"
    summary = run_batch(
        [cfg_quality, cfg_normal],
        tmp_path / "report",
        batch_id="qpar",
        workers=2,
        duckdb_path=db_path,
    )
    assert summary.n_error == 0

    with DatasetQuery(db_path) as q:
        quality = q.quality(scenario="net3_quality_age_1")
        assert not quality.empty
        # The plain run has no quality rows.
        assert q.quality(scenario="net3_plain_2").empty
