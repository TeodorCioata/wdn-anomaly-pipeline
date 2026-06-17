"""Batch driver for the pipeline.

A batch run accepts a list of YAML config paths and feeds each one
through :func:`wdn_pipeline.runner.run_from_config_file`. Failures are
captured per-config so a single broken config does not abort the batch.

Two summary artefacts are produced under ``<output_root>/<batch_id>/``:

- ``batch_summary.json``: machine-readable record of every run
  (config path, severity, elapsed time, output paths, error message).
- ``batch_summary.txt``: human-readable severity table grouped by
  scenario type.

Phase 5 Week 7 adds two capabilities:

- ``workers > 1`` runs scenarios in parallel via a
  :class:`concurrent.futures.ProcessPoolExecutor` using the ``spawn``
  start method so worker processes start with clean WNTR state.
- ``duckdb_path`` consolidates every scenario's tables into a single
  DuckDB file. The DuckDB writer is invoked in the main process so the
  shared file never sees concurrent writers; in parallel mode workers
  write only parquet/csv and the main process re-imports each
  scenario's tables on completion.
"""

from __future__ import annotations

import glob
import json
import logging
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from wdn_pipeline.config import load_config
from wdn_pipeline.network import derive_network_name
from wdn_pipeline.output import ALL_TABLE_NAMES, DuckDBWriter, build_basename
from wdn_pipeline.runner import RunSummary, run

logger = logging.getLogger("wdn_pipeline.batch")


@dataclass
class BatchRunResult:
    """Outcome of running one config inside a batch.

    Either ``summary`` is set (success path) or ``error`` is set
    (failure path); they never both carry data.
    """

    config_path: Path
    status: str  # "ok" | "warning" | "fail" | "error"
    elapsed_seconds: float
    summary: RunSummary | None = None
    error: str | None = None

    @property
    def scenario_type(self) -> str:
        """Recover the scenario family from the resolved fault counts.

        ``RunSummary`` does not store the source ``scenario.type`` value
        because the validator branches on the resolved fault families
        rather than the config label; we mirror that here so the batch
        report stays consistent with the runner.
        """

        if self.summary is None:
            return "unknown"
        has_leak = bool(self.summary.resolved_leaks)
        has_sensor = bool(self.summary.resolved_sensor_faults)
        if has_leak and has_sensor:
            return "cumulative"
        if has_leak:
            return "leak"
        if has_sensor:
            return "sensor_fault"
        return "normal"


@dataclass
class BatchSummary:
    """Aggregated outcome of one batch invocation."""

    batch_id: str
    started_at: float
    finished_at: float
    workers: int = 1
    duckdb_path: Path | None = None
    results: list[BatchRunResult] = field(default_factory=list)
    # Scenarios that ran successfully but whose tables could not be imported
    # into the consolidated DuckDB file (parallel path only). Each entry is a
    # human-readable "config: error" string. These runs still count as ok in
    # ``results`` but are absent from the DB, so they are surfaced explicitly.
    duckdb_import_errors: list[str] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return self.finished_at - self.started_at

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def n_warning(self) -> int:
        return sum(1 for r in self.results if r.status == "warning")

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")

    @property
    def n_error(self) -> int:
        return sum(1 for r in self.results if r.status == "error")


def resolve_config_paths(
    paths: list[str | Path] | None = None,
    config_dir: str | Path | None = None,
    glob_pattern: str = "*.yaml",
) -> list[Path]:
    """Expand globs and directory specifications into a sorted file list.

    Args:
        paths: A list of explicit paths or glob patterns. Each entry is
            expanded with :mod:`glob`; non-matching entries are kept as
            literal paths so a missing-file failure surfaces inside the
            batch (rather than silently being dropped here).
        config_dir: A directory to scan with ``glob_pattern``. May be
            combined with ``paths``.
        glob_pattern: Pattern used with ``config_dir``. Default
            ``*.yaml``.

    Returns:
        A list of unique :class:`Path` instances in sorted order.
    """

    resolved: list[Path] = []
    if paths:
        for entry in paths:
            entry_str = str(entry)
            matches = sorted(glob.glob(entry_str))
            if matches:
                resolved.extend(Path(m) for m in matches)
            else:
                resolved.append(Path(entry_str))
    if config_dir is not None:
        directory = Path(config_dir)
        for match in sorted(directory.glob(glob_pattern)):
            resolved.append(match)

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _classify_status(summary: RunSummary) -> str:
    severity = summary.validation.severity
    return severity if severity in {"ok", "warning", "fail"} else "ok"


def _detect_basename_collisions(config_paths: list[Path]) -> None:
    """Raise if two configs resolve to the same output basename.

    Two configs sharing a ``{network}_{label}_{seed}`` basename would
    silently overwrite each other's Parquet/CSV files and DuckDB rows. This
    pre-flight catches that before any scenario runs. Configs that fail to
    load are skipped here: their own run records the load error, preserving
    the batch's per-config failure isolation.
    """

    seen: dict[str, Path] = {}
    collisions: list[str] = []
    for path in config_paths:
        try:
            cfg = load_config(path)
        except Exception:  # noqa: BLE001 - a malformed config errors in its own run
            continue
        basename = build_basename(derive_network_name(cfg.network), cfg.scenario.label, cfg.seed)
        if basename in seen:
            collisions.append(f"{basename!r}: {seen[basename]} and {path}")
        else:
            seen[basename] = path
    if collisions:
        raise ValueError(
            "Duplicate output basenames in batch (they would overwrite each other's "
            "outputs and DuckDB rows):\n  " + "\n  ".join(collisions)
        )


def _run_one_config(
    config_path: Path,
    duckdb_path_override: Path | None,
    duckdb_wide_tables: bool = True,
) -> BatchRunResult:
    """Run one scenario; safe to call as a worker entry point.

    Defined at module scope so :class:`ProcessPoolExecutor` can pickle
    it under the ``spawn`` start method.

    Args:
        config_path: YAML config to load and run.
        duckdb_path_override: If non-None, the loaded config's
            ``output.duckdb`` is forced on and ``output.duckdb_path`` is
            replaced with this value before running. This lets a
            sequential batch funnel every scenario into the same DB
            without the per-config YAML mentioning DuckDB. Pass ``None``
            in parallel batches so workers never touch the shared DB.
    """

    run_started = time.perf_counter()
    try:
        cfg = load_config(config_path)
        if duckdb_path_override is not None:
            updated_output = cfg.output.model_copy(
                update={
                    "duckdb": True,
                    "duckdb_path": Path(duckdb_path_override),
                    "duckdb_wide_tables": duckdb_wide_tables,
                }
            )
            cfg = cfg.model_copy(update={"output": updated_output})
        summary = run(cfg, config_path=str(config_path))
        elapsed = time.perf_counter() - run_started
        return BatchRunResult(
            config_path=Path(config_path),
            status=_classify_status(summary),
            elapsed_seconds=elapsed,
            summary=summary,
        )
    except Exception as exc:  # noqa: BLE001 - batch isolates per-config failures
        elapsed = time.perf_counter() - run_started
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return BatchRunResult(
            config_path=Path(config_path),
            status="error",
            elapsed_seconds=elapsed,
            error=tb,
        )


def _import_run_to_duckdb(
    duckdb_path: Path,
    result: BatchRunResult,
    wide_tables: bool = True,
) -> None:
    """Read parquet outputs for a finished run and append to DuckDB.

    Used in parallel batches where workers must not touch the shared
    DuckDB file. The function reads back every per-table parquet file
    produced by the run, reconstructs the ``tables`` dict, and invokes
    :meth:`DuckDBWriter.write` from the main process.

    A run without parquet outputs (e.g. ``formats: [csv]``) raises
    :class:`RuntimeError`: writing the consolidated DB out of CSV would
    be needlessly slow and would lose dtype information.
    """

    assert result.summary is not None, "_import_run_to_duckdb expects a successful run"
    parquet_paths = [p for p in result.summary.output_paths if p.suffix == ".parquet"]
    if not parquet_paths:
        raise RuntimeError(
            f"Cannot import {result.config_path} into DuckDB: the run produced "
            "no parquet files. Add 'parquet' to output.formats."
        )

    tables: dict[str, pd.DataFrame] = {}
    basename = ""
    # Match longest table suffixes first so e.g. ``leak_demand`` wins
    # over ``demand`` when both are valid table names with a shared tail.
    sorted_table_names = sorted(ALL_TABLE_NAMES, key=len, reverse=True)
    for path in parquet_paths:
        stem = path.stem  # e.g. net3_cumulative_42_pressure
        matched_table: str | None = None
        for table_name in sorted_table_names:
            suffix = f"_{table_name}"
            if stem.endswith(suffix):
                matched_table = table_name
                basename = stem[: -len(suffix)]
                break
        if matched_table is None:
            continue
        df = pq.read_table(path).to_pandas()
        if "time_seconds" in df.columns:
            df = df.set_index("time_seconds")
        tables[matched_table] = df

    if not basename or not tables:
        raise RuntimeError(
            f"Could not infer scenario basename from parquet outputs of {result.config_path}"
        )

    metadata: dict = {}
    if result.summary.metadata_path is not None and result.summary.metadata_path.is_file():
        import yaml as _yaml

        metadata = _yaml.safe_load(result.summary.metadata_path.read_text()) or {}

    DuckDBWriter.write(
        db_path=duckdb_path,
        basename=basename,
        tables=tables,
        metadata=metadata,
        validation_severity=result.summary.validation.severity,
        config_path=str(result.config_path),
        wide_tables=wide_tables,
    )


def run_batch(
    config_paths: list[Path],
    report_dir: Path,
    batch_id: str | None = None,
    workers: int = 1,
    duckdb_path: Path | None = None,
    duckdb_wide_tables: bool = True,
) -> BatchSummary:
    """Run a sequence of configs and produce summary artefacts.

    Args:
        config_paths: Configs to run, in order. Each is processed
            independently; a failure does not stop the batch.
        report_dir: Directory to write ``batch_summary.{json,txt}`` into.
            Created on demand.
        batch_id: Identifier to embed in the summary. Defaults to a
            timestamp ``YYYYmmddTHHMMSS``.
        workers: Number of worker processes. ``1`` (the default) runs
            inline in the calling process. Any value above ``1`` uses
            :class:`concurrent.futures.ProcessPoolExecutor` with the
            ``spawn`` start method.
        duckdb_path: Optional consolidated DuckDB file. Every scenario's
            tables are inserted under the ``{basename}_{table}``
            namespace; a ``scenarios`` metadata table records one row
            per scenario. In sequential mode the DuckDB
            write happens inside the runner; in parallel mode the
            workers skip DuckDB and the main process imports each
            scenario's parquet outputs on completion.

    Returns:
        The :class:`BatchSummary` for the batch.
    """

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    if duckdb_path is not None:
        duckdb_path = Path(duckdb_path)
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    if batch_id is None:
        batch_id = time.strftime("%Y%m%dT%H%M%S")

    # Pre-flight: reject configs that would write to the same basename and
    # silently clobber each other's outputs before any simulation runs.
    _detect_basename_collisions(config_paths)

    started = time.perf_counter()
    results: list[BatchRunResult] = []
    duckdb_import_errors: list[str] = []

    if workers <= 1:
        for i, config_path in enumerate(config_paths, start=1):
            logger.info("[%d/%d] running %s", i, len(config_paths), config_path)
            result = _run_one_config(Path(config_path), duckdb_path, duckdb_wide_tables)
            if result.status == "error":
                logger.error(
                    "[%d/%d] %s failed: %s",
                    i,
                    len(config_paths),
                    config_path,
                    result.error,
                )
            results.append(result)
    else:
        spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn_ctx) as pool:
            future_to_meta: dict = {}
            for i, config_path in enumerate(config_paths, start=1):
                future = pool.submit(_run_one_config, Path(config_path), None)
                future_to_meta[future] = (i, Path(config_path))
            for future in as_completed(future_to_meta):
                i, config_path = future_to_meta[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - capture pool-level failures
                    tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                    result = BatchRunResult(
                        config_path=config_path,
                        status="error",
                        elapsed_seconds=0.0,
                        error=tb,
                    )
                logger.info(
                    "[%d/%d] %s -> %s (%.2fs)",
                    i,
                    len(config_paths),
                    config_path,
                    result.status,
                    result.elapsed_seconds,
                )
                if (
                    duckdb_path is not None
                    and result.status != "error"
                    and result.summary is not None
                ):
                    try:
                        _import_run_to_duckdb(duckdb_path, result, duckdb_wide_tables)
                    except Exception as exc:  # noqa: BLE001
                        msg = f"{config_path}: {exc}"
                        duckdb_import_errors.append(msg)
                        logger.warning("DuckDB import failed for %s: %s", config_path, exc)
                results.append(result)

    finished = time.perf_counter()
    summary = BatchSummary(
        batch_id=batch_id,
        started_at=started,
        finished_at=finished,
        workers=workers,
        duckdb_path=duckdb_path,
        results=results,
        duckdb_import_errors=duckdb_import_errors,
    )

    json_path = report_dir / "batch_summary.json"
    text_path = report_dir / "batch_summary.txt"
    json_path.write_text(_format_batch_json(summary), encoding="utf-8")
    text_path.write_text(_format_batch_text(summary), encoding="utf-8")
    logger.info("Wrote batch summary to %s and %s", json_path, text_path)

    return summary


def _format_batch_json(summary: BatchSummary) -> str:
    payload: dict[str, object] = {
        "batch_id": summary.batch_id,
        "elapsed_seconds": summary.elapsed_seconds,
        "workers": summary.workers,
        "duckdb_path": str(summary.duckdb_path) if summary.duckdb_path else None,
        "duckdb_import_errors": list(summary.duckdb_import_errors),
        "totals": {
            "total": summary.n_total,
            "ok": summary.n_ok,
            "warning": summary.n_warning,
            "fail": summary.n_fail,
            "error": summary.n_error,
            "duckdb_import_failures": len(summary.duckdb_import_errors),
        },
        "runs": [],
    }
    runs = []
    for r in summary.results:
        entry: dict[str, object] = {
            "config_path": str(r.config_path),
            "status": r.status,
            "elapsed_seconds": r.elapsed_seconds,
            "scenario_type": r.scenario_type,
        }
        if r.summary is not None:
            entry["network"] = r.summary.network_name
            entry["scenario_label"] = r.summary.scenario_label
            entry["seed"] = r.summary.seed
            entry["num_timesteps"] = r.summary.num_timesteps
            entry["output_paths"] = [str(p) for p in r.summary.output_paths]
            entry["metadata_path"] = (
                str(r.summary.metadata_path) if r.summary.metadata_path is not None else None
            )
            entry["validation"] = {
                "severity": r.summary.validation.severity,
                "checks": [
                    {
                        "name": c.name,
                        "severity": c.severity,
                        "detail": c.detail,
                    }
                    for c in r.summary.validation.checks
                ],
            }
            entry["resolved_leaks"] = len(r.summary.resolved_leaks)
            entry["resolved_sensor_faults"] = len(r.summary.resolved_sensor_faults)
            entry["interactions"] = list(r.summary.interactions)
        if r.error is not None:
            entry["error"] = r.error
        runs.append(entry)
    payload["runs"] = runs
    return json.dumps(payload, indent=2, default=str) + "\n"


def _format_batch_text(summary: BatchSummary) -> str:
    lines: list[str] = []
    lines.append(f"Batch summary ({summary.batch_id})")
    extras = [
        f"total={summary.n_total}",
        f"ok={summary.n_ok}",
        f"warning={summary.n_warning}",
        f"fail={summary.n_fail}",
        f"error={summary.n_error}",
        f"workers={summary.workers}",
        f"elapsed={summary.elapsed_seconds:.2f}s",
    ]
    lines.append("  " + " ".join(extras))
    if summary.duckdb_path is not None:
        suffix = (
            f" ({len(summary.duckdb_import_errors)} import failure(s))"
            if summary.duckdb_import_errors
            else ""
        )
        lines.append(f"  duckdb={summary.duckdb_path}{suffix}")
    lines.append("")
    groups: dict[str, list[BatchRunResult]] = {
        "normal": [],
        "leak": [],
        "sensor_fault": [],
        "cumulative": [],
        "unknown": [],
    }
    error_rows: list[BatchRunResult] = []
    for r in summary.results:
        if r.status == "error":
            error_rows.append(r)
        else:
            groups.setdefault(r.scenario_type, []).append(r)
    for group_name, rows in groups.items():
        if not rows:
            continue
        lines.append(f"[{group_name}]")
        for r in rows:
            lines.append(f"  {r.status:>7}  {r.elapsed_seconds:>6.2f}s  {r.config_path}")
        lines.append("")
    if error_rows:
        lines.append("[errors]")
        for r in error_rows:
            lines.append(f"  {r.status:>7}  {r.elapsed_seconds:>6.2f}s  {r.config_path}")
            if r.error:
                for chunk in r.error.splitlines():
                    lines.append(f"      {chunk}")
        lines.append("")
    if summary.duckdb_import_errors:
        # These scenarios ran ok but never made it into the consolidated DB.
        lines.append("[duckdb import failures]")
        for msg in summary.duckdb_import_errors:
            lines.append(f"  {msg}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
