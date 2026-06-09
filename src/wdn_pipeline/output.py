"""Serialise simulation results to disk.

The writer set is extensible (decision D6). Phase 5 Week 7 adds a
:class:`DuckDBWriter` alongside the existing Parquet and CSV backends
(decision D30: consolidated, materialised tables). The DuckDB writer is
opt-in via :attr:`OutputConfig.duckdb` and writes each scenario's tables
into a single shared file using a ``{basename}_{table}`` namespace, so a
batch run produces one queryable database.

Output layout for one scenario named ``net3_normal_42``::

    outputs/
      net3_normal_42_pressure.parquet
      net3_normal_42_pressure.csv
      net3_normal_42_flowrate.parquet
      ...
      net3_normal_42.meta.yaml      <- written by Runner

A single ``label`` column is appended to every long-form table so
downstream consumers can filter without joining across files.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from wdn_pipeline.labelling import Labels
from wdn_pipeline.simulation import SimulationResults

# Names of the data tables we serialise per scenario. ``leak_demand`` is
# included unconditionally; for no-leak runs it is a zero frame matching
# the pressure shape (see :class:`SimulationResults`). ``pressure_clean``
# and ``flowrate_clean`` always carry the uncorrupted simulator output
# (Phase 4): they equal the corrupted siblings element-for-element when
# no sensor faults run, but the downstream ML consumer reads them as
# the ground truth in cumulative scenarios.
TABLE_NAMES = (
    "pressure",
    "flowrate",
    "demand",
    "leak_demand",
    "pressure_clean",
    "flowrate_clean",
)


@dataclass(frozen=True)
class WriteResult:
    """Paths produced by writing one scenario."""

    data_paths: list[Path]
    metadata_path: Path | None


class Writer(ABC):
    """Abstract serialisation backend.

    Each backend writes the three tables (pressure, flowrate, demand)
    as a single file or set of files keyed by ``basename``.
    """

    extension: str

    @abstractmethod
    def write(
        self,
        directory: Path,
        basename: str,
        tables: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> list[Path]: ...


class ParquetWriter(Writer):
    """Write each table as a separate Parquet file.

    Scenario metadata is embedded in each file's pyarrow key/value
    metadata (under the ``wdn_pipeline`` key) for self-describing
    artefacts. The sidecar YAML produced by the runner is the canonical
    copy.
    """

    extension = "parquet"

    def write(
        self,
        directory: Path,
        basename: str,
        tables: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        meta_yaml = yaml.safe_dump(metadata, sort_keys=True).encode("utf-8")
        for table_name, df in tables.items():
            target = directory / f"{basename}_{table_name}.parquet"
            arrow_table = pa.Table.from_pandas(df.reset_index().rename(columns={"index": "time_seconds"}))
            arrow_table = arrow_table.replace_schema_metadata(
                {b"wdn_pipeline": meta_yaml, b"table": table_name.encode("utf-8")}
            )
            pq.write_table(arrow_table, target)
            paths.append(target)
        return paths


class CsvWriter(Writer):
    """Write each table as a separate CSV file.

    The first column is ``time_seconds``. Metadata is **not** embedded
    in the CSV; rely on the sidecar YAML.
    """

    extension = "csv"

    def write(
        self,
        directory: Path,
        basename: str,
        tables: dict[str, pd.DataFrame],
        metadata: dict,
    ) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for table_name, df in tables.items():
            target = directory / f"{basename}_{table_name}.csv"
            df.to_csv(target, index_label="time_seconds")
            paths.append(target)
        return paths


WRITERS: dict[str, type[Writer]] = {
    "parquet": ParquetWriter,
    "csv": CsvWriter,
}


SCENARIOS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS scenarios (
    scenario_basename VARCHAR PRIMARY KEY,
    config_path VARCHAR,
    scenario_type VARCHAR,
    scenario_label VARCHAR,
    network_name VARCHAR,
    seed INTEGER,
    demand_model VARCHAR,
    duration_seconds INTEGER,
    num_leaks INTEGER,
    num_sensor_faults INTEGER,
    num_interactions INTEGER,
    validation_severity VARCHAR,
    fault_summary_json VARCHAR,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""".strip()


def _ensure_scenarios_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create the ``scenarios`` metadata table on first use."""

    con.execute(SCENARIOS_TABLE_DDL)


# Wide table name -> long table name for the value (node/link) tables. The
# long tables (decision D32, Week 8) are additive: they accumulate every
# scenario's measurements in (scenario_basename, time_seconds, name,
# value) form so the query layer can slice across scenarios efficiently
# without UNION-ing per-scenario wide tables. The clean-signal siblings
# are intentionally not melted (wide-only) to keep the long tables to the
# primary signals.
LONG_VALUE_TABLES = {
    "pressure": "pressure_long",
    "flowrate": "flowrate_long",
    "demand": "demand_long",
    "leak_demand": "leak_demand_long",
}

LONG_TABLE_DDLS = {
    "pressure_long": (
        "CREATE TABLE IF NOT EXISTS pressure_long "
        "(scenario_basename VARCHAR, time_seconds BIGINT, name VARCHAR, value DOUBLE)"
    ),
    "flowrate_long": (
        "CREATE TABLE IF NOT EXISTS flowrate_long "
        "(scenario_basename VARCHAR, time_seconds BIGINT, name VARCHAR, value DOUBLE)"
    ),
    "demand_long": (
        "CREATE TABLE IF NOT EXISTS demand_long "
        "(scenario_basename VARCHAR, time_seconds BIGINT, name VARCHAR, value DOUBLE)"
    ),
    "leak_demand_long": (
        "CREATE TABLE IF NOT EXISTS leak_demand_long "
        "(scenario_basename VARCHAR, time_seconds BIGINT, name VARCHAR, value DOUBLE)"
    ),
    "labels_long": (
        "CREATE TABLE IF NOT EXISTS labels_long "
        "(scenario_basename VARCHAR, time_seconds BIGINT, label TINYINT)"
    ),
    "masks_long": (
        "CREATE TABLE IF NOT EXISTS masks_long "
        "(scenario_basename VARCHAR, time_seconds BIGINT, mask VARCHAR, value BOOLEAN)"
    ),
}

# Indexes on (scenario_basename, name, time_seconds) per decision D32:
# point/range lookups for one scenario and channel over a time window.
LONG_TABLE_INDEXES = {
    "pressure_long": "CREATE INDEX IF NOT EXISTS idx_pressure_long "
    "ON pressure_long(scenario_basename, name, time_seconds)",
    "flowrate_long": "CREATE INDEX IF NOT EXISTS idx_flowrate_long "
    "ON flowrate_long(scenario_basename, name, time_seconds)",
    "demand_long": "CREATE INDEX IF NOT EXISTS idx_demand_long "
    "ON demand_long(scenario_basename, name, time_seconds)",
    "leak_demand_long": "CREATE INDEX IF NOT EXISTS idx_leak_demand_long "
    "ON leak_demand_long(scenario_basename, name, time_seconds)",
    "labels_long": "CREATE INDEX IF NOT EXISTS idx_labels_long "
    "ON labels_long(scenario_basename, time_seconds)",
    "masks_long": "CREATE INDEX IF NOT EXISTS idx_masks_long "
    "ON masks_long(scenario_basename, mask, time_seconds)",
}


def _ensure_long_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create the consolidated long tables and their indexes on first use."""

    for ddl in LONG_TABLE_DDLS.values():
        con.execute(ddl)
    for ddl in LONG_TABLE_INDEXES.values():
        con.execute(ddl)


def _melt_for_long(df: pd.DataFrame, value_cols: list[str], var_name: str, value_name: str) -> pd.DataFrame:
    """Melt selected columns of a time-indexed table into long form."""

    long_df = (
        df[value_cols]
        .rename_axis("time_seconds")
        .reset_index()
        .melt(id_vars="time_seconds", var_name=var_name, value_name=value_name)
    )
    return long_df


def _populate_long_tables(
    con: duckdb.DuckDBPyConnection, basename: str, tables: dict[str, pd.DataFrame]
) -> None:
    """Append one scenario's measurements to the consolidated long tables.

    Idempotent per scenario via DELETE+INSERT, matching the ``scenarios``
    table convention. Tolerates a partial ``tables`` dict (only the
    tables present are melted), so the direct writer API and the runner
    both work. Mask and label columns are excluded from the value long
    tables and routed to ``masks_long`` / ``labels_long`` respectively.
    """

    def _insert(long_table: str, long_df: pd.DataFrame) -> None:
        long_df = long_df.copy()
        long_df.insert(0, "scenario_basename", basename)
        con.execute(f"DELETE FROM {long_table} WHERE scenario_basename = ?", [basename])
        con.register("_wdn_long_tmp", long_df)
        con.execute(f"INSERT INTO {long_table} SELECT * FROM _wdn_long_tmp")
        con.unregister("_wdn_long_tmp")

    for wide_name, long_table in LONG_VALUE_TABLES.items():
        if wide_name not in tables:
            continue
        df = tables[wide_name]
        value_cols = [c for c in df.columns if c != "label" and "_mask_" not in c]
        if not value_cols:
            continue
        _insert(long_table, _melt_for_long(df, value_cols, "name", "value"))

    # Labels: identical across tables; take from whichever value table
    # carries a label column (pressure first).
    for candidate in ("pressure", "flowrate", "demand"):
        df = tables.get(candidate)
        if df is not None and "label" in df.columns:
            labels_df = df[["label"]].rename_axis("time_seconds").reset_index()
            _insert("labels_long", labels_df)
            break

    # Masks: melt any per-channel mask columns from the corrupted tables.
    mask_frames: list[pd.DataFrame] = []
    for candidate in ("pressure", "flowrate"):
        df = tables.get(candidate)
        if df is None:
            continue
        mask_cols = [c for c in df.columns if "_mask_" in c]
        if mask_cols:
            mask_frames.append(_melt_for_long(df, mask_cols, "mask", "value"))
    if mask_frames:
        _insert("masks_long", pd.concat(mask_frames, ignore_index=True))
    else:
        # Keep the table idempotent even when a re-run removes all masks.
        con.execute("DELETE FROM masks_long WHERE scenario_basename = ?", [basename])


def _upsert_scenario_row(
    con: duckdb.DuckDBPyConnection,
    basename: str,
    metadata: dict,
    validation_severity: str | None,
    config_path: str | None,
) -> None:
    """Insert one row into ``scenarios``, replacing any existing row.

    DuckDB does not have ``REPLACE INTO`` syntax for arbitrary primary
    keys; we delete-then-insert under a transaction to keep the table
    idempotent across re-runs of the same scenario.
    """

    fault_summary = metadata.get("fault_summary", {}) or {}
    leaks = fault_summary.get("leaks", []) or []
    sensor_faults = fault_summary.get("sensor_faults", []) or []
    interactions = fault_summary.get("interactions", []) or []

    con.execute("DELETE FROM scenarios WHERE scenario_basename = ?", [basename])
    con.execute(
        """
        INSERT INTO scenarios (
            scenario_basename, config_path, scenario_type, scenario_label,
            network_name, seed, demand_model, duration_seconds,
            num_leaks, num_sensor_faults, num_interactions,
            validation_severity, fault_summary_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            basename,
            config_path,
            metadata.get("scenario_type"),
            metadata.get("scenario_label"),
            metadata.get("network_name"),
            metadata.get("seed"),
            metadata.get("demand_model"),
            metadata.get("duration_seconds"),
            len(leaks),
            len(sensor_faults),
            len(interactions),
            validation_severity,
            json.dumps(fault_summary, sort_keys=True, default=str),
        ],
    )


class DuckDBWriter:
    """Append one scenario's tables to a consolidated DuckDB file.

    The writer is independent of :class:`Writer` because its lifetime
    spans multiple scenarios (one DB file accumulates the whole batch)
    and it needs a target DB path that is not part of the per-scenario
    output directory contract. Every scenario's tables are materialised
    into the same DuckDB file, namespaced by ``{basename}_{table_name}``.
    A ``scenarios`` metadata table carries one row per scenario for
    cross-scenario queries.

    DuckDB does not support concurrent writers across processes; the
    batch driver (:mod:`wdn_pipeline.batch`) serialises DuckDB writes in
    the main process even when worker scenarios run in parallel.
    """

    extension = "duckdb"

    @staticmethod
    def write(
        db_path: Path,
        basename: str,
        tables: dict[str, pd.DataFrame],
        metadata: dict,
        validation_severity: str | None = None,
        config_path: str | None = None,
        wide_tables: bool = True,
    ) -> Path:
        """Materialise ``tables`` into ``db_path`` under ``basename``.

        Args:
            db_path: Target DuckDB file. Parent directory is created on
                demand.
            basename: Scenario stem used to namespace the inserted
                tables (``{basename}_pressure`` and so on).
            tables: Table name to DataFrame mapping (e.g. the output of
                :func:`assemble_tables`).
            metadata: Sidecar metadata dictionary; the scenario row is
                derived from it.
            validation_severity: Optional severity recorded into the
                ``scenarios`` row.
            config_path: Optional config path recorded into the
                ``scenarios`` row for traceability.
            wide_tables: When true (default, D30) write the per-scenario
                wide tables. When false (D35) write only the consolidated
                long tables and the ``scenarios`` row, keeping a large
                consolidated file compact (each wide table otherwise
                costs a DuckDB storage block).

        Returns:
            ``db_path`` as a :class:`Path`.
        """

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(db_path)) as con:
            _ensure_scenarios_table(con)
            _ensure_long_tables(con)
            if wide_tables:
                for table_name, df in tables.items():
                    full_table = f"{basename}_{table_name}"
                    df_to_write = df.reset_index().rename(
                        columns={"index": "time_seconds"}
                    )
                    con.register("_wdn_pipeline_tmp_df", df_to_write)
                    con.execute(f'DROP TABLE IF EXISTS "{full_table}"')
                    con.execute(
                        f'CREATE TABLE "{full_table}" AS '
                        "SELECT * FROM _wdn_pipeline_tmp_df"
                    )
                    con.unregister("_wdn_pipeline_tmp_df")
            _populate_long_tables(con, basename, tables)
            _upsert_scenario_row(
                con, basename, metadata, validation_severity, config_path
            )
        return db_path


def build_basename(network_name: str, scenario_label: str, seed: int) -> str:
    """Filename stem for one scenario.

    Format: ``{network}_{scenario}_{seed}``. May be revisited if batch
    runs need uniqueness across re-runs (decision D11).
    """

    safe_label = scenario_label.replace(" ", "_").replace("/", "-")
    safe_network = network_name.replace(" ", "_").replace("/", "-")
    return f"{safe_network}_{safe_label}_{seed}"


def assemble_tables(
    results: SimulationResults,
    labels: Labels,
    sensor_masks: dict[str, pd.Series] | None = None,
) -> dict[str, pd.DataFrame]:
    """Attach the per-timestep label column to every table.

    The label is the same column repeated across pressure / flowrate /
    demand to make every table independently filterable.

    Per-channel sensor-fault masks (D22, D26) are appended to the
    matching corrupted table: a pressure fault's mask column is added
    to ``pressure``, a flowrate fault's mask is added to ``flowrate``.
    Mask column names follow ``{fault_type}_mask_{target}``. The
    clean-signal tables and the demand / leak_demand tables do not
    carry mask columns: a downstream consumer reading the clean tables
    cares about ground truth only.
    """

    out: dict[str, pd.DataFrame] = {}
    for name in TABLE_NAMES:
        df = getattr(results, name).copy()
        # The label series is indexed identically to the simulation tables.
        df["label"] = labels.timestep_labels.reindex(df.index).fillna(0).astype("int8")
        out[name] = df

    if sensor_masks:
        for mask_name, mask_series in sensor_masks.items():
            # Names follow {type}_mask_{target}. We dispatch to the
            # right corrupted table by checking which column the target
            # belongs to. The target may equal a column name in both
            # tables in pathological topologies, but the convention is
            # that pressure faults populate the pressure mask and
            # flowrate faults populate the flowrate mask, so we route
            # by membership in the corrupted column set.
            target = mask_name.split("_mask_", 1)[1] if "_mask_" in mask_name else None
            if target is not None and target in out["pressure"].columns:
                out["pressure"][mask_name] = (
                    mask_series.reindex(out["pressure"].index)
                    .fillna(False)
                    .astype(bool)
                )
            if target is not None and target in out["flowrate"].columns:
                out["flowrate"][mask_name] = (
                    mask_series.reindex(out["flowrate"].index)
                    .fillna(False)
                    .astype(bool)
                )
    return out


def write_metadata_sidecar(directory: Path, basename: str, metadata: dict) -> Path:
    """Write the scenario metadata as a YAML sidecar.

    Returns the path to the written file.
    """

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{basename}.meta.yaml"
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=True)
    return target


def write_outputs(
    directory: Path,
    basename: str,
    formats: list[str],
    tables: dict[str, pd.DataFrame],
    metadata: dict,
    write_metadata_sidecar_flag: bool,
    duckdb_path: Path | None = None,
    validation_severity: str | None = None,
    config_path: str | None = None,
    duckdb_wide_tables: bool = True,
) -> WriteResult:
    """Write tables in every requested format plus an optional sidecar.

    When ``duckdb_path`` is provided the scenario's tables are also
    inserted into the DuckDB file under the ``{basename}_{table}``
    namespace and a row is upserted into the ``scenarios`` metadata
    table (decision D30). DuckDB is additive: the parquet/csv outputs
    are written exactly as before.
    """

    data_paths: list[Path] = []
    for fmt in formats:
        writer_cls = WRITERS.get(fmt)
        if writer_cls is None:
            raise ValueError(f"Unknown output format: {fmt}. Known: {sorted(WRITERS)}")
        data_paths.extend(writer_cls().write(directory, basename, tables, metadata))

    if duckdb_path is not None:
        db_path = DuckDBWriter.write(
            db_path=duckdb_path,
            basename=basename,
            tables=tables,
            metadata=metadata,
            validation_severity=validation_severity,
            config_path=config_path,
            wide_tables=duckdb_wide_tables,
        )
        data_paths.append(db_path)

    meta_path = (
        write_metadata_sidecar(directory, basename, metadata)
        if write_metadata_sidecar_flag
        else None
    )
    return WriteResult(data_paths=data_paths, metadata_path=meta_path)
