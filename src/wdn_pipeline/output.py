"""Serialise simulation results to disk.

The writer set is extensible (decision D6). Adding a queryable database
backend (DuckDB has been floated as a candidate) means writing a new
:class:`Writer` subclass and registering it in :data:`WRITERS`. No
other module changes.

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

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
) -> WriteResult:
    """Write tables in every requested format plus an optional sidecar."""

    data_paths: list[Path] = []
    for fmt in formats:
        writer_cls = WRITERS.get(fmt)
        if writer_cls is None:
            raise ValueError(f"Unknown output format: {fmt}. Known: {sorted(WRITERS)}")
        data_paths.extend(writer_cls().write(directory, basename, tables, metadata))

    meta_path = (
        write_metadata_sidecar(directory, basename, metadata)
        if write_metadata_sidecar_flag
        else None
    )
    return WriteResult(data_paths=data_paths, metadata_path=meta_path)
