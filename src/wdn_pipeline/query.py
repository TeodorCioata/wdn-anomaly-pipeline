"""Partial-retrieval query layer over the consolidated DuckDB file.

Phase 5 Week 8. Research consumers should never have
to download a whole dataset to use a slice of it. :class:`DatasetQuery`
wraps a read-only DuckDB connection and exposes parametrised slice
methods over the materialised long tables (``pressure_long`` etc.) and
the ``scenarios`` catalogue:

- filter by scenario or scenario set, by node/link set, by time window
  (half-open ``[t_start, t_end)`` consistent with the fault-window
  convention) and by scenario class;
- return tidy long-form pandas DataFrames, or pivot to the
  timestep x channel matrix for ML use (``wide=True``);
- escape to raw SQL via :meth:`sql` when a query outgrows the builder.

Every method builds parametrised SQL and binds values (never string
interpolation), so a node named ``'; DROP TABLE x;--`` is treated as
data, not SQL. The connection opens read-only by default so concurrent
readers are safe and a query can never mutate the dataset.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# Quantity -> (long table, entity column label used in the public API).
_VALUE_TABLES = {
    "pressure": "pressure_long",
    "flowrate": "flowrate_long",
    "demand": "demand_long",
    "leak_demand": "leak_demand_long",
    "quality": "quality_long",
}


class DatasetQuery:
    """Read-only, sliceable view over a pipeline DuckDB file.

    Args:
        db_path: Path to a DuckDB file produced by the pipeline (a batch
            ``--duckdb`` run or a per-config ``output.duckdb``).
        read_only: Open the connection read-only (default). Set ``False``
            only if a caller deliberately needs to write.

    Use as a context manager to close the connection deterministically::

        with DatasetQuery("outputs/scale_run.duckdb") as q:
            df = q.pressure(scenario="net3_leak_abrupt_42", nodes=["10"])
    """

    def __init__(self, db_path: str | Path, read_only: bool = True) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"DuckDB file not found: {self.db_path}")
        self._con = duckdb.connect(str(self.db_path), read_only=read_only)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> DatasetQuery:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _scenario_list(scenario: str | None, scenarios: list[str] | None) -> list[str] | None:
        out: list[str] = []
        if scenario is not None:
            out.append(scenario)
        if scenarios:
            out.extend(scenarios)
        return out or None

    @staticmethod
    def _in_clause(column: str, values: list[str], params: list[object]) -> str:
        placeholders = ", ".join("?" for _ in values)
        params.extend(values)
        return f"{column} IN ({placeholders})"

    def _time_clauses(
        self, t_start: int | None, t_end: int | None, where: list[str], params: list[object]
    ) -> None:
        if t_start is not None:
            where.append("time_seconds >= ?")
            params.append(int(t_start))
        if t_end is not None:
            # Half-open [t_start, t_end).
            where.append("time_seconds < ?")
            params.append(int(t_end))

    # -- catalogue ---------------------------------------------------------

    def scenarios(
        self,
        scenario_type: str | None = None,
        network: str | None = None,
        severity: str | None = None,
    ) -> pd.DataFrame:
        """Browse the ``scenarios`` catalogue, optionally filtered."""

        where: list[str] = []
        params: list[object] = []
        if scenario_type is not None:
            where.append("scenario_type = ?")
            params.append(scenario_type)
        if network is not None:
            where.append("network_name = ?")
            params.append(network)
        if severity is not None:
            where.append("validation_severity = ?")
            params.append(severity)
        sql = "SELECT * FROM scenarios"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY scenario_basename"
        return self._con.execute(sql, params).fetchdf()

    # -- value queries -----------------------------------------------------

    def _query_value(
        self,
        quantity: str,
        scenario: str | None,
        scenarios: list[str] | None,
        names: list[str] | None,
        t_start: int | None,
        t_end: int | None,
        wide: bool,
    ) -> pd.DataFrame:
        table = _VALUE_TABLES[quantity]
        where: list[str] = []
        params: list[object] = []
        scen = self._scenario_list(scenario, scenarios)
        if scen is not None:
            where.append(self._in_clause("scenario_basename", scen, params))
        if names:
            where.append(self._in_clause("name", list(names), params))
        self._time_clauses(t_start, t_end, where, params)

        sql = f"SELECT scenario_basename, time_seconds, name, value FROM {table}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY scenario_basename, name, time_seconds"
        df = self._con.execute(sql, params).fetchdf()
        if not wide:
            return df
        return self._pivot_wide(df)

    @staticmethod
    def _pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
        """Pivot long value rows to a timestep x channel matrix.

        Index is ``time_seconds`` for a single scenario, otherwise a
        ``(scenario_basename, time_seconds)`` MultiIndex.
        """

        if df.empty:
            return df
        wide = df.pivot_table(
            index=["scenario_basename", "time_seconds"],
            columns="name",
            values="value",
        )
        wide.columns.name = None
        if wide.index.get_level_values("scenario_basename").nunique() == 1:
            wide = wide.droplevel("scenario_basename")
        return wide

    def pressure(
        self,
        scenario: str | None = None,
        scenarios: list[str] | None = None,
        nodes: list[str] | None = None,
        t_start: int | None = None,
        t_end: int | None = None,
        wide: bool = False,
    ) -> pd.DataFrame:
        """Slice pressure (node-addressed). All filters optional."""

        return self._query_value("pressure", scenario, scenarios, nodes, t_start, t_end, wide)

    def flowrate(
        self,
        scenario: str | None = None,
        scenarios: list[str] | None = None,
        links: list[str] | None = None,
        t_start: int | None = None,
        t_end: int | None = None,
        wide: bool = False,
    ) -> pd.DataFrame:
        """Slice flowrate (link-addressed). All filters optional."""

        return self._query_value("flowrate", scenario, scenarios, links, t_start, t_end, wide)

    def demand(
        self,
        scenario: str | None = None,
        scenarios: list[str] | None = None,
        nodes: list[str] | None = None,
        t_start: int | None = None,
        t_end: int | None = None,
        wide: bool = False,
    ) -> pd.DataFrame:
        """Slice delivered demand (node-addressed)."""

        return self._query_value("demand", scenario, scenarios, nodes, t_start, t_end, wide)

    def leak_demand(
        self,
        scenario: str | None = None,
        scenarios: list[str] | None = None,
        nodes: list[str] | None = None,
        t_start: int | None = None,
        t_end: int | None = None,
        wide: bool = False,
    ) -> pd.DataFrame:
        """Slice leak outflow (node-addressed)."""

        return self._query_value("leak_demand", scenario, scenarios, nodes, t_start, t_end, wide)

    def quality(
        self,
        scenario: str | None = None,
        scenarios: list[str] | None = None,
        nodes: list[str] | None = None,
        t_start: int | None = None,
        t_end: int | None = None,
        wide: bool = False,
    ) -> pd.DataFrame:
        """Slice water quality (node-addressed).

        Populated only for scenarios that ran a water quality analysis
        (chemical / age / trace). Units follow the analysis: mg/L for
        chemical, seconds for age, percent for trace.
        """

        return self._query_value("quality", scenario, scenarios, nodes, t_start, t_end, wide)

    # -- labels and masks --------------------------------------------------

    def labels(
        self,
        scenario: str | None = None,
        scenarios: list[str] | None = None,
        t_start: int | None = None,
        t_end: int | None = None,
        label: int | None = None,
    ) -> pd.DataFrame:
        """Per-timestep anomaly labels.

        Pass ``label=1`` to fetch only anomalous timesteps (or ``0`` for
        only normal ones).
        """

        where: list[str] = []
        params: list[object] = []
        scen = self._scenario_list(scenario, scenarios)
        if scen is not None:
            where.append(self._in_clause("scenario_basename", scen, params))
        if label is not None:
            where.append("label = ?")
            params.append(int(label))
        self._time_clauses(t_start, t_end, where, params)
        sql = "SELECT scenario_basename, time_seconds, label FROM labels_long"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY scenario_basename, time_seconds"
        return self._con.execute(sql, params).fetchdf()

    def masks(
        self,
        scenario: str | None = None,
        scenarios: list[str] | None = None,
        masks: list[str] | None = None,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> pd.DataFrame:
        """Per-channel sensor-fault masks (long form)."""

        where: list[str] = []
        params: list[object] = []
        scen = self._scenario_list(scenario, scenarios)
        if scen is not None:
            where.append(self._in_clause("scenario_basename", scen, params))
        if masks:
            where.append(self._in_clause("mask", list(masks), params))
        self._time_clauses(t_start, t_end, where, params)
        sql = "SELECT scenario_basename, time_seconds, mask, value FROM masks_long"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY scenario_basename, mask, time_seconds"
        return self._con.execute(sql, params).fetchdf()

    # -- escape hatch ------------------------------------------------------

    def sql(self, query: str, params: list[object] | None = None) -> pd.DataFrame:
        """Run an arbitrary read-only SQL query and return a DataFrame."""

        return self._con.execute(query, params or []).fetchdf()
