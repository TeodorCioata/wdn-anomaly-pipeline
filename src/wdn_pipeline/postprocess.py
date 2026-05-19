"""Optional output post-processing: revert leak-node split artefacts.

When a leak is injected the pipeline splits a pipe and inserts a new
junction (the leak node). That junction and the extra pipe segment show
up as additional columns in the output tables, so a leak scenario and a
normal scenario on the same network no longer share a column schema.
For an ML consumer this is two problems at once: the feature count
changes between classes and the mere presence of a leak-node column
trivially leaks the label.

This module implements the optional cleanup step. It adapts the reference
implementation to this pipeline's naming conventions:

- leak nodes are named ``leak_{idx}_{pipe}``
- the downstream split segment is named ``pipe_{idx}_{pipe}_B``
- the upstream split segment keeps the original pipe name

The cleanup:

1. drops every leak-node column from the node-valued tables
   (``pressure``, ``demand`` and their clean siblings),
2. for the link-valued tables (``flowrate`` and its clean sibling)
   drops the upstream segment's flow column and renames the downstream
   segment's column back to the original pipe name, and
3. leaves the ``leak_demand`` diagnostic table untouched (it is a
   separate table, not fed to ML).

The step runs after labelling and before serialisation. It is a no-op
when no leaks were injected. Per-timestep ``label`` columns and
per-channel sensor-fault mask columns are preserved verbatim: they
never carry a bare leak-node or split-segment name.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from wdn_pipeline.faults.leak import ResolvedLeak

logger = logging.getLogger("wdn_pipeline.postprocess")

# Output tables whose columns are node names.
_NODE_TABLES = ("pressure", "demand", "pressure_clean")
# Output tables whose columns are link names.
_LINK_TABLES = ("flowrate", "flowrate_clean")
# Never touched by the cleanup.
_UNTOUCHED_TABLES = ("leak_demand",)


def remove_leak_artifacts(
    tables: dict[str, pd.DataFrame],
    resolved_leaks: Sequence[ResolvedLeak],
) -> dict[str, pd.DataFrame]:
    """Return a copy of ``tables`` with leak-node split artefacts reverted.

    Args:
        tables: The assembled output tables keyed by table name (the
            output of :func:`wdn_pipeline.output.assemble_tables`).
        resolved_leaks: The leaks realised by the leak injector. Each
            carries the inserted ``leak_node_name``, the original
            ``pipe`` name (the upstream segment after the split) and the
            ``new_pipe_name`` (the downstream segment).

    Returns:
        A new dict of new DataFrames. The input is not mutated. When
        ``resolved_leaks`` is empty the tables are returned copied but
        otherwise unchanged.
    """

    if not resolved_leaks:
        return {name: df.copy() for name, df in tables.items()}

    leak_node_names = {leak.leak_node_name for leak in resolved_leaks}
    out: dict[str, pd.DataFrame] = {}

    for name, df in tables.items():
        df = df.copy()
        if name in _UNTOUCHED_TABLES:
            out[name] = df
            continue
        if name in _NODE_TABLES:
            drop = [c for c in df.columns if c in leak_node_names]
            if drop:
                df = df.drop(columns=drop)
                logger.info(
                    "Cleanup: dropped %d leak-node column(s) from %s: %s",
                    len(drop),
                    name,
                    drop,
                )
        elif name in _LINK_TABLES:
            df = _revert_split_pipes(df, resolved_leaks, name)
        out[name] = df

    return out


def _revert_split_pipes(
    df: pd.DataFrame,
    resolved_leaks: Sequence[ResolvedLeak],
    table_name: str,
) -> pd.DataFrame:
    """Collapse split pipe segments back to the original pipe column.

    For each leak the upstream segment keeps the original pipe name and
    the downstream segment is ``new_pipe_name``. Following the
    supervisor's reference, the upstream column is dropped and the
    downstream column is renamed to the original pipe name, so the
    output carries a single flow column under the original pipe name.
    """

    for leak in resolved_leaks:
        upstream = leak.pipe
        downstream = leak.new_pipe_name
        if downstream not in df.columns:
            # Either no flow column for this segment or already reverted.
            continue
        if upstream in df.columns:
            df = df.drop(columns=[upstream])
        df = df.rename(columns={downstream: upstream})
        logger.info(
            "Cleanup: %s collapsed split pipe (%s, %s) back to %s",
            table_name,
            upstream,
            downstream,
            upstream,
        )
    return df
