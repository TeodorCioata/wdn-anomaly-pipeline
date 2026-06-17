"""Tests for the leak-node cleanup post-processing step (Week 5).

Covers:

- The :func:`remove_leak_artifacts` transformation at the unit level.
- End-to-end runs of a leak config with ``output.remove_leak_nodes``
  enabled and disabled, asserting the output column schema.
- The invariants the supervisor's reference cleanup must preserve:
  the ``leak_demand`` table is never touched, labels survive, and the
  step is a no-op on a no-leak scenario.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import wntr
import yaml

from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.faults.leak import ResolvedLeak
from wdn_pipeline.postprocess import remove_leak_artifacts
from wdn_pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolved_leak(
    pipe: str = "P1",
    leak_node: str = "leak_0_P1",
    new_pipe: str = "pipe_0_P1_B",
) -> ResolvedLeak:
    return ResolvedLeak(
        pipe=pipe,
        split_fraction=0.5,
        area_m2=0.01,
        diameter_m=None,
        discharge_coeff=0.75,
        start_time_seconds=0,
        end_time_seconds=3600,
        profile="abrupt",
        profile_steps=1,
        leak_node_name=leak_node,
        new_pipe_name=new_pipe,
        name=None,
    )


def _synthetic_tables() -> dict[str, pd.DataFrame]:
    idx = pd.Index([0, 3600, 7200], name="time_seconds")
    pressure = pd.DataFrame({"J1": 1.0, "J2": 2.0, "leak_0_P1": 9.0, "label": 0}, index=idx)
    demand = pd.DataFrame({"J1": 0.1, "J2": 0.2, "leak_0_P1": 0.0, "label": 0}, index=idx)
    flowrate = pd.DataFrame({"P1": 0.5, "pipe_0_P1_B": 0.4, "P2": 0.3, "label": 0}, index=idx)
    leak_demand = pd.DataFrame({"J1": 0.0, "J2": 0.0, "leak_0_P1": 0.05}, index=idx)
    return {
        "pressure": pressure,
        "flowrate": flowrate,
        "demand": demand,
        "leak_demand": leak_demand,
        "pressure_clean": pressure.copy(),
        "flowrate_clean": flowrate.copy(),
    }


# ---------------------------------------------------------------------------
# Unit-level behaviour
# ---------------------------------------------------------------------------


def test_remove_leak_artifacts_drops_leak_node_columns() -> None:
    out = remove_leak_artifacts(_synthetic_tables(), [_resolved_leak()])
    assert "leak_0_P1" not in out["pressure"].columns
    assert "leak_0_P1" not in out["demand"].columns
    assert "leak_0_P1" not in out["pressure_clean"].columns


def test_remove_leak_artifacts_collapses_split_pipes() -> None:
    out = remove_leak_artifacts(_synthetic_tables(), [_resolved_leak()])
    # Downstream segment renamed back to the original pipe name; the
    # upstream segment column is dropped.
    assert "pipe_0_P1_B" not in out["flowrate"].columns
    assert "P1" in out["flowrate"].columns
    # The surviving column carries the downstream segment's values.
    assert (out["flowrate"]["P1"] == 0.4).all()
    assert "pipe_0_P1_B" not in out["flowrate_clean"].columns
    assert "P1" in out["flowrate_clean"].columns


def test_remove_leak_artifacts_leaves_leak_demand_untouched() -> None:
    tables = _synthetic_tables()
    out = remove_leak_artifacts(tables, [_resolved_leak()])
    pd.testing.assert_frame_equal(out["leak_demand"], tables["leak_demand"])
    assert "leak_0_P1" in out["leak_demand"].columns


def test_remove_leak_artifacts_preserves_label_column() -> None:
    out = remove_leak_artifacts(_synthetic_tables(), [_resolved_leak()])
    for name in ("pressure", "flowrate", "demand"):
        assert "label" in out[name].columns


def test_remove_leak_artifacts_no_leaks_is_noop() -> None:
    tables = _synthetic_tables()
    out = remove_leak_artifacts(tables, [])
    for name, df in tables.items():
        pd.testing.assert_frame_equal(out[name], df)


def test_remove_leak_artifacts_does_not_mutate_input() -> None:
    tables = _synthetic_tables()
    before = {name: df.copy() for name, df in tables.items()}
    remove_leak_artifacts(tables, [_resolved_leak()])
    for name, df in before.items():
        pd.testing.assert_frame_equal(tables[name], df)


def test_remove_leak_artifacts_keeps_unrelated_columns() -> None:
    out = remove_leak_artifacts(_synthetic_tables(), [_resolved_leak()])
    assert {"J1", "J2"}.issubset(out["pressure"].columns)
    assert "P2" in out["flowrate"].columns


# ---------------------------------------------------------------------------
# End-to-end through the runner
# ---------------------------------------------------------------------------


def _leak_config(tmp_path: Path, remove_leak_nodes: bool) -> PipelineConfig:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "leak_abrupt_net3.yaml").read_text())
    raw["output"]["directory"] = str(tmp_path / "outputs")
    raw["output"]["remove_leak_nodes"] = remove_leak_nodes
    raw["output"]["formats"] = ["parquet"]
    return PipelineConfig.model_validate(raw)


def _columns(path: Path) -> set[str]:
    df = pq.read_table(path).to_pandas()
    return set(df.columns) - {"time_seconds", "label"}


def test_cleanup_disabled_keeps_leak_artifacts(tmp_path: Path) -> None:
    summary = run(_leak_config(tmp_path, remove_leak_nodes=False))
    leak = summary.resolved_leaks[0]
    pressure_path = next(p for p in summary.output_paths if p.name.endswith("pressure.parquet"))
    flow_path = next(p for p in summary.output_paths if p.name.endswith("flowrate.parquet"))
    assert leak.leak_node_name in _columns(pressure_path)
    assert leak.new_pipe_name in _columns(flow_path)


def test_cleanup_enabled_matches_original_topology(tmp_path: Path) -> None:
    summary = run(_leak_config(tmp_path, remove_leak_nodes=True))
    wn = wntr.network.WaterNetworkModel("Net3")
    pressure_path = next(p for p in summary.output_paths if p.name.endswith("pressure.parquet"))
    flow_path = next(p for p in summary.output_paths if p.name.endswith("flowrate.parquet"))
    assert _columns(pressure_path) == set(wn.node_name_list)
    assert _columns(flow_path) == set(wn.link_name_list)


def test_cleanup_enabled_leak_demand_still_has_leak_node(tmp_path: Path) -> None:
    summary = run(_leak_config(tmp_path, remove_leak_nodes=True))
    leak = summary.resolved_leaks[0]
    leak_demand_path = next(
        p for p in summary.output_paths if p.name.endswith("leak_demand.parquet")
    )
    assert leak.leak_node_name in _columns(leak_demand_path)


def test_cleanup_enabled_preserves_labels(tmp_path: Path) -> None:
    summary = run(_leak_config(tmp_path, remove_leak_nodes=True))
    pressure_path = next(p for p in summary.output_paths if p.name.endswith("pressure.parquet"))
    df = pq.read_table(pressure_path).to_pandas().set_index("time_seconds")
    assert "label" in df.columns
    expected = ((df.index >= 21600) & (df.index < 64800)).astype(int)
    assert list(df["label"].astype(int)) == list(expected)


def test_cleanup_noop_on_normal_scenario(tmp_path: Path) -> None:
    """remove_leak_nodes on a no-leak scenario leaves the schema intact."""

    raw = yaml.safe_load((REPO_ROOT / "configs" / "normal_net3_pdd.yaml").read_text())
    raw["output"]["directory"] = str(tmp_path / "outputs")
    raw["output"]["remove_leak_nodes"] = True
    raw["output"]["formats"] = ["parquet"]
    summary = run(PipelineConfig.model_validate(raw))
    wn = wntr.network.WaterNetworkModel("Net3")
    pressure_path = next(p for p in summary.output_paths if p.name.endswith("pressure.parquet"))
    assert _columns(pressure_path) == set(wn.node_name_list)
