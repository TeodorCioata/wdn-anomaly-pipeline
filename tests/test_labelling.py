"""Tests for labelling."""

from __future__ import annotations

import pandas as pd

from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.labelling import build_labels


def test_normal_labels_all_zero(net3_config: PipelineConfig) -> None:
    idx = pd.RangeIndex(start=0, stop=10) * 3600
    labels = build_labels(net3_config, idx, "net3")
    assert (labels.timestep_labels == 0).all()
    assert labels.timestep_labels.dtype == "int8"
    assert labels.timestep_labels.name == "label"


def test_metadata_captures_scenario(net3_config: PipelineConfig) -> None:
    idx = pd.RangeIndex(start=0, stop=2) * 3600
    labels = build_labels(net3_config, idx, "net3")
    md = labels.metadata
    assert md["scenario_type"] == "normal"
    assert md["scenario_label"] == "normal"
    assert md["network_name"] == "net3"
    assert md["seed"] == 42
    assert md["fault_summary"] == {
        "leaks": [],
        "sensor_faults": [],
        "interactions": [],
    }
    assert md["demand_mode"] == "default"


def test_label_index_matches_input(net3_config: PipelineConfig) -> None:
    idx = pd.Index([0, 3600, 7200])
    labels = build_labels(net3_config, idx, "net3")
    assert labels.timestep_labels.index.equals(idx)
