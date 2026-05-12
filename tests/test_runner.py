"""End-to-end tests for the runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.runner import run, run_from_config_file


def test_run_net3_end_to_end(net3_config: PipelineConfig) -> None:
    summary = run(net3_config)
    assert summary.network_name == "net3"
    assert summary.num_timesteps == 25
    assert summary.validation.passed  # warning is allowed; fail is not
    assert summary.metadata_path is not None and summary.metadata_path.is_file()
    # Two formats (parquet + csv) x six tables (pressure/flowrate/demand/
    # leak_demand/pressure_clean/flowrate_clean) = 12 files.
    assert len(summary.output_paths) == 12
    for p in summary.output_paths:
        assert p.exists()


def test_run_outputs_round_trip(net3_config: PipelineConfig) -> None:
    summary = run(net3_config)
    parquets = [p for p in summary.output_paths if p.suffix == ".parquet"]
    table = pq.read_table(next(p for p in parquets if "pressure" in p.name))
    df = table.to_pandas()
    assert "label" in df.columns
    assert "time_seconds" in df.columns
    assert (df["label"] == 0).all()


def test_run_is_deterministic(net3_config: PipelineConfig, tmp_path: Path) -> None:
    """Same config run twice produces identical pressure data."""

    cfg2 = net3_config.model_copy(
        update={
            "output": net3_config.output.model_copy(update={"directory": tmp_path / "second"})
        }
    )
    s1 = run(net3_config)
    s2 = run(cfg2)
    p1 = pq.read_table(next(p for p in s1.output_paths if "pressure.parquet" in p.name)).to_pandas()
    p2 = pq.read_table(next(p for p in s2.output_paths if "pressure.parquet" in p.name)).to_pandas()
    pd.testing.assert_frame_equal(p1, p2)


def test_run_from_config_file(tmp_path: Path) -> None:
    """Verify YAML-driven entry point works against a checked-in config."""

    yaml_path = Path("configs/normal_net3.yaml").resolve()
    if not yaml_path.is_file():
        return
    summary = run_from_config_file(yaml_path)
    assert summary.network_name == "net3"
