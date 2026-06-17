"""End-to-end tests for the runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from typer.testing import CliRunner

from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.runner import app, run, run_from_config_file

runner = CliRunner()


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
        update={"output": net3_config.output.model_copy(update={"directory": tmp_path / "second"})}
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


def test_cli_legacy_form_routes_to_run(tmp_path: Path) -> None:
    """``wdn-pipeline <config>`` (no subcommand) routes to the run command.

    Guards the _DefaultCommandGroup routing against click/typer version
    drift: it must prepend the default ``run`` command when the first
    token is a config path rather than a registered subcommand.
    """

    cfg = Path("configs/normal_net3.yaml").resolve()
    if not cfg.is_file():
        return
    result = runner.invoke(app, [str(cfg)])
    assert result.exit_code == 0, result.output
    assert "Validation:" in result.output


def test_cli_run_subcommand_and_help() -> None:
    """The explicit ``run`` subcommand and the help banner both resolve."""

    cfg = Path("configs/normal_net3.yaml").resolve()
    if not cfg.is_file():
        return
    explicit = runner.invoke(app, ["run", str(cfg)])
    assert explicit.exit_code == 0, explicit.output

    helped = runner.invoke(app, ["--help"])
    assert helped.exit_code == 0
    for sub in ("run", "batch", "query"):
        assert sub in helped.output
