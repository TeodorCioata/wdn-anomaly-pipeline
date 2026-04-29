"""End-to-end pipeline orchestrator and CLI.

Pipeline order::

    config -> network -> demand -> simulation -> validation -> labels -> output

Both a Python entry point (:func:`run_from_config_file`) and a Typer
CLI are exposed. The CLI is registered as the ``wdn-pipeline`` console
script in ``pyproject.toml``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from wdn_pipeline.config import PipelineConfig, load_config
from wdn_pipeline.demand import apply_demand
from wdn_pipeline.labelling import build_labels
from wdn_pipeline.network import derive_network_name, load_network
from wdn_pipeline.output import (
    WriteResult,
    assemble_tables,
    build_basename,
    write_outputs,
)
from wdn_pipeline.simulation import SimulationResults, run_simulation
from wdn_pipeline.validation import ValidationReport, validate_normal_scenario

logger = logging.getLogger("wdn_pipeline.runner")


@dataclass
class RunSummary:
    """Surface details about one completed pipeline run."""

    network_name: str
    scenario_label: str
    seed: int
    num_timesteps: int
    elapsed_seconds: float
    output_paths: list[Path]
    metadata_path: Path | None
    validation: ValidationReport

    def format(self) -> str:
        lines = [
            f"network         : {self.network_name}",
            f"scenario_label  : {self.scenario_label}",
            f"seed            : {self.seed}",
            f"num_timesteps   : {self.num_timesteps}",
            f"elapsed_seconds : {self.elapsed_seconds:.3f}",
            f"output_paths    : {[str(p) for p in self.output_paths]}",
            f"metadata_path   : {self.metadata_path}",
            self.validation.format(),
        ]
        return "\n".join(lines)


def _seed_everything(seed: int) -> None:
    """Best-effort process-wide deterministic seeding."""

    random.seed(seed)
    np.random.seed(seed)


def run(config: PipelineConfig) -> RunSummary:
    """Execute the pipeline for a single validated config.

    Args:
        config: A validated :class:`PipelineConfig`.

    Returns:
        A :class:`RunSummary` with output paths and validation details.
    """

    _seed_everything(config.seed)
    network_name = derive_network_name(config.network)
    logger.info("Loading network %s", network_name)
    wn = load_network(config.network, config.simulation)

    logger.info("Applying demand strategy: %s", config.demand.mode)
    apply_demand(wn, config.demand, config.seed)

    logger.info("Running WNTR simulation")
    results: SimulationResults = run_simulation(wn)
    logger.info("Simulation finished in %.3fs", results.elapsed_seconds)

    logger.info("Building labels")
    labels = build_labels(config, results.pressure.index, network_name)

    logger.info("Validating outputs")
    report = validate_normal_scenario(wn, results, config.validation)
    logger.info("Validation severity: %s", report.severity.upper())

    basename = build_basename(network_name, config.scenario.label, config.seed)
    tables = assemble_tables(results, labels)

    logger.info("Writing outputs to %s", config.output.directory)
    write_result: WriteResult = write_outputs(
        directory=config.output.directory,
        basename=basename,
        formats=list(config.output.formats),
        tables=tables,
        metadata=labels.metadata,
        write_metadata_sidecar_flag=config.output.write_metadata_sidecar,
    )

    return RunSummary(
        network_name=network_name,
        scenario_label=config.scenario.label,
        seed=config.seed,
        num_timesteps=len(results.pressure.index),
        elapsed_seconds=results.elapsed_seconds,
        output_paths=write_result.data_paths,
        metadata_path=write_result.metadata_path,
        validation=report,
    )


def run_from_config_file(config_path: str | Path) -> RunSummary:
    """Load a YAML config file and run the pipeline."""

    cfg = load_config(config_path)
    return run(cfg)


app = typer.Typer(add_completion=False, help="WDN Anomaly Simulation Pipeline")


@app.command("run")
def cli_run(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to the YAML pipeline config.",
        ),
    ],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable INFO-level logging.")
    ] = False,
) -> None:
    """Run the pipeline for a single config and print a summary."""

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    summary = run_from_config_file(config)
    typer.echo(summary.format())
    if not summary.validation.passed:
        raise typer.Exit(code=1)


def main() -> None:  # pragma: no cover - thin CLI entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
