"""End-to-end pipeline orchestrator and CLI.

Pipeline order::

    config -> network -> demand -> [leak injection] -> simulation
           -> [sensor fault injection] -> validation -> labels -> output

Both a Python entry point (:func:`run_from_config_file`) and a Typer
CLI are exposed. The CLI is registered as the ``wdn-pipeline`` console
script in ``pyproject.toml``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import click
import numpy as np
import pandas as pd
import typer
from typer.core import TyperGroup

from wdn_pipeline.config import PipelineConfig, load_config
from wdn_pipeline.demand import apply_demand
from wdn_pipeline.faults.leak import LeakInjector, ResolvedLeak
from wdn_pipeline.faults.sensor import (
    ResolvedSensorFault,
    SensorFaultInjector,
)
from wdn_pipeline.labelling import build_labels
from wdn_pipeline.network import derive_network_name, load_network
from wdn_pipeline.output import (
    WriteResult,
    assemble_tables,
    build_basename,
    write_outputs,
)
from wdn_pipeline.postprocess import remove_leak_artifacts
from wdn_pipeline.simulation import SimulationResults, run_simulation
from wdn_pipeline.validation import (
    ValidationReport,
    validate_cumulative_scenario,
    validate_leak_scenario,
    validate_normal_scenario,
    validate_sensor_fault_scenario,
)

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
    resolved_leaks: list[ResolvedLeak]
    resolved_sensor_faults: list[ResolvedSensorFault] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)

    def format(self) -> str:
        lines = [
            f"network         : {self.network_name}",
            f"scenario_label  : {self.scenario_label}",
            f"seed            : {self.seed}",
            f"num_timesteps   : {self.num_timesteps}",
            f"elapsed_seconds : {self.elapsed_seconds:.3f}",
            f"output_paths    : {[str(p) for p in self.output_paths]}",
            f"metadata_path   : {self.metadata_path}",
        ]
        if self.resolved_leaks:
            lines.append(f"resolved_leaks  : {len(self.resolved_leaks)}")
            for i, leak in enumerate(self.resolved_leaks):
                lines.append(
                    f"  [{i}] pipe={leak.pipe} split={leak.split_fraction:.3f} "
                    f"area={leak.area_m2:.3e} m^2 profile={leak.profile} "
                    f"window=[{leak.start_time_seconds},{leak.end_time_seconds}]s"
                )
        if self.resolved_sensor_faults:
            lines.append(
                f"resolved_sensors: {len(self.resolved_sensor_faults)}"
            )
            for i, f in enumerate(self.resolved_sensor_faults):
                lines.append(
                    f"  [{i}] type={f.type} quantity={f.quantity} target={f.target} "
                    f"window=[{f.start_time_seconds},{f.end_time_seconds})s"
                )
        if self.interactions:
            lines.append(
                f"interactions    : {len(self.interactions)} (informational)"
            )
            for i, it in enumerate(self.interactions):
                lines.append(
                    f"  [{i}] {it['kind']}: sensor target {it['sensor_target']} "
                    f"coincides with leak node {it['leak_node']}"
                )
        lines.append(self.validation.format())
        return "\n".join(lines)


def _seed_everything(seed: int) -> None:
    """Best-effort process-wide deterministic seeding."""

    random.seed(seed)
    np.random.seed(seed)


def run(config: PipelineConfig, config_path: str | None = None) -> RunSummary:
    """Execute the pipeline for a single validated config.

    Args:
        config: A validated :class:`PipelineConfig`.
        config_path: Optional source path of the YAML config. Forwarded
            to the DuckDB writer's ``scenarios`` row for traceability.

    Returns:
        A :class:`RunSummary` with output paths and validation details.
    """

    _seed_everything(config.seed)
    network_name = derive_network_name(config.network)
    logger.info("Loading network %s", network_name)
    wn = load_network(config.network, config.simulation)

    logger.info("Applying demand strategy: %s", config.demand.mode)
    apply_demand(wn, config.demand, config.seed)

    # Single RNG drives both leak and sensor-fault randomness. Order
    # matters: leaks resolve first (they may modify the model that
    # the sensor injector picks targets from), then sensors.
    rng = np.random.default_rng(config.seed)

    if config.faults.leaks:
        logger.info("Injecting %d leak(s)", len(config.faults.leaks))
        resolved_leaks = LeakInjector().apply(wn, list(config.faults.leaks), rng)
    else:
        resolved_leaks = []

    logger.info("Running WNTR simulation")
    results: SimulationResults = run_simulation(wn)
    logger.info("Simulation finished in %.3fs", results.elapsed_seconds)

    sensor_masks: dict[str, pd.Series] = {}
    resolved_sensor_faults: list[ResolvedSensorFault] = []
    if config.faults.sensor_faults:
        logger.info(
            "Injecting %d sensor fault(s)", len(config.faults.sensor_faults)
        )
        sensor_result = SensorFaultInjector().apply(
            results, list(config.faults.sensor_faults), rng, wn
        )
        results = sensor_result.results
        resolved_sensor_faults = sensor_result.resolved
        sensor_masks = sensor_result.masks

    logger.info("Building labels")
    labels = build_labels(
        config,
        results.pressure.index,
        network_name,
        resolved_leaks,
        resolved_sensor_faults,
    )

    interactions = labels.metadata.get("fault_summary", {}).get("interactions", [])
    if interactions:
        logger.info(
            "Cumulative interaction(s) recorded (informational): %s",
            "; ".join(
                f"{it['kind']} on {it['sensor_target']}" for it in interactions
            ),
        )

    logger.info("Validating outputs")
    if resolved_leaks and resolved_sensor_faults:
        report = validate_cumulative_scenario(
            wn,
            results,
            resolved_leaks,
            resolved_sensor_faults,
            sensor_masks,
            config.validation,
        )
    elif resolved_sensor_faults:
        report = validate_sensor_fault_scenario(
            wn, results, resolved_sensor_faults, sensor_masks, config.validation
        )
    elif resolved_leaks:
        report = validate_leak_scenario(wn, results, resolved_leaks, config.validation)
    else:
        report = validate_normal_scenario(wn, results, config.validation)
    logger.info("Validation severity: %s", report.severity.upper())

    basename = build_basename(network_name, config.scenario.label, config.seed)
    tables = assemble_tables(results, labels, sensor_masks)

    if config.output.remove_leak_nodes:
        if resolved_leaks:
            logger.info(
                "Reverting leak-node split artefacts (output.remove_leak_nodes)"
            )
            tables = remove_leak_artifacts(tables, resolved_leaks)
        else:
            logger.info(
                "output.remove_leak_nodes set but no leaks injected; cleanup is a no-op"
            )

    logger.info("Writing outputs to %s", config.output.directory)
    duckdb_target = config.output.duckdb_path if config.output.duckdb else None
    write_result: WriteResult = write_outputs(
        directory=config.output.directory,
        basename=basename,
        formats=list(config.output.formats),
        tables=tables,
        metadata=labels.metadata,
        write_metadata_sidecar_flag=config.output.write_metadata_sidecar,
        duckdb_path=duckdb_target,
        validation_severity=report.severity,
        config_path=config_path,
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
        resolved_leaks=resolved_leaks,
        resolved_sensor_faults=resolved_sensor_faults,
        interactions=interactions,
    )


def run_from_config_file(config_path: str | Path) -> RunSummary:
    """Load a YAML config file and run the pipeline."""

    cfg = load_config(config_path)
    return run(cfg, config_path=str(config_path))


class _DefaultCommandGroup(TyperGroup):
    """Typer group that routes unknown subcommands to a default command.

    Lets ``wdn-pipeline configs/foo.yaml`` keep working (the legacy
    single-config form) while ``wdn-pipeline batch ...`` invokes the
    new batch subcommand. When the first positional token is not a
    registered subcommand name, the token is treated as the first
    argument of the default command ``run``.
    """

    default_command_name = "run"

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.exceptions.UsageError:
            if args and not args[0].startswith("-"):
                args.insert(0, self.default_command_name)
                return super().resolve_command(ctx, args)
            raise


app = typer.Typer(
    add_completion=False,
    help="WDN Anomaly Simulation Pipeline",
    no_args_is_help=True,
    cls=_DefaultCommandGroup,
)


@app.command("run")
def cli_run(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to a single YAML pipeline config.",
        ),
    ],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable INFO-level logging.")
    ] = False,
) -> None:
    """Run a single config and print a summary.

    This is the default command: ``wdn-pipeline configs/foo.yaml``
    forwards here transparently when the first argument is not a
    registered subcommand name.
    """

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    summary = run_from_config_file(config)
    typer.echo(summary.format())
    if not summary.validation.passed:
        raise typer.Exit(code=1)


@app.command("batch")
def cli_batch(
    configs: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Explicit YAML config paths or glob patterns (e.g. "
            "'configs/cumulative_*.yaml'). May be combined with "
            "--config-dir.",
        ),
    ] = None,
    config_dir: Annotated[
        Path | None,
        typer.Option(
            "--config-dir",
            "-d",
            exists=True,
            file_okay=False,
            help="Directory to scan for YAML configs (default pattern *.yaml).",
        ),
    ] = None,
    glob_pattern: Annotated[
        str,
        typer.Option(
            "--glob",
            help="Glob pattern used with --config-dir.",
        ),
    ] = "*.yaml",
    report_dir: Annotated[
        Path,
        typer.Option(
            "--report-dir",
            help="Where to write batch_summary.json and batch_summary.txt.",
        ),
    ] = Path("outputs/batch_runs"),
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Number of worker processes. 1 = sequential (default); "
            ">1 uses a ProcessPoolExecutor with the spawn start method.",
        ),
    ] = 1,
    duckdb: Annotated[
        Path | None,
        typer.Option(
            "--duckdb",
            help="If set, consolidate every scenario's tables into this "
            "DuckDB file (decision D30). Overrides any per-config "
            "output.duckdb_path. Adds a 'scenarios' metadata table.",
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable INFO-level logging.")
    ] = False,
) -> None:
    """Run a batch of configs and write summary artefacts.

    A failure of any individual config is captured in the summary and
    does not abort the batch. Exit code is 1 if any run had severity
    ``fail`` or threw an exception, 0 otherwise.

    Phase 5 Week 7: ``--workers N`` enables parallel execution via
    :class:`concurrent.futures.ProcessPoolExecutor`; ``--duckdb PATH``
    consolidates the batch into a single queryable DuckDB file.
    """

    import time as _time

    from wdn_pipeline.batch import resolve_config_paths, run_batch

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config_paths = resolve_config_paths(
        paths=[str(p) for p in configs] if configs else None,
        config_dir=config_dir,
        glob_pattern=glob_pattern,
    )
    if not config_paths:
        raise typer.BadParameter(
            "No config paths resolved. Provide explicit paths, a glob, "
            "or --config-dir."
        )

    batch_id = _time.strftime("%Y%m%dT%H%M%S")
    out_dir = report_dir / batch_id
    summary = run_batch(
        config_paths,
        out_dir,
        batch_id=batch_id,
        workers=workers,
        duckdb_path=duckdb,
    )
    typer.echo(
        f"batch {summary.batch_id}: total={summary.n_total} ok={summary.n_ok} "
        f"warning={summary.n_warning} fail={summary.n_fail} error={summary.n_error} "
        f"workers={summary.workers} elapsed={summary.elapsed_seconds:.2f}s"
    )
    if summary.duckdb_path is not None:
        typer.echo(f"duckdb written to {summary.duckdb_path}")
    typer.echo(f"summary written to {out_dir / 'batch_summary.txt'}")
    if summary.n_fail or summary.n_error:
        raise typer.Exit(code=1)


def main() -> None:  # pragma: no cover - thin CLI entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
