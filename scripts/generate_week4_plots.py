"""Generate the Week 4 sensor-fault verification plots.

Produces verification artefacts under ``outputs/plots/`` covering:

- Per-fault-type illustrative figures: clean vs corrupted pressure at
  the faulted node on Net3, with the fault window shaded.
- Per-fault-type residual plots: ``corrupted - clean`` at the faulted
  node over time. For bias this is a flat line at ``bias_value``; for
  drift it is a ramp; for stuck it mirrors the inverted clean signal;
  for dropout it shows NaN gaps; for noise it shows the additive
  process.
- ``validation_summary_week4.txt``: per-config severity report covering
  every normal, leak, and sensor config currently in ``configs/``.

The script re-runs every config end to end so the plots are never stale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from wdn_pipeline.config import PipelineConfig, load_config  # noqa: E402
from wdn_pipeline.demand import apply_demand  # noqa: E402
from wdn_pipeline.faults.leak import LeakInjector  # noqa: E402
from wdn_pipeline.faults.sensor import SensorFaultInjector  # noqa: E402
from wdn_pipeline.network import load_network  # noqa: E402
from wdn_pipeline.runner import run  # noqa: E402
from wdn_pipeline.simulation import SimulationResults, run_simulation  # noqa: E402

logging.basicConfig(level=logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PLOTS_DIR = REPO / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SensorRun:
    """Convenience bundle for sensor-fault plotting."""

    label: str
    config: PipelineConfig
    results: SimulationResults
    target: str
    fault_type: str
    start_time: int
    end_time: int


def _save(fig: Figure, name: str) -> Path:
    target = PLOTS_DIR / name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def _shade_fault_window(ax: plt.Axes, start: int, end: int) -> None:
    ax.axvspan(start / 3600.0, end / 3600.0, alpha=0.15, color="red")


def _run_sensor_scenario(config_name: str) -> SensorRun:
    """Run one sensor config end-to-end through the same code path the runner uses."""

    cfg = load_config(CONFIGS / f"{config_name}.yaml")
    np_seed = cfg.seed
    wn = load_network(cfg.network, cfg.simulation)
    apply_demand(wn, cfg.demand, np_seed)

    rng = np.random.default_rng(np_seed)
    if cfg.faults.leaks:
        LeakInjector().apply(wn, list(cfg.faults.leaks), rng)
    results = run_simulation(wn)
    sf = SensorFaultInjector().apply(
        results, list(cfg.faults.sensor_faults), rng, wn
    )
    fault = sf.resolved[0]
    return SensorRun(
        label=cfg.scenario.label,
        config=cfg,
        results=sf.results,
        target=fault.target,
        fault_type=fault.type,
        start_time=fault.start_time_seconds,
        end_time=fault.end_time_seconds,
    )


def _plot_clean_vs_corrupted(run_: SensorRun, outfile: str) -> Path:
    clean = run_.results.pressure_clean[run_.target]
    corrupted = run_.results.pressure[run_.target]
    t_hours = corrupted.index.to_numpy() / 3600.0
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(t_hours, clean.to_numpy(), label="clean", linewidth=2.0)
    ax.plot(
        t_hours,
        corrupted.to_numpy(),
        label="corrupted",
        linewidth=1.6,
        linestyle="--",
        color="C3",
    )
    _shade_fault_window(ax, run_.start_time, run_.end_time)
    ax.set_xlabel("time [h]")
    ax.set_ylabel(f"pressure at node {run_.target} [m]")
    ax.set_title(f"Sensor {run_.fault_type} fault on Net3 (node {run_.target})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, outfile)


def _plot_residual(run_: SensorRun, outfile: str) -> Path:
    clean = run_.results.pressure_clean[run_.target].to_numpy()
    corrupted = run_.results.pressure[run_.target].to_numpy()
    residual = corrupted - clean
    t_hours = run_.results.pressure.index.to_numpy() / 3600.0
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(t_hours, residual, color="C2", linewidth=1.6)
    _shade_fault_window(ax, run_.start_time, run_.end_time)
    ax.axhline(0.0, color="black", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("corrupted - clean [m]")
    ax.set_title(
        f"Residual at node {run_.target} ({run_.fault_type}, "
        f"window [{run_.start_time // 3600}h, {run_.end_time // 3600}h))"
    )
    ax.grid(True, alpha=0.3)
    return _save(fig, outfile)


def _format_check(name: str, severity: str, detail: str, max_len: int = 80) -> str:
    truncated = detail if len(detail) <= max_len else detail[: max_len - 3] + "..."
    return f"  [{severity:>7}] {name}: {truncated}"


def _summary_lines(label: str, summary) -> list[str]:
    lines = [f"=== {label} ==="]
    lines.append(f"  network         : {summary.network_name}")
    lines.append(f"  scenario_label  : {summary.scenario_label}")
    lines.append(f"  severity        : {summary.validation.severity.upper()}")
    if summary.resolved_leaks:
        lines.append(f"  resolved_leaks  : {len(summary.resolved_leaks)}")
    if summary.resolved_sensor_faults:
        lines.append(
            f"  resolved_sensors: {len(summary.resolved_sensor_faults)}"
        )
        for f in summary.resolved_sensor_faults:
            lines.append(
                f"    - type={f.type} quantity={f.quantity} target={f.target} "
                f"window=[{f.start_time_seconds}, {f.end_time_seconds})s"
            )
    for c in summary.validation.checks:
        lines.append(_format_check(c.name, c.severity, c.detail))
    return lines


def main() -> None:
    sensor_configs = [
        "sensor_bias_net3",
        "sensor_drift_net3",
        "sensor_stuck_net3",
        "sensor_dropout_net3",
        "sensor_noise_net3",
    ]

    written: list[Path] = []
    for cfg_name in sensor_configs:
        run_ = _run_sensor_scenario(cfg_name)
        fault_type = run_.fault_type
        written.append(
            _plot_clean_vs_corrupted(run_, f"sensor_{fault_type}_net3.png")
        )
        # Residual plots are most informative for bias / drift / stuck;
        # we still emit them for dropout and noise for completeness.
        written.append(
            _plot_residual(
                run_,
                f"sensor_fault_residual_{fault_type}_net3.png",
            )
        )

    # Run every config (normal + leak + sensor) for the validation summary.
    all_configs = sorted(p.stem for p in CONFIGS.glob("*.yaml"))
    summary_lines: list[str] = ["Week 4 validation summary", "=" * 30, ""]
    for cfg_name in all_configs:
        cfg = load_config(CONFIGS / f"{cfg_name}.yaml")
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(
                    update={"directory": REPO / "outputs"}
                )
            }
        )
        summary = run(cfg)
        summary_lines.extend(_summary_lines(cfg_name, summary))
        summary_lines.append("")
    summary_path = PLOTS_DIR / "validation_summary_week4.txt"
    summary_path.write_text("\n".join(summary_lines))
    written.append(summary_path)

    print("Wrote:")
    for p in written:
        print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
