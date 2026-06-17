"""Generate the Phase 3 verification plots.

Produces verification artefacts under ``outputs/plots/`` covering:

- Per-network normal baseline pressure + flow time-series
  (FOWM, Jilin, Hanoi).
- Leak validation plots: pressure timeseries with onset markers,
  leak-demand profile, multi-leak heatmap, residual heatmap and time
  series, determinism check.
- ``validation_summary_phase3.txt``: per-config summary of severity,
  leak parameters and key metrics.

This script is run-once; it is not part of the package itself. It
re-runs every config end-to-end on each invocation so the plots are
always consistent with the current code rather than stale outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from wdn_pipeline.config import (
    PipelineConfig,
    SimulationConfig,
    load_config,
)
from wdn_pipeline.demand import apply_demand
from wdn_pipeline.faults.leak import LeakInjector
from wdn_pipeline.network import load_network
from wdn_pipeline.runner import RunSummary, run
from wdn_pipeline.simulation import SimulationResults, run_simulation

logging.basicConfig(level=logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PLOTS_DIR = REPO / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ScenarioRun:
    """Convenience bundle for plotting routines."""

    label: str
    config: PipelineConfig
    summary: RunSummary
    results: SimulationResults


def _select_columns(df: pd.DataFrame, k: int) -> list[str]:
    """Pick ``k`` evenly spaced columns by name for a representative slice."""

    cols = [c for c in df.columns if c != "label"]
    if len(cols) <= k:
        return cols
    step = max(1, len(cols) // k)
    return cols[::step][:k]


def _save(fig: Figure, name: str) -> Path:
    target = PLOTS_DIR / name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def _runtime_simulate(config_path: Path) -> ScenarioRun:
    """Run a config end-to-end and return both summary and raw results.

    The runner only returns serialisation metadata; we re-execute the
    simulation in-process so the plotting code works with live
    DataFrames rather than reloading parquet.
    """

    cfg = load_config(config_path)
    summary = run(cfg)
    wn = load_network(cfg.network, cfg.simulation)
    apply_demand(wn, cfg.demand, cfg.seed)
    if cfg.faults.leaks:
        rng = np.random.default_rng(cfg.seed)
        LeakInjector().apply(wn, list(cfg.faults.leaks), rng)
    results = run_simulation(wn)
    return ScenarioRun(label=config_path.stem, config=cfg, summary=summary, results=results)


def _normal_simulate_for_baseline(
    leak_config: PipelineConfig, sim_cfg_override: SimulationConfig | None = None
) -> SimulationResults:
    """Run the same scenario with the leaks stripped out, used as a baseline.

    Phase 3 residual plots compare ``baseline.pressure - leak.pressure``
    so the leak's effect can be isolated from the diurnal demand signal.
    """

    # Strip leaks but keep everything else (demand, seed, timesteps).
    base = leak_config.model_copy(
        update={
            "faults": leak_config.faults.model_copy(update={"leaks": []}),
            "scenario": leak_config.scenario.model_copy(update={"label": "baseline"}),
            "simulation": (
                sim_cfg_override
                if sim_cfg_override is not None
                else leak_config.simulation.model_copy(update={"demand_model": "DDA"})
            ),
            "validation": leak_config.validation,
        }
    )
    wn = load_network(base.network, base.simulation)
    apply_demand(wn, base.demand, base.seed)
    return run_simulation(wn)


# ---------------------------------------------------------------------------
# Per-network normal baselines
# ---------------------------------------------------------------------------


def plot_normal_baselines(scenarios: dict[str, ScenarioRun]) -> list[Path]:
    out: list[Path] = []
    for network, sc in scenarios.items():
        results = sc.results

        # Pressure
        nodes = _select_columns(results.pressure, 5)
        fig, ax = plt.subplots(figsize=(10, 5))
        for n in nodes:
            ax.plot(results.pressure.index / 3600, results.pressure[n], label=n)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Pressure (m)")
        ax.set_title(f"Normal pressure timeseries — {network}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        out.append(_save(fig, f"pressure_timeseries_{network}.png"))

        # Flow
        links = _select_columns(results.flowrate, 5)
        fig, ax = plt.subplots(figsize=(10, 5))
        for n in links:
            ax.plot(results.flowrate.index / 3600, results.flowrate[n], label=n)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Flowrate (m^3/s)")
        ax.set_title(f"Normal flowrate timeseries — {network}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        out.append(_save(fig, f"flow_timeseries_{network}.png"))
    return out


# ---------------------------------------------------------------------------
# Leak-specific plots
# ---------------------------------------------------------------------------


def plot_leak_pressure_drop_net3(net3_leak: ScenarioRun) -> Path:
    """Pressure at the leak node and 2-3 nearby nodes, baseline vs leak."""

    leak = net3_leak.summary.resolved_leaks[0]
    leak_results = net3_leak.results
    baseline_cfg_override = net3_leak.config.simulation.model_copy(update={"demand_model": "DDA"})
    baseline_results = _normal_simulate_for_baseline(net3_leak.config, baseline_cfg_override)

    pressure_leak = leak_results.pressure
    nearby = ["10", "20", "40", "50"]
    nearby = [n for n in nearby if n in pressure_leak.columns][:3]

    fig, axes = plt.subplots(len(nearby) + 1, 1, figsize=(10, 2.5 * (len(nearby) + 1)), sharex=True)
    if len(nearby) + 1 == 1:
        axes = [axes]

    leak_node = leak.leak_node_name
    axes[0].plot(
        pressure_leak.index / 3600,
        pressure_leak[leak_node],
        label=f"with leak ({leak_node})",
        color="tab:red",
    )
    axes[0].set_ylabel("Pressure (m)")
    axes[0].set_title(f"Leak node pressure — Net3 abrupt leak on pipe {leak.pipe}")
    axes[0].axvline(
        leak.start_time_seconds / 3600,
        color="black",
        linestyle="--",
        label=f"onset t={leak.start_time_seconds // 3600}h",
    )
    axes[0].axvline(
        leak.end_time_seconds / 3600,
        color="black",
        linestyle=":",
        label=f"end t={leak.end_time_seconds // 3600}h",
    )
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, node in enumerate(nearby, start=1):
        axes[i].plot(
            baseline_results.pressure.index / 3600,
            baseline_results.pressure[node],
            label=f"baseline ({node})",
            color="tab:blue",
            linestyle="--",
        )
        axes[i].plot(
            pressure_leak.index / 3600,
            pressure_leak[node],
            label=f"leak ({node})",
            color="tab:red",
        )
        axes[i].axvline(leak.start_time_seconds / 3600, color="black", linestyle="--")
        axes[i].axvline(leak.end_time_seconds / 3600, color="black", linestyle=":")
        axes[i].set_ylabel("Pressure (m)")
        axes[i].legend(loc="best", fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (hours)")
    return _save(fig, "leak_pressure_drop_net3.png")


def plot_leak_demand_profile_net3(net3_leak: ScenarioRun) -> Path:
    leak = net3_leak.summary.resolved_leaks[0]
    series = net3_leak.results.leak_demand[leak.leak_node_name]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index / 3600, series, label="leak_demand", color="tab:red")
    ax.axvline(leak.start_time_seconds / 3600, color="black", linestyle="--", label="onset")
    ax.axvline(leak.end_time_seconds / 3600, color="black", linestyle=":", label="end")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Leak demand (m^3/s)")
    ax.set_title(f"Abrupt leak demand profile — Net3 pipe {leak.pipe}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "leak_demand_profile_net3.png")


def plot_leak_incipient_profile_hanoi(hanoi_leak: ScenarioRun) -> Path:
    leak = hanoi_leak.summary.resolved_leaks[0]
    series = hanoi_leak.results.leak_demand[leak.leak_node_name]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index / 3600, series, label="leak_demand", marker="o", color="tab:red")
    ax.axvline(leak.start_time_seconds / 3600, color="black", linestyle="--", label="onset")
    ax.axvline(leak.end_time_seconds / 3600, color="black", linestyle=":", label="end")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Leak demand (m^3/s)")
    ax.set_title(f"Incipient (linear) leak — Hanoi pipe {leak.pipe}, {leak.profile_steps} steps")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "leak_incipient_profile_hanoi.png")


def plot_leak_pressure_heatmap_jilin(jilin_leak: ScenarioRun) -> Path:
    pressure = jilin_leak.results.pressure
    onsets = [leak.start_time_seconds for leak in jilin_leak.summary.resolved_leaks]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        pressure.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=(
            float(pressure.index[0]) / 3600,
            float(pressure.index[-1]) / 3600,
            -0.5,
            len(pressure.columns) - 0.5,
        ),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pressure (m)")
    ax.set_yticks(range(len(pressure.columns)))
    ax.set_yticklabels(pressure.columns, fontsize=6)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Node")
    ax.set_title("Multi-leak Jilin pressure heatmap")
    for i, t in enumerate(onsets):
        ax.axvline(
            t / 3600, color="white", linestyle="--", label=f"leak {i} onset" if i < 3 else None
        )
    if onsets:
        ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "leak_pressure_heatmap_jilin.png")


def plot_leak_multi_demand_jilin(jilin_leak: ScenarioRun) -> Path:
    leak_demand = jilin_leak.results.leak_demand
    fig, ax = plt.subplots(figsize=(10, 5))
    for leak in jilin_leak.summary.resolved_leaks:
        col = leak_demand[leak.leak_node_name]
        ax.plot(col.index / 3600, col, label=f"{leak.name or leak.leak_node_name} ({leak.profile})")
        ax.axvline(leak.start_time_seconds / 3600, color="grey", linestyle="--", alpha=0.4)
        ax.axvline(leak.end_time_seconds / 3600, color="grey", linestyle=":", alpha=0.4)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Leak demand (m^3/s)")
    ax.set_title("Multi-leak demand profiles — Jilin")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "leak_multi_demand_jilin.png")


def plot_leak_residual_heatmap_net3(net3_leak: ScenarioRun) -> Path:
    """Residual = baseline_pressure - leak_pressure across all nodes."""

    leak_results = net3_leak.results
    baseline_cfg = net3_leak.config.simulation.model_copy(update={"demand_model": "DDA"})
    baseline_results = _normal_simulate_for_baseline(net3_leak.config, baseline_cfg)

    leak_pressure = leak_results.pressure
    base_pressure = baseline_results.pressure
    common_cols = [c for c in leak_pressure.columns if c in base_pressure.columns]
    residual = base_pressure[common_cols] - leak_pressure[common_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    abs_max = float(np.abs(residual.to_numpy()).max())
    im = ax.imshow(
        residual.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
        extent=(
            float(residual.index[0]) / 3600,
            float(residual.index[-1]) / 3600,
            -0.5,
            len(common_cols) - 0.5,
        ),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("baseline - leak pressure (m)")
    ax.set_yticks(range(len(common_cols)))
    ax.set_yticklabels(common_cols, fontsize=4)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Node")
    ax.set_title("Pressure residual (baseline - leak) — Net3 abrupt leak")
    for leak in net3_leak.summary.resolved_leaks:
        ax.axvline(
            leak.start_time_seconds / 3600, color="black", linestyle="--", alpha=0.6, label="onset"
        )
        ax.axvline(
            leak.end_time_seconds / 3600, color="black", linestyle=":", alpha=0.6, label="end"
        )
    return _save(fig, "leak_residual_net3.png")


def plot_leak_residual_timeseries_net3(net3_leak: ScenarioRun) -> Path:
    leak_results = net3_leak.results
    baseline_cfg = net3_leak.config.simulation.model_copy(update={"demand_model": "DDA"})
    baseline_results = _normal_simulate_for_baseline(net3_leak.config, baseline_cfg)

    leak = net3_leak.summary.resolved_leaks[0]
    leak_pressure = leak_results.pressure
    base_pressure = baseline_results.pressure
    common_cols = [c for c in leak_pressure.columns if c in base_pressure.columns]
    residual = base_pressure[common_cols] - leak_pressure[common_cols]

    nearby = [c for c in ["10", "20", "40", "50"] if c in residual.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    for node in nearby:
        ax.plot(residual.index / 3600, residual[node], label=node)
    ax.axvline(
        leak.start_time_seconds / 3600,
        color="black",
        linestyle="--",
        label=f"onset (t={leak.start_time_seconds // 3600}h)",
    )
    ax.axvline(
        leak.end_time_seconds / 3600,
        color="black",
        linestyle=":",
        label=f"end (t={leak.end_time_seconds // 3600}h)",
    )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("baseline - leak pressure (m)")
    ax.set_title("Pressure residual time-series at nearby nodes — Net3")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "leak_residual_timeseries_net3.png")


def plot_leak_determinism_fowm(config_path: Path) -> Path:
    """Two identical runs of leak_random_fowm.yaml must agree to ~1e-12 m."""

    cfg = load_config(config_path)

    def _simulate_once() -> SimulationResults:
        wn = load_network(cfg.network, cfg.simulation)
        apply_demand(wn, cfg.demand, cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        LeakInjector().apply(wn, list(cfg.faults.leaks), rng)
        return run_simulation(wn)

    r1 = _simulate_once()
    r2 = _simulate_once()
    common_cols = [c for c in r1.pressure.columns if c in r2.pressure.columns]
    residual = r1.pressure[common_cols] - r2.pressure[common_cols]
    abs_max = float(np.abs(residual.to_numpy()).max())

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in residual.columns[:10]:
        ax.plot(residual.index / 3600, residual[col], label=col)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Pressure residual (m)")
    ax.set_title(
        f"Determinism check — leak_random_fowm.yaml\n"
        f"max |residual| = {abs_max:.3e} m (Newton-solver noise floor)"
    )
    ax.grid(True, alpha=0.3)
    return _save(fig, "leak_determinism_fowm.png")


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------


def write_validation_summary(scenarios: dict[str, ScenarioRun]) -> Path:
    target = PLOTS_DIR / "validation_summary_phase3.txt"
    lines: list[str] = []
    for label, sc in scenarios.items():
        lines.append("=" * 78)
        lines.append(f"Config: {label}")
        lines.append(f"  network: {sc.summary.network_name}")
        lines.append(f"  scenario_label: {sc.summary.scenario_label}")
        lines.append(f"  seed: {sc.summary.seed}")
        lines.append(f"  validation_severity: {sc.summary.validation.severity}")
        if sc.summary.resolved_leaks:
            for i, leak in enumerate(sc.summary.resolved_leaks):
                lines.append(
                    f"  leak[{i}]: pipe={leak.pipe} split={leak.split_fraction:.3f} "
                    f"area={leak.area_m2:.3e} m^2 profile={leak.profile} "
                    f"steps={leak.profile_steps} window=[{leak.start_time_seconds}, "
                    f"{leak.end_time_seconds}]s node={leak.leak_node_name}"
                )
        lines.append("  checks:")
        for check in sc.summary.validation.checks:
            lines.append(f"    [{check.severity:>7}] {check.name}: {check.detail}")
        # Surface key metrics for the report.
        p = sc.results.pressure.to_numpy()
        lines.append(f"  pressure_range: [{float(np.nanmin(p)):.3f}, {float(np.nanmax(p)):.3f}] m")
        lines.append("")
    target.write_text("\n".join(lines))
    return target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    plt.rcParams["figure.autolayout"] = True
    config_paths = {
        "normal_fowm": CONFIGS / "normal_fowm.yaml",
        "normal_jilin": CONFIGS / "normal_jilin.yaml",
        "normal_hanoi": CONFIGS / "normal_hanoi.yaml",
        "leak_abrupt_net3": CONFIGS / "leak_abrupt_net3.yaml",
        "leak_incipient_hanoi": CONFIGS / "leak_incipient_hanoi.yaml",
        "leak_multi_jilin": CONFIGS / "leak_multi_jilin.yaml",
        "leak_random_fowm": CONFIGS / "leak_random_fowm.yaml",
    }
    scenarios: dict[str, ScenarioRun] = {}
    for label, path in config_paths.items():
        print(f"Running {label}...")
        scenarios[label] = _runtime_simulate(path)

    artefacts: list[Path] = []
    print("Plotting normal baselines...")
    artefacts.extend(
        plot_normal_baselines(
            {
                "fowm": scenarios["normal_fowm"],
                "jilin": scenarios["normal_jilin"],
                "hanoi": scenarios["normal_hanoi"],
            }
        )
    )
    print("Plotting leak validation plots...")
    artefacts.append(plot_leak_pressure_drop_net3(scenarios["leak_abrupt_net3"]))
    artefacts.append(plot_leak_demand_profile_net3(scenarios["leak_abrupt_net3"]))
    artefacts.append(plot_leak_incipient_profile_hanoi(scenarios["leak_incipient_hanoi"]))
    artefacts.append(plot_leak_pressure_heatmap_jilin(scenarios["leak_multi_jilin"]))
    artefacts.append(plot_leak_multi_demand_jilin(scenarios["leak_multi_jilin"]))
    artefacts.append(plot_leak_residual_heatmap_net3(scenarios["leak_abrupt_net3"]))
    artefacts.append(plot_leak_residual_timeseries_net3(scenarios["leak_abrupt_net3"]))
    artefacts.append(plot_leak_determinism_fowm(config_paths["leak_random_fowm"]))

    print("Writing validation summary...")
    artefacts.append(write_validation_summary(scenarios))

    print(f"Wrote {len(artefacts)} artefacts to {PLOTS_DIR}:")
    for p in artefacts:
        print(f"  - {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
