"""Generate the definitive Phase 4 plot set.

Replaces the per-week scripts (Week 4 / Week 5) as the single source
of truth for the Phase 4 deliverable figures. Re-runs every config end
to end so plots are never stale.

Outputs under ``outputs/plots/``:

Normal scenarios
    - ``normal_pressure_comparison.png``  (2x2 grid, PDD normals)
    - ``pdd_vs_dda_net3.png``             (overlay at node "10")

Leak scenarios
    - ``leak_demand_profiles.png``        (abrupt / incipient / multi)
    - ``leak_residual_heatmap_net3.png``  (baseline minus leak pressures)

Sensor faults
    - ``sensor_faults_gallery.png``       (2x3 clean vs corrupted)
    - ``sensor_fault_residuals_gallery.png`` (2x3 residuals)

Cumulative scenarios
    - ``cumulative_heatmap_net3.png``
    - ``cumulative_residual_jilin.png``
    - ``cumulative_timeline_net3.png``

Cleanup & validation
    - ``leak_cleanup_columns.png``        (before/after column counts)
    - ``validation_summary_phase4.txt``   (severity table for every config)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from wdn_pipeline.config import (  # noqa: E402
    NetworkConfig,
    SimulationConfig,
    load_config,
)
from wdn_pipeline.demand import apply_demand  # noqa: E402
from wdn_pipeline.faults.leak import LeakInjector  # noqa: E402
from wdn_pipeline.faults.sensor import SensorFaultInjector  # noqa: E402
from wdn_pipeline.network import load_network  # noqa: E402
from wdn_pipeline.output import assemble_tables, build_basename  # noqa: E402
from wdn_pipeline.postprocess import remove_leak_artifacts  # noqa: E402
from wdn_pipeline.runner import run  # noqa: E402
from wdn_pipeline.simulation import run_simulation  # noqa: E402

logging.basicConfig(level=logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PLOTS_DIR = REPO / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Re-usable scenario runner
# ---------------------------------------------------------------------------


def _run_scenario(config_name: str):
    """Run one config through the same stages the runner uses.

    Returns ``(SimulationResults, resolved_leaks, resolved_sensor_faults,
    sensor_masks, cfg)``.
    """

    cfg = load_config(CONFIGS / f"{config_name}.yaml")
    wn = load_network(cfg.network, cfg.simulation)
    apply_demand(wn, cfg.demand, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    resolved_leaks = []
    if cfg.faults.leaks:
        resolved_leaks = LeakInjector().apply(wn, list(cfg.faults.leaks), rng)
    results = run_simulation(wn)
    resolved_sensors = []
    masks: dict[str, pd.Series] = {}
    if cfg.faults.sensor_faults:
        sf = SensorFaultInjector().apply(results, list(cfg.faults.sensor_faults), rng, wn)
        results = sf.results
        resolved_sensors = sf.resolved
        masks = sf.masks
    return results, resolved_leaks, resolved_sensors, masks, cfg


def _save(fig: Figure, name: str) -> Path:
    target = PLOTS_DIR / name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def _hours(index: pd.Index) -> np.ndarray:
    return np.asarray(index) / 3600.0


# ---------------------------------------------------------------------------
# Normal scenario plots
# ---------------------------------------------------------------------------


def _plot_normal_pressure_comparison() -> Path:
    """Pressure at 3 representative nodes for each of the four networks."""

    cases = [
        ("normal_net3_pdd", "Net3 (PDD)"),
        ("normal_hanoi_pdd", "Hanoi (PDD)"),
        ("normal_jilin_pdd", "Jilin (PDD)"),
        ("normal_fowm_pdd", "FOWM (PDD)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (config_name, label) in zip(axes.flat, cases, strict=False):
        results, *_ = _run_scenario(config_name)
        # Pick three representative nodes: take the first, middle and
        # last junction column that has a sensible (non-tank) pressure.
        cols = list(results.pressure.columns)
        picked = [cols[0], cols[len(cols) // 2], cols[-1]]
        hours = _hours(results.pressure.index)
        for col in picked:
            ax.plot(hours, results.pressure[col], label=f"node {col}")
        ax.set_title(label)
        ax.set_ylabel("pressure (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("hours")
    fig.suptitle("Normal PDD pressure traces — three representative nodes per network")
    fig.tight_layout()
    return _save(fig, "normal_pressure_comparison.png")


def _plot_pdd_vs_dda_net3() -> Path:
    """Pressure at Net3 node 10 under DDA and PDD — the dip is identical."""

    common = dict(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    sim_dda = SimulationConfig(**common, demand_model="DDA")
    sim_pdd = SimulationConfig(**common, demand_model="PDD")
    wn_dda = load_network(NetworkConfig(inp_path="Net3", name="net3"), sim_dda)
    wn_pdd = load_network(NetworkConfig(inp_path="Net3", name="net3"), sim_pdd)
    r_dda = run_simulation(wn_dda)
    r_pdd = run_simulation(wn_pdd)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    hours = _hours(r_dda.pressure.index)
    ax.plot(hours, r_dda.pressure["10"], label="DDA", linewidth=2)
    ax.plot(
        hours,
        r_pdd.pressure["10"],
        label="PDD",
        linewidth=2,
        linestyle="--",
    )
    ax.axhline(0, color="red", linestyle=":", alpha=0.5, label="0 m")
    ax.set_title(
        'Net3 node "10" — pressure dip is bit-identical under DDA and PDD\n'
        "(topology artefact, not a demand-model effect)"
    )
    ax.set_xlabel("hours")
    ax.set_ylabel("pressure (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, "pdd_vs_dda_net3.png")


# ---------------------------------------------------------------------------
# Leak plots
# ---------------------------------------------------------------------------


def _plot_leak_demand_profiles() -> Path:
    """Leak_demand profiles for the three Phase 3 demonstration leaks."""

    cases = [
        ("leak_abrupt_net3", "Net3 abrupt"),
        ("leak_incipient_hanoi", "Hanoi incipient (linear)"),
        ("leak_multi_jilin", "Jilin multi-leak"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (config_name, label) in zip(axes, cases, strict=False):
        results, leaks, *_ = _run_scenario(config_name)
        hours = _hours(results.leak_demand.index)
        for leak in leaks:
            col = leak.leak_node_name
            ax.plot(
                hours,
                results.leak_demand[col] * 1000.0,  # m^3/s -> L/s
                label=f"{col} ({leak.profile})",
            )
            ax.axvspan(
                leak.start_time_seconds / 3600.0,
                leak.end_time_seconds / 3600.0,
                color="orange",
                alpha=0.08,
            )
        ax.set_title(label)
        ax.set_ylabel("leak (L/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("hours")
    fig.suptitle("Leak demand profiles — abrupt vs incipient vs multi-leak")
    fig.tight_layout()
    return _save(fig, "leak_demand_profiles.png")


def _plot_leak_residual_heatmap_net3() -> Path:
    """(normal_pdd - leak_abrupt) pressure residual heatmap for Net3."""

    res_base, *_ = _run_scenario("normal_net3_pdd")
    res_leak, leaks, *_ = _run_scenario("leak_abrupt_net3")
    common_cols = sorted(set(res_base.pressure.columns) & set(res_leak.pressure.columns))
    diff = res_base.pressure[common_cols] - res_leak.pressure[common_cols]
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        diff.T.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.abs(diff.to_numpy()).max(),
        vmax=np.abs(diff.to_numpy()).max(),
    )
    ax.set_yticks(range(len(common_cols)))
    ax.set_yticklabels(common_cols, fontsize=6)
    n_steps = diff.shape[0]
    ax.set_xticks(np.linspace(0, n_steps - 1, 9, dtype=int))
    ax.set_xticklabels([f"{h:.0f}" for h in np.linspace(0, 24, 9)])
    ax.set_xlabel("hours")
    ax.set_ylabel("node")
    for leak in leaks:
        ax.axvline(
            leak.start_time_seconds / 3600.0,
            color="black",
            linestyle="--",
            linewidth=1,
        )
        ax.axvline(
            leak.end_time_seconds / 3600.0,
            color="black",
            linestyle="--",
            linewidth=1,
        )
    plt.colorbar(im, ax=ax, label="pressure residual (m)")
    ax.set_title(
        "Net3 leak residual heatmap: normal_pdd minus leak_abrupt pressures\n"
        "(positive = leak depresses pressure; vertical dashes mark leak window)"
    )
    fig.tight_layout()
    return _save(fig, "leak_residual_heatmap_net3.png")


# ---------------------------------------------------------------------------
# Sensor fault galleries
# ---------------------------------------------------------------------------


_SENSOR_GRID = [
    ("sensor_bias_net3", "bias"),
    ("sensor_drift_net3", "drift"),
    ("sensor_stuck_net3", "stuck"),
    ("sensor_dropout_net3", "dropout"),
    ("sensor_noise_net3", "noise"),
    ("sensor_gain_net3", "gain"),
]


def _plot_sensor_faults_gallery() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (config_name, fault_type) in zip(axes.flat, _SENSOR_GRID, strict=False):
        results, _, sensors, *_ = _run_scenario(config_name)
        sensor = sensors[0]
        clean = results.pressure_clean[sensor.target]
        corrupted = results.pressure[sensor.target]
        hours = _hours(results.pressure.index)
        ax.plot(hours, clean, label="clean", linewidth=2, color="steelblue")
        ax.plot(
            hours,
            corrupted,
            label="corrupted",
            linewidth=1.2,
            color="crimson",
            linestyle="--",
        )
        ax.axvspan(
            sensor.start_time_seconds / 3600.0,
            sensor.end_time_seconds / 3600.0,
            color="orange",
            alpha=0.1,
        )
        ax.set_title(f"{fault_type} @ {sensor.target}")
        ax.set_ylabel("pressure (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("hours")
    fig.suptitle("Sensor fault gallery — clean vs corrupted (Net3, pressure @ junction 15)")
    fig.tight_layout()
    return _save(fig, "sensor_faults_gallery.png")


def _plot_sensor_fault_residuals_gallery() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (config_name, fault_type) in zip(axes.flat, _SENSOR_GRID, strict=False):
        results, _, sensors, *_ = _run_scenario(config_name)
        sensor = sensors[0]
        clean = results.pressure_clean[sensor.target]
        corrupted = results.pressure[sensor.target]
        residual = corrupted - clean
        hours = _hours(results.pressure.index)
        ax.plot(hours, residual, color="crimson", linewidth=1.5)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.6)
        ax.axvspan(
            sensor.start_time_seconds / 3600.0,
            sensor.end_time_seconds / 3600.0,
            color="orange",
            alpha=0.08,
        )
        ax.set_title(f"{fault_type}: corrupted − clean")
        ax.set_ylabel("residual (m)")
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("hours")
    fig.suptitle("Sensor fault residuals — each subplot is the exact signature of its fault type")
    fig.tight_layout()
    return _save(fig, "sensor_fault_residuals_gallery.png")


# ---------------------------------------------------------------------------
# Cumulative scenarios
# ---------------------------------------------------------------------------


def _plot_cumulative_heatmap_net3() -> Path:
    """Pressure heatmap for cumulative_leak_bias_net3 with annotations."""

    results, leaks, sensors, *_ = _run_scenario("cumulative_leak_bias_net3")
    fig, ax = plt.subplots(figsize=(12, 7))
    cols = list(results.pressure.columns)
    arr = results.pressure.to_numpy()
    im = ax.imshow(arr.T, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=6)
    n_steps = arr.shape[0]
    ax.set_xticks(np.linspace(0, n_steps - 1, 9, dtype=int))
    ax.set_xticklabels([f"{h:.0f}" for h in np.linspace(0, 24, 9)])
    ax.set_xlabel("hours")
    ax.set_ylabel("node")
    for leak in leaks:
        ax.axvline(
            leak.start_time_seconds / 3600.0,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="leak",
        )
        ax.axvline(
            leak.end_time_seconds / 3600.0,
            color="red",
            linestyle="--",
            linewidth=1.2,
        )
    for sensor in sensors:
        ax.axvline(
            sensor.start_time_seconds / 3600.0,
            color="white",
            linestyle=":",
            linewidth=1.2,
            label="sensor",
        )
        ax.axvline(
            sensor.end_time_seconds / 3600.0,
            color="white",
            linestyle=":",
            linewidth=1.2,
        )
    plt.colorbar(im, ax=ax, label="pressure (m)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax.set_title("Net3 cumulative scenario (leak + bias sensor) — corrupted pressures over time")
    fig.tight_layout()
    return _save(fig, "cumulative_heatmap_net3.png")


def _plot_cumulative_residual_jilin() -> Path:
    """Residual heatmap: normal_jilin_pdd minus cumulative_leak_dropout_jilin (clean signal)."""

    res_base, *_ = _run_scenario("normal_jilin_pdd")
    res_cum, leaks, *_ = _run_scenario("cumulative_leak_dropout_jilin")
    # Use the clean signal so the dropout NaNs don't dominate.
    base_p = res_base.pressure
    cum_p = res_cum.pressure_clean
    common = sorted(set(base_p.columns) & set(cum_p.columns))
    diff = base_p[common] - cum_p[common]
    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = np.nanmax(np.abs(diff.to_numpy()))
    im = ax.imshow(diff.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(common)))
    ax.set_yticklabels(common, fontsize=5)
    n_steps = diff.shape[0]
    ax.set_xticks(np.linspace(0, n_steps - 1, 9, dtype=int))
    ax.set_xticklabels([f"{h:.0f}" for h in np.linspace(0, 24, 9)])
    ax.set_xlabel("hours")
    ax.set_ylabel("node")
    for leak in leaks:
        ax.axvline(leak.start_time_seconds / 3600.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(leak.end_time_seconds / 3600.0, color="black", linestyle="--", linewidth=1)
    plt.colorbar(im, ax=ax, label="pressure residual (m)")
    ax.set_title(
        "Jilin cumulative residual: normal_pdd − cumulative_leak_dropout (clean pressure)\n"
        "(dashes mark the two leak windows)"
    )
    fig.tight_layout()
    return _save(fig, "cumulative_residual_jilin.png")


def _plot_cumulative_timeline_net3() -> Path:
    """Gantt-style timeline of fault windows for the cumulative_leak_bias config."""

    _, leaks, sensors, *_ = _run_scenario("cumulative_leak_bias_net3")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    row = 0
    rows: list[str] = []
    for leak in leaks:
        ax.add_patch(
            Rectangle(
                (leak.start_time_seconds / 3600.0, row - 0.4),
                (leak.end_time_seconds - leak.start_time_seconds) / 3600.0,
                0.8,
                color="firebrick",
                alpha=0.7,
            )
        )
        rows.append(f"leak: {leak.leak_node_name} ({leak.profile})")
        row += 1
    for sensor in sensors:
        ax.add_patch(
            Rectangle(
                (sensor.start_time_seconds / 3600.0, row - 0.4),
                (sensor.end_time_seconds - sensor.start_time_seconds) / 3600.0,
                0.8,
                color="steelblue",
                alpha=0.7,
            )
        )
        rows.append(f"sensor: {sensor.type} @ {sensor.target}")
        row += 1
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlim(0, 24)
    ax.set_xlabel("hours")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Cumulative scenario timeline — overlapping leak and sensor-fault windows (Net3)")
    fig.tight_layout()
    return _save(fig, "cumulative_timeline_net3.png")


# ---------------------------------------------------------------------------
# Leak cleanup illustration
# ---------------------------------------------------------------------------


def _columns_before_after(config_name: str) -> tuple[int, int]:
    """Return (before, after) numeric column counts for the pressure table."""

    cfg = load_config(CONFIGS / f"{config_name}.yaml")
    wn = load_network(cfg.network, cfg.simulation)
    apply_demand(wn, cfg.demand, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    resolved_leaks = LeakInjector().apply(wn, list(cfg.faults.leaks), rng)
    results = run_simulation(wn)
    from wdn_pipeline.labelling import build_labels  # local import keeps module-level imports tidy

    labels = build_labels(
        cfg,
        results.pressure.index,
        cfg.network.name or "net3",
        resolved_leaks,
        [],
    )
    tables = assemble_tables(results, labels, sensor_masks={})
    pressure_before = len(tables["pressure"].columns)
    flowrate_before = len(tables["flowrate"].columns)
    cleaned = remove_leak_artifacts(tables, resolved_leaks)
    pressure_after = len(cleaned["pressure"].columns)
    flowrate_after = len(cleaned["flowrate"].columns)
    return (
        pressure_before + flowrate_before,
        pressure_after + flowrate_after,
    )


def _plot_leak_cleanup_columns() -> Path:
    """Bar chart comparing column counts before/after leak-node cleanup."""

    before, after = _columns_before_after("leak_abrupt_net3")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        ["with split artefacts", "after cleanup"],
        [before, after],
        color=["lightcoral", "mediumseagreen"],
        edgecolor="black",
    )
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("numeric columns (pressure + flowrate)")
    ax.set_title(
        "Leak-node cleanup: leak node and split-segment pipe collapsed\n"
        "back into the original network schema (Net3, leak_abrupt scenario)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "leak_cleanup_columns.png")


# ---------------------------------------------------------------------------
# Validation summary text
# ---------------------------------------------------------------------------


_TYPE_GROUPS = (
    ("normal", lambda r: not r.resolved_leaks and not r.resolved_sensor_faults),
    ("leak", lambda r: r.resolved_leaks and not r.resolved_sensor_faults),
    ("sensor_fault", lambda r: not r.resolved_leaks and r.resolved_sensor_faults),
    ("cumulative", lambda r: r.resolved_leaks and r.resolved_sensor_faults),
)


def _write_validation_summary() -> Path:
    """Run every config and record severities per check."""

    target = PLOTS_DIR / "validation_summary_phase4.txt"
    rows: list[tuple[str, str, str, str, list[tuple[str, str, str]]]] = []
    configs = sorted(CONFIGS.glob("*.yaml"))
    for cfg_path in configs:
        cfg = load_config(cfg_path)
        summary = run(cfg)
        # Pick a scenario group based on resolved-faults.
        for group, predicate in _TYPE_GROUPS:
            if predicate(summary):
                grp = group
                break
        else:
            grp = "unknown"
        checks = [(c.name, c.severity, c.detail) for c in summary.validation.checks]
        rows.append(
            (
                cfg_path.stem,
                grp,
                summary.validation.severity.upper(),
                build_basename(summary.network_name, summary.scenario_label, summary.seed),
                checks,
            )
        )
    lines: list[str] = []
    lines.append("# Phase 4 validation summary (every config under configs/)")
    lines.append("")
    for grp_name, _ in _TYPE_GROUPS:
        grp_rows = [r for r in rows if r[1] == grp_name]
        if not grp_rows:
            continue
        lines.append(f"## {grp_name} scenarios")
        lines.append("")
        for stem, _, severity, basename, checks in grp_rows:
            lines.append(f"### {stem}  [{severity}]")
            lines.append(f"  basename: {basename}")
            for check_name, check_severity, detail in checks:
                lines.append(f"  [{check_severity:>7}] {check_name}: {detail}")
            lines.append("")
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    artefacts: list[Path] = []
    print("Normal plots...")
    artefacts.append(_plot_normal_pressure_comparison())
    artefacts.append(_plot_pdd_vs_dda_net3())
    print("Leak plots...")
    artefacts.append(_plot_leak_demand_profiles())
    artefacts.append(_plot_leak_residual_heatmap_net3())
    print("Sensor fault plots...")
    artefacts.append(_plot_sensor_faults_gallery())
    artefacts.append(_plot_sensor_fault_residuals_gallery())
    print("Cumulative plots...")
    artefacts.append(_plot_cumulative_heatmap_net3())
    artefacts.append(_plot_cumulative_residual_jilin())
    artefacts.append(_plot_cumulative_timeline_net3())
    print("Cleanup illustration...")
    artefacts.append(_plot_leak_cleanup_columns())
    print("Validation summary...")
    artefacts.append(_write_validation_summary())
    print("Done. Wrote:")
    for path in artefacts:
        print(f"  {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
