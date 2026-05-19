"""Generate the Week 5 verification plots.

Produces verification artefacts under ``outputs/plots/`` covering the
Week 5 deliverables:

- ``pdd_vs_dda_pressure_net3.png``: pressure at Net3 node "10" under
  DDA and PDD. The two traces coincide, documenting that PDD does not
  resolve the known node-"10" negative-pressure artefact.
- ``sensor_gain_net3.png`` and ``sensor_fault_residual_gain_net3.png``:
  the multiplicative gain fault, clean vs corrupted and residual.
- ``cumulative_pressure_heatmap_net3.png``: a pressure heatmap for the
  cumulative leak+bias scenario with the leak onset and the sensor
  window marked.
- ``cumulative_demand_and_mask_net3.png``: the leak_demand trace and
  the sensor-fault mask on a shared timeline.
- ``cumulative_residual_jilin.png``: the pressure residual between the
  Jilin normal PDD baseline and the cumulative two-leak + dropout
  scenario.
- ``validation_summary_week5.txt``: per-config severity for every
  config in ``configs/``.

The script re-runs every config end to end so the plots are never
stale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from wdn_pipeline.config import (  # noqa: E402
    NetworkConfig,
    SimulationConfig,
    load_config,
)
from wdn_pipeline.demand import apply_demand  # noqa: E402
from wdn_pipeline.faults.leak import LeakInjector  # noqa: E402
from wdn_pipeline.faults.sensor import SensorFaultInjector  # noqa: E402
from wdn_pipeline.network import load_network  # noqa: E402
from wdn_pipeline.runner import run  # noqa: E402
from wdn_pipeline.simulation import run_simulation  # noqa: E402

logging.basicConfig(level=logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PLOTS_DIR = REPO / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: Figure, name: str) -> Path:
    target = PLOTS_DIR / name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def _run_scenario(config_name: str):
    """Run one config through the same stages the runner uses.

    Returns the corrupted :class:`SimulationResults`, the resolved
    leaks and the resolved sensor faults.
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
    if cfg.faults.sensor_faults:
        sf = SensorFaultInjector().apply(
            results, list(cfg.faults.sensor_faults), rng, wn
        )
        results = sf.results
        resolved_sensors = sf.resolved
    return results, resolved_leaks, resolved_sensors


# ---------------------------------------------------------------------------
# PDD vs DDA at Net3 node "10"
# ---------------------------------------------------------------------------


def _plot_pdd_vs_dda_net3() -> Path:
    common = dict(
        duration_seconds=24 * 3600,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )
    dda = run_simulation(
        load_network(
            NetworkConfig(inp_path="Net3"),
            SimulationConfig(demand_model="DDA", **common),
        )
    )
    pdd = run_simulation(
        load_network(
            NetworkConfig(inp_path="Net3"),
            SimulationConfig(demand_model="PDD", **common),
        )
    )
    t = dda.pressure.index.to_numpy() / 3600.0
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(t, dda.pressure["10"].to_numpy(), label="DDA", linewidth=2.2)
    ax.plot(
        t,
        pdd.pressure["10"].to_numpy(),
        label="PDD",
        linewidth=1.6,
        linestyle="--",
        color="C3",
    )
    ax.axhline(0.0, color="black", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("time [h]")
    ax.set_ylabel('pressure at Net3 node "10" [m]')
    ax.set_title(
        'PDD does not resolve the Net3 node "10" artefact '
        "(traces coincide)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, "pdd_vs_dda_pressure_net3.png")


# ---------------------------------------------------------------------------
# Gain sensor fault
# ---------------------------------------------------------------------------


def _plot_gain_net3() -> list[Path]:
    results, _, sensors = _run_scenario("sensor_gain_net3")
    fault = sensors[0]
    clean = results.pressure_clean[fault.target].to_numpy()
    corrupted = results.pressure[fault.target].to_numpy()
    t = results.pressure.index.to_numpy() / 3600.0
    s, e = fault.start_time_seconds / 3600.0, fault.end_time_seconds / 3600.0

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(t, clean, label="clean", linewidth=2.2)
    ax.plot(
        t,
        corrupted,
        label=f"corrupted (gain x{fault.gain_factor})",
        linewidth=1.6,
        linestyle="--",
        color="C3",
    )
    ax.axvspan(s, e, alpha=0.15, color="red")
    ax.set_xlabel("time [h]")
    ax.set_ylabel(f"pressure at node {fault.target} [m]")
    ax.set_title(f"Sensor gain fault on Net3 (node {fault.target})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = _save(fig, "sensor_gain_net3.png")

    residual = corrupted - clean
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(t, residual, color="C2", linewidth=1.6)
    ax.axvspan(s, e, alpha=0.15, color="red")
    ax.axhline(0.0, color="black", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("corrupted - clean [m]")
    ax.set_title(
        "Gain residual is proportional to the clean signal "
        f"((gain - 1) x y, node {fault.target})"
    )
    ax.grid(True, alpha=0.3)
    p2 = _save(fig, "sensor_fault_residual_gain_net3.png")
    return [p1, p2]


# ---------------------------------------------------------------------------
# Cumulative scenario plots
# ---------------------------------------------------------------------------


def _plot_cumulative_heatmap_net3() -> Path:
    results, leaks, sensors = _run_scenario("cumulative_leak_bias_net3")
    pressure = results.pressure
    arr = pressure.to_numpy().T  # rows = nodes, cols = time
    t = pressure.index.to_numpy() / 3600.0
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    im = ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, arr.shape[0]],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="pressure [m]")
    for leak in leaks:
        ax.axvline(
            leak.start_time_seconds / 3600.0,
            color="red",
            linewidth=2.0,
            label="leak onset",
        )
    for fault in sensors:
        ax.axvspan(
            fault.start_time_seconds / 3600.0,
            fault.end_time_seconds / 3600.0,
            alpha=0.18,
            color="white",
            label="sensor fault window",
        )
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lbl in zip(handles, labels, strict=False):
        seen.setdefault(lbl, h)
    ax.legend(seen.values(), seen.keys(), loc="upper right")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("node index")
    ax.set_title("Cumulative leak + bias on Net3: corrupted pressure heatmap")
    return _save(fig, "cumulative_pressure_heatmap_net3.png")


def _plot_cumulative_demand_and_mask_net3() -> Path:
    results, leaks, sensors = _run_scenario("cumulative_leak_bias_net3")
    leak = leaks[0]
    fault = sensors[0]
    t = results.leak_demand.index.to_numpy() / 3600.0
    leak_demand = results.leak_demand[leak.leak_node_name].to_numpy()
    times = results.pressure.index.to_numpy()
    mask = (times >= fault.start_time_seconds) & (
        times < fault.end_time_seconds
    )

    fig, ax1 = plt.subplots(figsize=(8.5, 4.0))
    ax1.plot(
        t, leak_demand, color="C0", linewidth=2.0, label="leak_demand"
    )
    ax1.set_xlabel("time [h]")
    ax1.set_ylabel("leak demand [m^3/s]", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(
        t, mask.astype(float), step="mid", alpha=0.25, color="C3"
    )
    ax2.set_ylabel(
        f"{fault.type} sensor mask on {fault.target}", color="C3"
    )
    ax2.set_ylim(-0.05, 1.4)
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title(
        "Cumulative leak + bias on Net3: leak_demand vs sensor-fault mask"
    )
    return _save(fig, "cumulative_demand_and_mask_net3.png")


def _plot_cumulative_residual_jilin() -> Path:
    baseline, _, _ = _run_scenario("normal_jilin_pdd")
    cumulative, _, _ = _run_scenario("cumulative_leak_dropout_jilin")
    # Compare on the columns present in both (the cumulative run adds
    # leak-node columns; the dropout corrupts pressure with NaN, so use
    # the clean signal for a meaningful residual).
    common = [
        c for c in baseline.pressure.columns if c in cumulative.pressure_clean.columns
    ]
    base = baseline.pressure[common].to_numpy()
    cumul = cumulative.pressure_clean[common].to_numpy()
    residual = (base - cumul).T
    t = baseline.pressure.index.to_numpy() / 3600.0
    vmax = float(np.nanmax(np.abs(residual))) or 1.0
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    im = ax.imshow(
        residual,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, residual.shape[0]],
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label="pressure residual [m]")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("node index")
    ax.set_title(
        "Jilin: normal PDD baseline minus cumulative two-leak + dropout"
    )
    return _save(fig, "cumulative_residual_jilin.png")


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------


def _format_check(name: str, severity: str, detail: str, max_len: int = 90) -> str:
    truncated = detail if len(detail) <= max_len else detail[: max_len - 3] + "..."
    return f"  [{severity:>7}] {name}: {truncated}"


def _write_validation_summary() -> Path:
    all_configs = sorted(p.stem for p in CONFIGS.glob("*.yaml"))
    lines: list[str] = ["Week 5 validation summary", "=" * 30, ""]
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
        lines.append(f"=== {cfg_name} ===")
        lines.append(f"  network         : {summary.network_name}")
        lines.append(f"  scenario_label  : {summary.scenario_label}")
        lines.append(
            f"  severity        : {summary.validation.severity.upper()}"
        )
        if summary.resolved_leaks:
            lines.append(
                f"  resolved_leaks  : {len(summary.resolved_leaks)}"
            )
        if summary.resolved_sensor_faults:
            lines.append(
                f"  resolved_sensors: {len(summary.resolved_sensor_faults)}"
            )
        if summary.interactions:
            lines.append(
                f"  interactions    : {len(summary.interactions)} "
                "(informational)"
            )
        for c in summary.validation.checks:
            lines.append(_format_check(c.name, c.severity, c.detail))
        lines.append("")
    summary_path = PLOTS_DIR / "validation_summary_week5.txt"
    summary_path.write_text("\n".join(lines))
    return summary_path


def main() -> None:
    written: list[Path] = []
    written.append(_plot_pdd_vs_dda_net3())
    written.extend(_plot_gain_net3())
    written.append(_plot_cumulative_heatmap_net3())
    written.append(_plot_cumulative_demand_and_mask_net3())
    written.append(_plot_cumulative_residual_jilin())
    written.append(_write_validation_summary())

    print("Wrote:")
    for p in written:
        print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
