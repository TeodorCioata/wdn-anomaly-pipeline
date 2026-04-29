"""Generate the Phase 2 verification plots.

Produces five artefacts under ``outputs/plots/``:

1. ``pressure_timeseries_net3.png``: pressure at five representative
   nodes over 24 h.
2. ``flow_timeseries_net3.png``: flow at five representative pipes.
3. ``pressure_heatmap_net3.png``: every node's pressure over time as a
   colour matrix; gives a quick visual overview of the network.
4. ``determinism_residual_net3.png``: pressure residual (m) between two
   pipeline runs on identical config; should be at the numerical noise
   floor (<1e-12 m).
5. ``validation_summary.txt``: human-readable validation report for
   both example configs.

This script is run-once; it is not part of the package itself.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wdn_pipeline.config import load_config
from wdn_pipeline.demand import apply_demand
from wdn_pipeline.network import load_network
from wdn_pipeline.runner import run
from wdn_pipeline.simulation import run_simulation
from wdn_pipeline.validation import residuals, validate_normal_scenario

logging.basicConfig(level=logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _select_columns(df: pd.DataFrame, k: int) -> list[str]:
    """Pick ``k`` evenly spaced columns by name for a representative slice."""

    cols = list(df.columns)
    if len(cols) <= k:
        return cols
    step = len(cols) // k
    return cols[::step][:k]


def _simulate(config_path: Path):
    cfg = load_config(config_path)
    wn = load_network(cfg.network, cfg.simulation)
    apply_demand(wn, cfg.demand, cfg.seed)
    results = run_simulation(wn)
    report = validate_normal_scenario(wn, results, cfg.validation)
    return cfg, wn, results, report


def plot_pressure_timeseries(results, out: Path) -> None:
    nodes = _select_columns(results.pressure, 5)
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in nodes:
        ax.plot(results.pressure.index / 3600.0, results.pressure[n], label=f"node {n}")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("pressure (m)")
    ax.set_title("Net3 — pressure at five representative nodes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_flow_timeseries(results, out: Path) -> None:
    pipes = _select_columns(results.flowrate, 5)
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in pipes:
        ax.plot(results.flowrate.index / 3600.0, results.flowrate[p], label=f"link {p}")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("flow (m^3/s)")
    ax.set_title("Net3 — flow at five representative links")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_pressure_heatmap(results, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    arr = results.pressure.to_numpy().T  # nodes x time
    im = ax.imshow(arr, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("timestep index (1 h)")
    ax.set_ylabel("node index")
    ax.set_title(f"Net3 — pressure heatmap ({arr.shape[0]} nodes × {arr.shape[1]} steps)")
    fig.colorbar(im, ax=ax, label="pressure (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_determinism_residual(config_path: Path, out: Path) -> float:
    """Run the pipeline twice on the same config and plot the difference."""

    cfg = load_config(config_path)
    cfg2 = cfg.model_copy(
        update={
            "output": cfg.output.model_copy(
                update={"directory": REPO / "outputs" / "_determinism"}
            )
        }
    )
    s1 = run(cfg)
    s2 = run(cfg2)
    p1 = pd.read_csv(
        next(p for p in s1.output_paths if p.name.endswith("pressure.csv")),
        index_col="time_seconds",
    ).drop(columns=["label"])
    p2 = pd.read_csv(
        next(p for p in s2.output_paths if p.name.endswith("pressure.csv")),
        index_col="time_seconds",
    ).drop(columns=["label"])
    diff = residuals(p1, p2)
    max_abs = float(np.abs(diff.to_numpy()).max())

    fig, ax = plt.subplots(figsize=(10, 5))
    arr = np.abs(diff.to_numpy())
    im = ax.imshow(arr.T, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("timestep index")
    ax.set_ylabel("node index")
    ax.set_title(
        f"Determinism check — |pressure_run1 - pressure_run2|, max = {max_abs:.2e} m"
    )
    fig.colorbar(im, ax=ax, label="abs residual (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return max_abs


def write_validation_summary(out: Path) -> None:
    lines: list[str] = []
    for name in ("normal_net3.yaml", "normal_hanoi.yaml"):
        cfg_path = REPO / "configs" / name
        cfg, _, _, report = _simulate(cfg_path)
        lines.append(f"=== {name} ===")
        lines.append(f"network        : {cfg.network.inp_path}")
        lines.append(f"demand_mode    : {cfg.demand.mode}")
        lines.append(f"seed           : {cfg.seed}")
        lines.append(report.format())
        lines.append("")
    out.write_text("\n".join(lines))


def main() -> None:
    net3_cfg_path = REPO / "configs" / "normal_net3.yaml"
    cfg, wn, results, report = _simulate(net3_cfg_path)

    plot_pressure_timeseries(results, PLOTS_DIR / "pressure_timeseries_net3.png")
    plot_flow_timeseries(results, PLOTS_DIR / "flow_timeseries_net3.png")
    plot_pressure_heatmap(results, PLOTS_DIR / "pressure_heatmap_net3.png")
    max_abs = plot_determinism_residual(
        net3_cfg_path, PLOTS_DIR / "determinism_residual_net3.png"
    )
    write_validation_summary(PLOTS_DIR / "validation_summary.txt")

    print(f"Plots written to {PLOTS_DIR}")
    print(f"Net3 validation severity: {report.severity.upper()}")
    print(f"Determinism max abs pressure residual: {max_abs:.3e} m")
    if max_abs > 1e-9:
        print("WARN: determinism residual unexpectedly large.", file=sys.stderr)


if __name__ == "__main__":
    main()
