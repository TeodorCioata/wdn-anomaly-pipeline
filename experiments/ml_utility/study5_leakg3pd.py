"""Study 5 - cross-dataset sanity check vs LeakG3PD.

LeakG3PD (Pilotto et al., IDEAL 2024) generates leak scenarios by inserting a
new node at a random point along a random pipe via ``wntr.morph.split_pipe`` and
attaching a WNTR leak whose demand follows the WNTR leak-model equation
``Q = Cd * A * sqrt(2 * g * h)`` (its headline improvement over LeakDB is exactly
that "leak demands track pressure values according to WNTR leak model
equation"). It sizes the leak hole as a fraction of the host pipe's diameter:
*small* leaks in ``[d/25, d/5]`` and *big* leaks in ``[d/5, d]``.

This study checks that, where our generator and LeakG3PD overlap (leaks on the
shared Hanoi network), our pipeline reproduces the same accepted hydraulic
behaviour. It is framed as an external-validity sanity check, **not** a
competition or a superiority claim.

LeakG3PD's generator ships against an older WNTR API and bundled ``.mat`` /
pickle demand assets, so rather than re-run it in-environment we compare our
matched-Hanoi leaks against its **documented numerical expectation**: the WNTR
leak law it states it follows, and its small/big diameter classes. We generate
matched leaks with our own pipeline (explicit pipe, ``Cd = 0.75``, diameters
across LeakG3PD's classes), then verify:

1. our simulated leak outflow ``Q_sim`` tracks the analytic law
   ``Q = Cd * A * sqrt(2 g h)`` across the diurnal head range (agreement in
   behaviour with the model LeakG3PD reports);
2. the near-node pressure drop grows with leak size - the qualitative leak
   signature.

Outputs ``data/study5.json`` and ``figures/study5_leakg3pd.png``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr

from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.output import build_basename
from wdn_pipeline.runner import run

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
SCN_DIR = DATA_DIR / "study5_scenarios"
HANOI_INP = "networks/Hanoi.inp"

G = 9.81
CD = 0.75
SEED = 7
DURATION = 86400
TIMESTEP = 3600
LEAK_START = 3600
LEAK_END = 82800
# Leak diameters as fractions of the host pipe diameter, spanning LeakG3PD's
# small ([d/25, d/5]) and big ([d/5, d]) classes.
DIAMETER_FRACTIONS = [1 / 25, 1 / 15, 1 / 10, 1 / 5, 1 / 3, 1 / 2, 1 / 1.5]


def _pick_pipe() -> tuple[str, float, list[str]]:
    """Pick a representative (median-diameter) Hanoi pipe and its end junctions."""

    wn = wntr.network.WaterNetworkModel(HANOI_INP)
    pipes = [(n, wn.get_link(n)) for n in wn.pipe_name_list]
    pipes.sort(key=lambda p: p[1].diameter)
    name, link = pipes[len(pipes) // 2]
    end_nodes = [link.start_node_name, link.end_node_name]
    return name, float(link.diameter), end_nodes


def _config(pipe: str, diameter_m: float, frac: float) -> tuple[dict, str]:
    label = f"leakg3pd_match_f{frac:.3f}"
    cfg = {
        "network": {"inp_path": HANOI_INP, "name": "hanoi_s5"},
        "simulation": {
            "duration_seconds": DURATION,
            "hydraulic_timestep_seconds": TIMESTEP,
            "report_timestep_seconds": TIMESTEP,
            "pattern_timestep_seconds": TIMESTEP,
            "demand_model": "PDD",
        },
        "seed": SEED,
        "demand": {"mode": "fourier"},
        "scenario": {"type": "leak", "label": label},
        "faults": {
            "leaks": [
                {
                    "pipe": pipe,
                    "split_fraction": 0.5,
                    "diameter_m": diameter_m,
                    "discharge_coeff": CD,
                    "start_time_seconds": LEAK_START,
                    "end_time_seconds": LEAK_END,
                    "profile": "abrupt",
                }
            ]
        },
        "validation": {"pressure_min_warning_tolerance_m": 5.0},
        "output": {"directory": str(SCN_DIR), "formats": ["parquet"]},
    }
    return cfg, label


def _read_table(basename: str, table: str) -> pd.DataFrame:
    df = pd.read_parquet(SCN_DIR / f"{basename}_{table}.parquet")
    return df.set_index("time_seconds")


def main() -> None:
    SCN_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pipe, pipe_d, end_nodes = _pick_pipe()
    print(f"Matched network: Hanoi | pipe {pipe} (d={pipe_d:.3f} m) | ends {end_nodes}")

    records: list[dict] = []
    scatter_h: list[np.ndarray] = []
    scatter_q: list[np.ndarray] = []
    scatter_area: list[float] = []

    for frac in DIAMETER_FRACTIONS:
        diameter = pipe_d * frac
        area = np.pi * (diameter / 2.0) ** 2
        leak_class = "small" if frac <= 1 / 5 else "big"
        cfg, label = _config(pipe, diameter, frac)
        summary = run(PipelineConfig.model_validate(cfg))
        leak_node = summary.resolved_leaks[0].leak_node_name
        basename = build_basename("hanoi_s5", label, SEED)

        leak_demand = _read_table(basename, "leak_demand")
        pressure = _read_table(basename, "pressure")
        in_window = (leak_demand.index >= LEAK_START) & (leak_demand.index < LEAK_END)

        q_sim = leak_demand.loc[in_window, leak_node].to_numpy(dtype=float)
        h = pressure.loc[in_window, leak_node].to_numpy(dtype=float)
        h_pos = np.clip(h, 0.0, None)
        q_theory = CD * area * np.sqrt(2.0 * G * h_pos)

        active = q_sim > 1e-9
        rel_err = (
            float(np.mean(np.abs(q_sim[active] - q_theory[active]) / q_theory[active]))
            if active.any()
            else float("nan")
        )

        # Near-node pressure drop: mean pressure before vs during the leak, at
        # the host pipe's end junctions.
        before = pressure.index < LEAK_START
        near = [n for n in end_nodes if n in pressure.columns]
        if before.any() and near:
            p_before = pressure.loc[before, near].to_numpy().mean()
            p_during = pressure.loc[in_window, near].to_numpy().mean()
            drop = float(p_before - p_during)
        else:
            drop = float("nan")

        records.append(
            {
                "diameter_fraction": frac,
                "leak_class": leak_class,
                "diameter_m": diameter,
                "area_m2": area,
                "mean_q_sim_m3s": float(np.mean(q_sim)),
                "mean_q_theory_m3s": float(np.mean(q_theory)),
                "leak_law_relative_error": rel_err,
                "near_node_pressure_drop_m": drop,
            }
        )
        scatter_h.append(h_pos)
        scatter_q.append(q_sim)
        scatter_area.append(area)
        print(
            f"  f={frac:.3f} ({leak_class:5s}) area={area:.2e} m^2  "
            f"mean Q={np.mean(q_sim):.4e}  leak-law rel.err={rel_err:.2e}  "
            f"near-node drop={drop:.3f} m"
        )

    worst_err = float(np.nanmax([r["leak_law_relative_error"] for r in records]))
    payload = {
        "matched_network": "Hanoi",
        "host_pipe": pipe,
        "host_pipe_diameter_m": pipe_d,
        "discharge_coeff": CD,
        "leakg3pd_model": "Q = Cd * A * sqrt(2 g h) (WNTR leak law, per LeakG3PD)",
        "leakg3pd_size_classes": {"small": "[d/25, d/5]", "big": "[d/5, d]"},
        "ran_leakg3pd_generator": False,
        "fallback_note": (
            "LeakG3PD's generator targets an older WNTR API and bundled demand "
            "assets; per its documentation we compare against the WNTR leak law "
            "it states it follows and its small/big diameter classes."
        ),
        "worst_leak_law_relative_error": worst_err,
        "scenarios": records,
    }
    (DATA_DIR / "study5.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    h_grid = np.linspace(0.1, max(float(np.max(np.concatenate(scatter_h))), 1.0), 100)
    cmap = plt.get_cmap("viridis")
    for i, (h_arr, q_arr, area) in enumerate(zip(scatter_h, scatter_q, scatter_area, strict=True)):
        color = cmap(i / max(1, len(scatter_area) - 1))
        axes[0].scatter(h_arr, q_arr, s=18, color=color, alpha=0.7, label=f"A={area:.1e} m^2")
        axes[0].plot(h_grid, CD * area * np.sqrt(2 * G * h_grid), color=color, lw=1.0, alpha=0.8)
    axes[0].set_xlabel("gauge head h at leak node (m)")
    axes[0].set_ylabel("leak outflow Q (m^3/s)")
    axes[0].set_title("Leak outflow vs head\npoints = our simulator, lines = WNTR/LeakG3PD law")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    fracs = [r["diameter_fraction"] for r in records]
    drops = [r["near_node_pressure_drop_m"] for r in records]
    colors = ["tab:blue" if r["leak_class"] == "small" else "tab:red" for r in records]
    axes[1].bar(range(len(fracs)), drops, color=colors)
    axes[1].set_xticks(range(len(fracs)))
    axes[1].set_xticklabels([f"{f:.3f}" for f in fracs], rotation=45)
    axes[1].set_xlabel("leak diameter / pipe diameter")
    axes[1].set_ylabel("near-node pressure drop (m)")
    axes[1].set_title("Leak signature: near-node pressure drop\n(blue=small class, red=big class)")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.suptitle("Study 5: matched-Hanoi leak behaviour vs LeakG3PD's documented model")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study5_leakg3pd.png", dpi=130)
    plt.close(fig)

    print(f"\nWorst leak-law relative error across scenarios: {worst_err:.2e}")
    print(f"Wrote {DATA_DIR / 'study5.json'} and {FIG_DIR / 'study5_leakg3pd.png'}")


if __name__ == "__main__":
    main()
