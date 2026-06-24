"""Refresh the paper's supporting studies (1-3) from one consistent corpus.

So every number in the paper comes from the same machine and software stack,
this regenerates:

- **Study 1 (physics validation):** mass-balance and Hazen-Williams energy
  residual distributions across the experiment corpus, harvested from the
  experiment batch summary (no re-simulation needed - the pipeline already runs
  both checks on every scenario). Produces a box plot and a worst-case table,
  and reports the residual floor.
- **Study 2 (network sweep):** regenerates ``docs/network_sweep_report.md`` by
  rebuilding the sweep configs, running them and re-rendering the report.
- **Study 3 (parallel scaling):** re-runs the parallel benchmark, regenerating
  ``docs/parallelisation_benchmark.md`` and the speedup figure.

Studies 2 and 3 drive the existing pipeline scripts (they are the source of
truth); this module only orchestrates and captures their headline numbers into
``data/study{1,2,3}.json`` for the consolidated report.

Usage::

    python experiments/ml_utility/refresh_supporting_studies.py            # all three
    python experiments/ml_utility/refresh_supporting_studies.py --only 1   # just Study 1
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wdn_pipeline.batch import run_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
EXP_BATCH_SUMMARY = DATA_DIR / "batch_report" / "batch_summary.json"
PY = sys.executable

_NUM = r"([0-9.eE+-]+)"
_MASS_RE = re.compile(r"net_inflow - demand - leak\|\s*=\s*" + _NUM)
_HEADLOSS_RE = re.compile(r"dh_obs - dh_pred\|\s*=\s*" + _NUM)


# -- Study 1: physics residuals -------------------------------------------


def _residuals_from_summary(summary_path: Path) -> dict[str, list[float]]:
    summary = json.loads(summary_path.read_text())
    runs = summary.get("runs", [])
    mass: list[float] = []
    headloss: list[float] = []
    for run in runs:
        checks = run.get("validation", {}).get("checks", [])
        for c in checks:
            detail = c.get("detail", "")
            if c["name"] == "mass_balance":
                m = _MASS_RE.search(detail)
                if m:
                    mass.append(float(m.group(1)))
            elif c["name"] == "hazen_williams_headloss":
                m = _HEADLOSS_RE.search(detail)
                if m:
                    headloss.append(float(m.group(1)))
    return {"mass_balance_m3s": mass, "hazen_williams_m": headloss}


def study1() -> dict:
    if not EXP_BATCH_SUMMARY.exists():
        raise SystemExit(
            f"{EXP_BATCH_SUMMARY} not found. Run generate_experiment_dataset.py first."
        )
    res = _residuals_from_summary(EXP_BATCH_SUMMARY)
    out: dict = {"n_scenarios": 0, "metrics": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (key, label, unit) in zip(
        axes,
        [
            ("mass_balance_m3s", "mass-balance residual", "m^3/s"),
            ("hazen_williams_m", "Hazen-Williams energy residual", "m"),
        ],
        strict=True,
    ):
        vals = np.array(res[key], dtype=float)
        vals = vals[np.isfinite(vals)]
        out["n_scenarios"] = max(out["n_scenarios"], len(vals))
        if vals.size:
            out["metrics"][key] = {
                "count": int(vals.size),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
                "worst": float(np.max(vals)),
                "floor": float(np.min(vals)),
            }
            # Clip exact zeros to a tiny floor so a log axis can show them.
            plot_vals = np.where(vals <= 0, 1e-18, vals)
            ax.boxplot(plot_vals, vert=True, showfliers=True, widths=0.5)
            ax.set_yscale("log")
        ax.set_title(f"{label}\n(n={vals.size})")
        ax.set_ylabel(f"|residual| ({unit})")
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Study 1: physics-validation residual distributions across the corpus")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "physics_residuals.png", dpi=130)
    plt.close(fig)
    (DATA_DIR / "study1.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"Study 1: {out['n_scenarios']} scenarios | "
        f"mass-balance worst {out['metrics'].get('mass_balance_m3s', {}).get('worst')} | "
        f"headloss worst {out['metrics'].get('hazen_williams_m', {}).get('worst')}"
    )
    return out


# -- Study 2: network sweep -----------------------------------------------


def study2() -> dict:
    print("Study 2: regenerating network sweep ...")
    subprocess.run([PY, "scripts/generate_sweep_configs.py"], cwd=REPO_ROOT, check=True)
    config_paths = sorted((REPO_ROOT / "configs" / "sweep").glob("*.yaml"))
    batch_id = time.strftime("%Y%m%dT%H%M%S")
    report_dir = REPO_ROOT / "outputs" / "batch_runs" / batch_id
    run_batch(config_paths, report_dir=report_dir, batch_id=batch_id, workers=4)
    summary_path = report_dir / "batch_summary.json"
    subprocess.run(
        [PY, "scripts/generate_sweep_report.py", "--batch-summary", str(summary_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    summary = json.loads(summary_path.read_text())
    runs = summary.get("runs", [])
    by_sev: dict[str, int] = {}
    for r in runs:
        by_sev[r["status"]] = by_sev.get(r["status"], 0) + 1
    out = {
        "batch_id": batch_id,
        "n_configs": len(config_paths),
        "by_status": by_sev,
        "report": "docs/network_sweep_report.md",
    }
    (DATA_DIR / "study2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Study 2: {len(config_paths)} configs | status {by_sev}")
    return out


# -- Study 3: parallel benchmark ------------------------------------------


def _parse_speedup_table(doc: str) -> list[dict]:
    rows: list[dict] = []
    for line in doc.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)x\s*\|\s*([0-9.]+)\s*\|", line)
        if m:
            rows.append(
                {
                    "workers": int(m.group(1)),
                    "wall_clock_s": float(m.group(2)),
                    "speedup": float(m.group(3)),
                    "efficiency": float(m.group(4)),
                }
            )
    return rows


def study3() -> dict:
    print("Study 3: re-running parallel benchmark ...")
    subprocess.run(
        [PY, "scripts/benchmark_parallel.py", "--subset", "100", "--workers", "1,2,4,8"],
        cwd=REPO_ROOT,
        check=True,
    )
    doc = (REPO_ROOT / "docs" / "parallelisation_benchmark.md").read_text()
    rows = _parse_speedup_table(doc)
    out = {"report": "docs/parallelisation_benchmark.md", "rows": rows}
    (DATA_DIR / "study3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: r["speedup"]) if rows else {}
    print(f"Study 3: peak speedup {best.get('speedup')}x at {best.get('workers')} workers")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=int, choices=[1, 2, 3], default=None)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in (None, 1):
        study1()
    if args.only in (None, 2):
        study2()
    if args.only in (None, 3):
        study3()


if __name__ == "__main__":
    main()
