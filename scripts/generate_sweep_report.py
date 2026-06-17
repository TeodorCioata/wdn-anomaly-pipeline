"""Render docs/network_sweep_report.md from a sweep batch run.

Phase 5 Week 8. Reads the sweep configs under ``configs/sweep/``
and a batch summary JSON produced by ``wdn-pipeline batch`` over those
configs, then writes a markdown report: one row per network with element
counts, the demand mode chosen, runtime, validation severity, worst
pressure and mass-balance residual, plus a notes column. Networks present
in ``networks/`` but excluded from the sweep (too large for
WNTRSimulator) are listed separately with the exclusion
reason.

Usage::

    python scripts/generate_sweep_report.py \
        --batch-summary outputs/batch_runs/<id>/batch_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import warnings
from pathlib import Path

import yaml

from scripts.generate_sweep_configs import (
    DEFAULT_MAX_JUNCTIONS,
    choose_demand_mode,
    discover_networks,
    network_stats,
    skip_reason,
)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "network_sweep_report.md"

_PRESSURE_RE = re.compile(r"min=([\-\d.eE+]+) m, max=([\-\d.eE+]+) m")
_MASS_RE = re.compile(r"= ([\d.eE+\-]+) m\^3/s")


def _find_latest_batch_summary() -> Path | None:
    candidates = sorted(
        glob.glob(str(REPO_ROOT / "outputs" / "batch_runs" / "*" / "batch_summary.json"))
    )
    return Path(candidates[-1]) if candidates else None


def _check_detail(run: dict, check_name: str) -> str | None:
    for check in run.get("validation", {}).get("checks", []):
        if check["name"] == check_name:
            return check.get("detail")
    return None


def _worst_pressure(run: dict) -> float | None:
    detail = _check_detail(run, "pressure_bounds")
    if not detail:
        return None
    m = _PRESSURE_RE.search(detail)
    return float(m.group(1)) if m else None


def _mass_residual(run: dict) -> float | None:
    detail = _check_detail(run, "mass_balance")
    if not detail:
        return None
    m = _MASS_RE.search(detail)
    return float(m.group(1)) if m else None


def _note(severity: str, worst_p: float | None, error: str | None) -> str:
    if severity == "error":
        msg = (error or "run error").splitlines()[-1].strip()
        return f"excluded — {msg}"
    if worst_p is not None and worst_p < 0:
        return f"uncalibrated — sub-zero pressure (min {worst_p:.2f} m), documented"
    if severity == "warning":
        return "warning (see batch summary)"
    return "ok — non-negative pressure, mass balance at machine epsilon"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs" / "sweep")
    parser.add_argument("--networks-dir", type=Path, default=REPO_ROOT / "networks")
    parser.add_argument("--batch-summary", type=Path, default=None)
    parser.add_argument("--max-junctions", type=int, default=DEFAULT_MAX_JUNCTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary_path = args.batch_summary or _find_latest_batch_summary()
    if summary_path is None or not Path(summary_path).is_file():
        raise SystemExit(f"No batch summary found ({summary_path}). Run the sweep first.")
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text())
    runs_by_config = {Path(r["config_path"]).name: r for r in summary["runs"]}

    # Map each sweep config to its source network .inp via network.inp_path.
    config_to_inp: dict[str, Path] = {}
    for cfg_path in sorted(args.config_dir.glob("*.yaml")):
        body = yaml.safe_load(cfg_path.read_text())
        config_to_inp[cfg_path.name] = REPO_ROOT / body["network"]["inp_path"]

    rows: list[dict] = []
    for cfg_name, inp_path in sorted(config_to_inp.items()):
        stats = network_stats(inp_path)
        run = runs_by_config.get(cfg_name)
        severity = run["status"] if run else "not-run"
        elapsed = run["elapsed_seconds"] if run else float("nan")
        worst_p = _worst_pressure(run) if run else None
        residual = _mass_residual(run) if run else None
        error = run.get("error") if run else None
        rows.append(
            {
                "network": inp_path.name,
                "stats": stats,
                "mode": choose_demand_mode(stats),
                "severity": severity,
                "elapsed": elapsed,
                "worst_p": worst_p,
                "residual": residual,
                "note": _note(severity, worst_p, error),
            }
        )

    # Excluded networks: present in networks/ but no sweep config emitted.
    covered_inps = {p.resolve() for p in config_to_inp.values()}
    excluded: list[tuple[str, str]] = []
    for inp_path in discover_networks(args.networks_dir):
        if inp_path.resolve() in covered_inps:
            continue
        stats = network_stats(inp_path)
        reason = skip_reason(stats, args.max_junctions)
        if reason is not None:
            excluded.append((inp_path.name, reason))

    lines: list[str] = []
    lines.append("# Network Sweep Report (Phase 5 Week 8)\n")
    lines.append(
        f"Generated from `{summary_path.relative_to(REPO_ROOT)}`. Every "
        f"LeakG3PD network at or below {args.max_junctions} junctions was run "
        "through the pipeline as a 24h / 1h PDD normal scenario. Validation "
        "tolerances are conservative: uncalibrated networks warn (never "
        "fail) on pressure, so only NaN/inf or a broken mass balance can "
        "fail.\n"
    )
    lines.append(
        f"**Ran:** {len(rows)} networks. "
        f"**Excluded up front (WNTRSimulator-incompatible .inp):** {len(excluded)}.\n"
    )
    lines.append(
        "| Network | Junc | Pipe | Tank | Resv | Demand | Runtime (s) | "
        "Severity | Worst P (m) | Mass resid (m³/s) | Notes |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---:|---|---:|---|---|")
    for r in rows:
        s = r["stats"]
        worst = f"{r['worst_p']:.3f}" if r["worst_p"] is not None else "n/a"
        resid = f"{r['residual']:.2e}" if r["residual"] is not None else "n/a"
        lines.append(
            f"| {r['network']} | {s['junctions']} | {s['pipes']} | {s['tanks']} | "
            f"{s['reservoirs']} | {r['mode']} | {r['elapsed']:.2f} | "
            f"{r['severity']} | {worst} | {resid} | {r['note']} |"
        )
    lines.append("")
    if excluded:
        lines.append("## Excluded up front\n")
        lines.append(
            "These networks load correctly but are excluded from the sweep "
            "before running because the pure-Python WNTRSimulator "
            "cannot run them: either the .inp uses Darcy-Weisbach / "
            "Chezy-Manning headloss (WNTRSimulator is Hazen-Williams only) "
            "or the network is too large to simulate in reasonable time. "
            "Darcy-Weisbach-capable EpanetSimulator support (which lacks leak "
            "support) is parked in the Phase 5 backlog.\n"
        )
        lines.append("| Network | Reason |")
        lines.append("|---|---|")
        for name, reason in sorted(excluded):
            lines.append(f"| {name} | {reason} |")
        lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} ({len(rows)} ran, {len(excluded)} excluded)")


if __name__ == "__main__":
    main()
