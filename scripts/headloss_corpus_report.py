"""Run the Hazen-Williams head loss validator across a config corpus.

Reports, per config and in aggregate: the head loss check severity and the
max ``|dh_observed - dh_predicted|`` residual, so the report can cite a
corpus-wide energy-conservation figure alongside the mass-balance result.

The validator is exercised directly (no outputs are written): each config is
loaded, the network prepared exactly as the runner does (hydraulic options,
quality, demand, leaks), simulated, then passed to
``_check_hazen_williams_headloss``. Sensor faults are intentionally skipped:
the check reads the clean head/flow frames, so a sensor fault cannot change
its result.

Usage::

    python scripts/headloss_corpus_report.py                 # standard configs
    python scripts/headloss_corpus_report.py configs/scale   # a directory
    python scripts/headloss_corpus_report.py 'configs/*.yaml' configs/sweep
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np

from wdn_pipeline.config import load_config
from wdn_pipeline.demand import apply_demand
from wdn_pipeline.faults.leak import LeakInjector
from wdn_pipeline.network import (
    apply_hydraulic_options,
    apply_quality_options,
    load_network,
)
from wdn_pipeline.simulation import run_simulation
from wdn_pipeline.validation import _check_hazen_williams_headloss

_RES_RE = re.compile(r"max \|dh_obs - dh_pred\| = ([0-9.eE+-]+) m")


def residual_of(detail: str) -> float:
    m = _RES_RE.search(detail)
    return float(m.group(1)) if m else float("nan")


def run_one(path: str) -> tuple[str, float, str]:
    cfg = load_config(path)
    wn = load_network(cfg.network, cfg.simulation)
    apply_hydraulic_options(wn, cfg.simulation)
    apply_quality_options(wn, cfg.simulation)
    apply_demand(wn, cfg.demand, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if cfg.faults.leaks:
        LeakInjector().apply(wn, list(cfg.faults.leaks), rng)
    results = run_simulation(wn, use_epanet_for_quality=cfg.simulation.quality.enabled)
    check = _check_hazen_williams_headloss(wn, results, cfg.validation)
    return (
        check.severity,
        residual_of(check.detail),
        "EPANET" if cfg.simulation.quality.enabled else "WNTR",
    )


def expand(args: list[str]) -> list[str]:
    if not args:
        return sorted(glob.glob("configs/*.yaml"))
    out: list[str] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            out.extend(sorted(glob.glob(str(p / "*.yaml"))))
        else:
            out.extend(sorted(glob.glob(a)))
    return out


def main(argv: list[str]) -> int:
    paths = expand(argv)
    residuals: list[float] = []
    n_ok = n_warn = n_fail = n_err = 0
    worst: list[tuple[float, str, str]] = []
    for path in paths:
        try:
            sev, res, kind = run_one(path)
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            print(f"  ERROR  {path}: {type(exc).__name__}: {exc}")
            continue
        residuals.append(res)
        worst.append((res, path, kind))
        if sev == "ok":
            n_ok += 1
        elif sev == "warning":
            n_warn += 1
            print(f"  WARN   {path} ({kind}): residual={res:.3e} m")
        else:
            n_fail += 1
            print(f"  FAIL   {path} ({kind}): residual={res:.3e} m")

    arr = np.array([r for r in residuals if np.isfinite(r)])
    print("\n=== Hazen-Williams head loss corpus report ===")
    print(f"configs run        : {len(residuals)} (errors: {n_err})")
    print(f"severity           : ok={n_ok} warning={n_warn} fail={n_fail}")
    if arr.size:
        print(f"max residual       : {arr.max():.3e} m")
        print(f"median residual    : {np.median(arr):.3e} m")
        print(f"95th pct residual  : {np.percentile(arr, 95):.3e} m")
        worst.sort(reverse=True)
        print("worst 5 configs    :")
        for res, path, kind in worst[:5]:
            print(f"    {res:.3e} m  {kind:6s}  {path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
