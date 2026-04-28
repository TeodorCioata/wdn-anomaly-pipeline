"""Physics-based validation checks for simulation outputs.

Per supervisor guidance, validation prioritises physics correctness
over similarity to existing datasets. Each check returns a
:class:`ValidationCheck` carrying one of three severities (D12):

- ``ok``: clean.
- ``warning``: a soft anomaly within a configured tolerance band
  (e.g. small DDA-mode pressure dips like Net3 node "10" producing
  ~-0.66 m, which both WNTRSimulator and the EpanetSimulator reference
  also produce). Logged but does not fail the pipeline.
- ``fail``: a hard violation. The pipeline returns non-zero.

The aggregate :class:`ValidationReport` exposes overall severity, a
``passed`` shortcut (``True`` iff no ``fail``), and a printable summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from wntr.network import WaterNetworkModel

from wdn_pipeline.config import ValidationConfig
from wdn_pipeline.simulation import SimulationResults

Severity = Literal["ok", "warning", "fail"]


def _max_severity(severities: list[Severity]) -> Severity:
    if "fail" in severities:
        return "fail"
    if "warning" in severities:
        return "warning"
    return "ok"


@dataclass(frozen=True)
class ValidationCheck:
    """One named check with a severity and a human-readable summary."""

    name: str
    severity: Severity
    detail: str

    @property
    def passed(self) -> bool:
        """``True`` unless the check is a hard fail."""

        return self.severity != "fail"


@dataclass(frozen=True)
class ValidationReport:
    """Result of running every validation check on one scenario."""

    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def severity(self) -> Severity:
        return _max_severity([c.severity for c in self.checks])

    @property
    def passed(self) -> bool:
        """``True`` iff no check has severity ``fail``."""

        return self.severity != "fail"

    def format(self) -> str:
        """Return a multi-line printable summary."""

        lines = [f"Validation: {self.severity.upper()}"]
        for c in self.checks:
            lines.append(f"  [{c.severity:>7}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _check_finite(results: SimulationResults) -> ValidationCheck:
    bad = {
        "pressure": int(results.pressure.isna().sum().sum())
        + int(np.isinf(results.pressure.to_numpy()).sum()),
        "flowrate": int(results.flowrate.isna().sum().sum())
        + int(np.isinf(results.flowrate.to_numpy()).sum()),
        "demand": int(results.demand.isna().sum().sum())
        + int(np.isinf(results.demand.to_numpy()).sum()),
    }
    if sum(bad.values()) == 0:
        return ValidationCheck("finite_values", "ok", "no NaN or inf in any output")
    return ValidationCheck(
        "finite_values",
        "fail",
        f"NaN/inf counts: pressure={bad['pressure']}, flowrate={bad['flowrate']}, demand={bad['demand']}",
    )


def _negative_pressure_locations(
    pressure: pd.DataFrame, threshold_m: float
) -> list[tuple[str, int, float]]:
    """Return (node, time_seconds, pressure) tuples below ``threshold_m``."""

    arr = pressure.to_numpy()
    rows, cols = np.where(arr < threshold_m - 1e-12)
    out: list[tuple[str, int, float]] = []
    for r, c in zip(rows, cols, strict=False):
        out.append((str(pressure.columns[c]), int(pressure.index[r]), float(arr[r, c])))
    return out


def _check_pressure_bounds(
    results: SimulationResults, cfg: ValidationConfig
) -> ValidationCheck:
    p = results.pressure.to_numpy()
    pmin = float(np.nanmin(p))
    pmax = float(np.nanmax(p))
    warn_floor = cfg.pressure_min_m - cfg.pressure_min_warning_tolerance_m

    severity: Severity
    issues: list[str] = []
    if pmin < warn_floor:
        severity = "fail"
    elif pmin < cfg.pressure_min_m - 1e-9:
        severity = "warning"
        offenders = _negative_pressure_locations(results.pressure, cfg.pressure_min_m)
        # Group by node and show the worst offender per node.
        by_node: dict[str, tuple[int, float]] = {}
        for node, t, v in offenders:
            if node not in by_node or v < by_node[node][1]:
                by_node[node] = (t, v)
        issues.append(
            "low-pressure nodes (worst sample): "
            + ", ".join(
                f"{n}@t={t}s={v:.3f}m" for n, (t, v) in sorted(by_node.items())
            )
        )
    else:
        severity = "ok"

    if pmax > cfg.pressure_max_m + 1e-9:
        severity = "fail"
        issues.append(f"max pressure {pmax:.3f} m exceeds upper bound {cfg.pressure_max_m} m")

    detail = (
        f"min={pmin:.3f} m, max={pmax:.3f} m, "
        f"strict=[{cfg.pressure_min_m}, {cfg.pressure_max_m}], "
        f"warn floor={warn_floor:.3f} m"
    )
    if issues:
        detail += " | " + "; ".join(issues)
    return ValidationCheck("pressure_bounds", severity, detail)


def _build_incidence(
    wn: WaterNetworkModel,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Signed junction-link incidence matrix.

    Entry ``M[j, l]`` is ``+1`` if link ``l`` ends at junction ``j``,
    ``-1`` if it starts there, ``0`` otherwise.
    """

    junctions = list(wn.junction_name_list)
    links = list(wn.link_name_list)
    j_idx = {n: i for i, n in enumerate(junctions)}
    matrix = np.zeros((len(junctions), len(links)), dtype=float)
    for col, link_name in enumerate(links):
        link = wn.get_link(link_name)
        if link.start_node_name in j_idx:
            matrix[j_idx[link.start_node_name], col] = -1.0
        if link.end_node_name in j_idx:
            matrix[j_idx[link.end_node_name], col] = +1.0
    return matrix, junctions, links


def _check_mass_balance(
    wn: WaterNetworkModel,
    results: SimulationResults,
    cfg: ValidationConfig,
) -> ValidationCheck:
    incidence, junctions, links = _build_incidence(wn)
    flows = results.flowrate.reindex(columns=links).to_numpy()
    demands = results.demand.reindex(columns=junctions).to_numpy()
    net_inflow = flows @ incidence.T
    residual = net_inflow - demands
    max_residual = float(np.abs(residual).max())
    severity: Severity = "ok" if max_residual <= cfg.mass_balance_tol_m3s else "fail"
    return ValidationCheck(
        "mass_balance",
        severity,
        f"max |net_inflow - demand| = {max_residual:.3e} m^3/s "
        f"(tol {cfg.mass_balance_tol_m3s:.0e})",
    )


def validate_normal_scenario(
    wn: WaterNetworkModel,
    results: SimulationResults,
    cfg: ValidationConfig,
) -> ValidationReport:
    """Run every check for a normal (no-fault) scenario."""

    return ValidationReport(
        checks=[
            _check_finite(results),
            _check_pressure_bounds(results, cfg),
            _check_mass_balance(wn, results, cfg),
        ]
    )


def residuals(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise difference of two aligned DataFrames.

    Useful for determinism checks (two runs with the same seed should
    produce all-zero residuals) and external baselines.
    """

    aligned_left, aligned_right = left.align(right, join="inner", axis=None)
    return aligned_left - aligned_right
