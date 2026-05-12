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

Leak scenarios add two leak-specific checks :

- The mass-balance residual subtracts ``leak_demand`` from the demand
  side. Without this fix every leak scenario would fail mass balance
  by exactly the leak outflow.
- A pressure-drop sanity check compares mean pressure at the leak
  junction during the leak window to the mean over the pre-onset
  baseline. The post-onset value must be lower; failure is a warning,
  not a hard failure, because tiny leaks against high-static-head
  baselines may not produce a visible drop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from wntr.network import WaterNetworkModel

from wdn_pipeline.config import ValidationConfig
from wdn_pipeline.faults.leak import ResolvedLeak
from wdn_pipeline.faults.sensor import ResolvedSensorFault
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
    """Net inflow minus (consumer demand + leak demand) at every junction.

    The leak-demand term is critical: WNTR reports leak outflow in the
    ``leak_demand`` frame separately from consumer ``demand``, so a
    leak scenario without this correction registers a residual equal
    to the leak outflow at every active step.
    """

    incidence, junctions, links = _build_incidence(wn)
    flows = results.flowrate.reindex(columns=links).to_numpy()
    demands = results.demand.reindex(columns=junctions).to_numpy()
    leak = (
        results.leak_demand.reindex(columns=junctions)
        .fillna(0.0)
        .to_numpy()
    )
    net_inflow = flows @ incidence.T
    residual = net_inflow - demands - leak
    max_residual = float(np.abs(residual).max())
    severity: Severity = "ok" if max_residual <= cfg.mass_balance_tol_m3s else "fail"
    return ValidationCheck(
        "mass_balance",
        severity,
        f"max |net_inflow - demand - leak| = {max_residual:.3e} m^3/s "
        f"(tol {cfg.mass_balance_tol_m3s:.0e})",
    )


def _check_leak_demand_active(
    results: SimulationResults,
    resolved_leaks: Sequence[ResolvedLeak],
) -> ValidationCheck:
    """Structural check: leak_demand is strictly positive while the leak is active.

    Active window is the half-open interval ``[start, end)``. WNTR's
    end-control flips ``leak_status`` to False at ``end_time_seconds``,
    so the reported ``leak_demand`` at that exact timestep is already
    zero by design and is not part of the active window.

    This check is **not** confounded by diurnal demand patterns: it
    asserts that the leak fires while it should and stops when it
    should, independent of network-wide pressure trends. It replaces
    the pre-onset-vs-during pressure-drop check as the primary
    correctness signal for leak scenarios.
    """

    leak_df = results.leak_demand
    times = leak_df.index.to_numpy()
    issues: list[str] = []
    summaries: list[str] = []
    for leak in resolved_leaks:
        if leak.leak_node_name not in leak_df.columns:
            issues.append(f"leak node {leak.leak_node_name} missing from leak_demand")
            continue
        col = leak_df[leak.leak_node_name].to_numpy()
        active = (times >= leak.start_time_seconds) & (times < leak.end_time_seconds)
        inactive = ~active
        n_active_samples = int(active.sum())
        if n_active_samples == 0:
            issues.append(
                f"{leak.leak_node_name}: leak window covers no reported timesteps "
                f"(start={leak.start_time_seconds}s, end={leak.end_time_seconds}s, "
                f"report_step too coarse?)"
            )
            continue
        active_min = float(col[active].min())
        inactive_max = float(np.abs(col[inactive]).max()) if inactive.any() else 0.0
        summaries.append(
            f"{leak.leak_node_name}: active n={n_active_samples} "
            f"min_active={active_min:.3e} m^3/s "
            f"max|inactive|={inactive_max:.3e} m^3/s"
        )
        if active_min <= 0.0:
            issues.append(
                f"{leak.leak_node_name}: leak_demand <= 0 at an active timestep"
            )
        # Tight tolerance: WNTR reports exact zeros outside the
        # half-open window; anything above 1e-12 m^3/s indicates a
        # window-mismatch bug rather than numerical noise.
        if inactive_max > 1e-12:
            issues.append(
                f"{leak.leak_node_name}: leak_demand non-zero outside active window "
                f"(max |inactive| = {inactive_max:.3e})"
            )

    severity: Severity = "fail" if issues else "ok"
    detail = "; ".join(summaries) if summaries else "no resolved leaks"
    if issues:
        detail += " | " + "; ".join(issues)
    return ValidationCheck("leak_demand_active", severity, detail)


def _check_leak_pressure_drop(
    results: SimulationResults,
    resolved_leaks: Sequence[ResolvedLeak],
    cfg: ValidationConfig,
) -> ValidationCheck:
    """Soft diagnostic: mean leak-junction pressure should drop after onset.

    For every resolved leak this compares the mean pressure at the leak
    junction over the half-open active window ``[start_time, end_time)``
    to the mean over ``[0, start_time)``. The during-leak mean must be
    lower by at least ``leak_pressure_drop_min_m``; otherwise we emit a
    warning.

    This check is **vulnerable to diurnal pressure trends**: a network
    with rising afternoon demand can produce a higher mean pressure
    during the leak window than before it (because the leak is
    coincident with the demand peak) even though the leak is firing
    correctly. For that reason this check never raises ``fail``; it is
    purely informational and flagged as a warning so the supervisor
    can inspect the residual heatmap, which isolates the leak's effect
    cleanly. The structural correctness signal is the leak_demand
    active check above. A future-phase improvement is to replace this
    with a baseline-vs-leak residual comparison over matched
    timestamps; that doubles simulator cost so it is not the default.
    """

    pressure = results.pressure
    times = pressure.index.to_numpy()
    issues: list[str] = []
    summaries: list[str] = []
    for leak in resolved_leaks:
        if leak.leak_node_name not in pressure.columns:
            issues.append(f"leak node {leak.leak_node_name} missing from results")
            continue
        col = pressure[leak.leak_node_name].to_numpy()
        baseline_mask = times < leak.start_time_seconds
        leak_mask = (times >= leak.start_time_seconds) & (times < leak.end_time_seconds)
        if not baseline_mask.any() or not leak_mask.any():
            issues.append(
                f"{leak.leak_node_name}: insufficient samples for baseline/leak window"
            )
            continue
        baseline = float(col[baseline_mask].mean())
        during = float(col[leak_mask].mean())
        drop = baseline - during
        summaries.append(
            f"{leak.leak_node_name}: baseline={baseline:.3f}m during={during:.3f}m drop={drop:.3f}m"
        )
        if drop < cfg.leak_pressure_drop_min_m:
            issues.append(
                f"{leak.leak_node_name} drop {drop:.3f} m < min {cfg.leak_pressure_drop_min_m:.3f} m "
                "(diurnal-confound, not a correctness signal)"
            )

    severity: Severity = "warning" if issues else "ok"
    detail = "; ".join(summaries) if summaries else "no resolved leaks"
    if issues:
        detail += " | " + "; ".join(issues)
    return ValidationCheck("leak_pressure_drop", severity, detail)


def _check_sensor_fault_mask_consistent(
    results: SimulationResults,
    resolved_faults: Sequence[ResolvedSensorFault],
    masks: dict[str, pd.Series],
) -> ValidationCheck:
    """Structural check: each fault's mask matches its [start, end) window.

    For dropout faults the mask is the union of the configured
    sub-intervals (a strict subset of the outer window). For every
    other type the mask must equal the outer window mask exactly.
    """

    if not resolved_faults:
        return ValidationCheck(
            "sensor_fault_mask_consistent", "ok", "no sensor faults"
        )

    time_index = results.pressure.index
    times = time_index.to_numpy()
    issues: list[str] = []

    for fault in resolved_faults:
        mask_col = f"{fault.type}_mask_{fault.target}"
        if mask_col not in masks:
            issues.append(f"{mask_col}: mask not produced by injector")
            continue
        actual = masks[mask_col].to_numpy(dtype=bool)
        if fault.type == "dropout":
            expected = np.zeros_like(times, dtype=bool)
            for a, b in fault.intervals or []:
                expected |= (times >= int(a)) & (times < int(b))
        else:
            expected = (times >= fault.start_time_seconds) & (
                times < fault.end_time_seconds
            )
        if not np.array_equal(actual, expected):
            n_diff = int(np.sum(actual != expected))
            issues.append(
                f"{mask_col}: mask disagrees with window at {n_diff} timesteps"
            )

    severity: Severity = "fail" if issues else "ok"
    detail = (
        f"checked {len(resolved_faults)} fault mask(s)"
        if not issues
        else "; ".join(issues)
    )
    return ValidationCheck("sensor_fault_mask_consistent", severity, detail)


def _check_sensor_fault_signal_applied(
    results: SimulationResults,
    resolved_faults: Sequence[ResolvedSensorFault],
) -> ValidationCheck:
    """Structural check: corrupted minus clean matches the spec formula.

    Bias / drift / stuck / dropout are exact comparisons (up to
    floating-point tolerance). Noise is validated statistically: the
    residual must have near-zero mean (within ``4*sigma/sqrt(n)``) and
    its sample std must be near sigma (relative tolerance 30% to absorb
    small samples, default for Net3's 24-hour 1-hour-step grid).
    """

    if not resolved_faults:
        return ValidationCheck(
            "sensor_fault_signal_applied", "ok", "no sensor faults"
        )

    issues: list[str] = []
    summaries: list[str] = []
    for fault in resolved_faults:
        if fault.quantity == "pressure":
            clean_df = results.pressure_clean
            corrupted_df = results.pressure
        else:
            clean_df = results.flowrate_clean
            corrupted_df = results.flowrate

        if fault.target not in corrupted_df.columns:
            issues.append(
                f"{fault.type}@{fault.target}: target missing from corrupted frame"
            )
            continue

        clean = clean_df[fault.target].to_numpy(dtype=float)
        corrupted = corrupted_df[fault.target].to_numpy(dtype=float)
        times = corrupted_df.index.to_numpy()
        active = (times >= fault.start_time_seconds) & (
            times < fault.end_time_seconds
        )
        inactive = ~active

        # Outside the fault window the two must match bit-for-bit
        # (dropout's outer window is the same; the sub-intervals all
        # lie inside).
        if inactive.any():
            outside_diff = float(
                np.nanmax(np.abs(corrupted[inactive] - clean[inactive]))
                if np.any(np.isfinite(corrupted[inactive]) & np.isfinite(clean[inactive]))
                else 0.0
            )
            if outside_diff > 1e-9:
                issues.append(
                    f"{fault.type}@{fault.target}: corrupted differs from clean "
                    f"outside fault window (max |diff| = {outside_diff:.3e})"
                )

        if fault.type == "bias":
            inside_diff = corrupted[active] - clean[active]
            expected = float(fault.bias_value)  # type: ignore[arg-type]
            max_err = float(np.max(np.abs(inside_diff - expected)))
            summaries.append(
                f"bias@{fault.target}: residual={inside_diff.mean():.3f} "
                f"(expected {expected:.3f}, max_err={max_err:.3e})"
            )
            if max_err > 1e-9:
                issues.append(
                    f"bias@{fault.target}: residual deviates from bias_value "
                    f"(max_err={max_err:.3e})"
                )
        elif fault.type == "drift":
            slope = float(fault.slope_per_second)  # type: ignore[arg-type]
            dt = times[active] - fault.start_time_seconds
            expected_curve = slope * dt
            inside_diff = corrupted[active] - clean[active]
            max_err = float(np.max(np.abs(inside_diff - expected_curve)))
            summaries.append(
                f"drift@{fault.target}: slope={slope:.3e}/s n_active={int(active.sum())} "
                f"max_err={max_err:.3e}"
            )
            if max_err > 1e-9:
                issues.append(
                    f"drift@{fault.target}: residual deviates from "
                    f"slope * dt (max_err={max_err:.3e})"
                )
        elif fault.type == "stuck":
            active_idx = np.where(active)[0]
            if active_idx.size == 0:
                summaries.append(f"stuck@{fault.target}: no active samples")
                continue
            stuck_value = float(clean[active_idx[0]])
            inside_vals = corrupted[active]
            max_err = float(np.max(np.abs(inside_vals - stuck_value)))
            summaries.append(
                f"stuck@{fault.target}: stuck_value={stuck_value:.3f} "
                f"max_err={max_err:.3e}"
            )
            if max_err > 1e-9:
                issues.append(
                    f"stuck@{fault.target}: corrupted differs from "
                    f"y(start) (max_err={max_err:.3e})"
                )
        elif fault.type == "dropout":
            fill_is_nan = fault.fill_value is None
            # Build the sub-interval mask.
            sub_active = np.zeros_like(times, dtype=bool)
            for a, b in fault.intervals or []:
                sub_active |= (times >= int(a)) & (times < int(b))
            inside_vals = corrupted[sub_active]
            if fill_is_nan:
                ok = bool(np.all(np.isnan(inside_vals)))
                summaries.append(
                    f"dropout@{fault.target}: NaN-filled {int(sub_active.sum())} sample(s)"
                )
                if not ok:
                    issues.append(
                        f"dropout@{fault.target}: not all dropout samples are NaN"
                    )
            else:
                fill = float(fault.fill_value)  # type: ignore[arg-type]
                max_err = (
                    float(np.max(np.abs(inside_vals - fill)))
                    if inside_vals.size
                    else 0.0
                )
                summaries.append(
                    f"dropout@{fault.target}: fill={fill:.3f} max_err={max_err:.3e}"
                )
                if max_err > 1e-9:
                    issues.append(
                        f"dropout@{fault.target}: corrupted differs from fill_value "
                        f"(max_err={max_err:.3e})"
                    )
        elif fault.type == "noise":
            sigma = float(fault.sigma)  # type: ignore[arg-type]
            inside_diff = corrupted[active] - clean[active]
            n = inside_diff.size
            if n == 0:
                summaries.append(f"noise@{fault.target}: no active samples")
                continue
            mean = float(np.mean(inside_diff))
            std = float(np.std(inside_diff, ddof=0))
            # Statistical tolerances: 4*sigma/sqrt(n) is ~99.99% CI for
            # the sample mean of a Gaussian. Std uses a 30% relative
            # band to absorb small sample sizes (Net3's 24-hour-1-hour
            # grid has at most 25 samples).
            mean_tol = max(4.0 * sigma / max(1.0, n**0.5), 1e-9)
            std_tol = max(0.30 * sigma, 1e-9)
            summaries.append(
                f"noise@{fault.target}: n={n} mean={mean:.3e} std={std:.3f} "
                f"(sigma={sigma:.3f})"
            )
            if abs(mean) > mean_tol:
                issues.append(
                    f"noise@{fault.target}: residual mean {mean:.3e} exceeds tolerance "
                    f"{mean_tol:.3e}"
                )
            if abs(std - sigma) > std_tol:
                issues.append(
                    f"noise@{fault.target}: residual std {std:.3f} deviates from "
                    f"sigma {sigma:.3f} by more than {std_tol:.3f}"
                )
        else:
            issues.append(f"unknown sensor fault type: {fault.type}")

    severity: Severity = "fail" if issues else "ok"
    detail = "; ".join(summaries) if summaries else "no checks executed"
    if issues:
        detail += " | " + "; ".join(issues)
    return ValidationCheck("sensor_fault_signal_applied", severity, detail)


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


def validate_leak_scenario(
    wn: WaterNetworkModel,
    results: SimulationResults,
    resolved_leaks: Sequence[ResolvedLeak],
    cfg: ValidationConfig,
) -> ValidationReport:
    """Run normal-scenario checks plus the leak-specific ones.

    Two leak-specific checks run:

    - ``leak_demand_active`` (structural, can ``fail``): leak_demand
      is strictly positive while the leak is active and exactly zero
      otherwise.
    - ``leak_pressure_drop`` (diagnostic, only ``warning``): mean
      pressure at the leak junction over the active window vs the
      pre-onset baseline. Confounded by diurnal demand patterns; the
      residual heatmap is a more reliable visual diagnostic.
    """

    return ValidationReport(
        checks=[
            _check_finite(results),
            _check_pressure_bounds(results, cfg),
            _check_mass_balance(wn, results, cfg),
            _check_leak_demand_active(results, resolved_leaks),
            _check_leak_pressure_drop(results, resolved_leaks, cfg),
        ]
    )


def validate_sensor_fault_scenario(
    wn: WaterNetworkModel,
    results: SimulationResults,
    resolved_sensor_faults: Sequence[ResolvedSensorFault],
    masks: dict[str, pd.Series],
    cfg: ValidationConfig,
) -> ValidationReport:
    """Run normal-scenario checks plus the two structural sensor-fault checks.

    ``finite_values`` is intentionally skipped for the corrupted
    ``pressure`` / ``flowrate`` frames when dropout faults are present
    (NaN is the *correct* output there). The clean-signal frames are
    still validated to catch any genuine simulator NaNs.
    """

    has_dropout = any(f.type == "dropout" for f in resolved_sensor_faults)
    if has_dropout:
        # Validate finiteness on the clean ground-truth signals; the
        # corrupted ones may legitimately contain NaN.
        clean_results = SimulationResults(
            pressure=results.pressure_clean,
            flowrate=results.flowrate_clean,
            demand=results.demand,
            leak_demand=results.leak_demand,
            elapsed_seconds=results.elapsed_seconds,
            pressure_clean=results.pressure_clean,
            flowrate_clean=results.flowrate_clean,
        )
        finite_check = _check_finite(clean_results)
        finite_check = ValidationCheck(
            name=finite_check.name,
            severity=finite_check.severity,
            detail="(clean signals only; dropout NaNs are expected) "
            + finite_check.detail,
        )
        bounds_check = _check_pressure_bounds(clean_results, cfg)
        mass_check = _check_mass_balance(wn, clean_results, cfg)
    else:
        finite_check = _check_finite(results)
        bounds_check = _check_pressure_bounds(results, cfg)
        mass_check = _check_mass_balance(wn, results, cfg)

    return ValidationReport(
        checks=[
            finite_check,
            bounds_check,
            mass_check,
            _check_sensor_fault_mask_consistent(
                results, resolved_sensor_faults, masks
            ),
            _check_sensor_fault_signal_applied(results, resolved_sensor_faults),
        ]
    )


def residuals(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Element-wise difference of two aligned DataFrames.

    Useful for determinism checks (two runs with the same seed should
    produce all-zero residuals) and external baselines.
    """

    aligned_left, aligned_right = left.align(right, join="inner", axis=None)
    return aligned_left - aligned_right
