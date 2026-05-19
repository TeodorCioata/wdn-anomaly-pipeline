"""Sensor fault injection (post-simulation, D5).

A sensor fault corrupts the reported signal at a single channel without
touching the hydraulic model. The injector takes a
:class:`SimulationResults` plus a list of :class:`SensorFaultSpec` and
returns a *new* :class:`SimulationResults` whose ``pressure`` and / or
``flowrate`` columns are corrupted at the faulted channels. The
original simulator output is preserved verbatim in the ``pressure_clean``
and ``flowrate_clean`` companion frames.

Six strategies are implemented (D18):

- ``bias``: constant offset ``y'(t) = y(t) + bias_value``
- ``drift``: linear ramp ``y'(t) = y(t) + slope * (t - start)``
- ``stuck``: freeze at start value ``y'(t) = y(start)``
- ``dropout``: ``y'(t) = NaN`` (or ``fill_value``) on the configured
  sub-intervals
- ``noise``: additive Gaussian ``y'(t) = y(t) + N(0, sigma**2)``
- ``gain``: multiplicative factor ``y'(t) = gain_factor * y(t)``

Every fault uses the half-open window convention ``[start, end)`` from
Phase 3 leaks. RNG threading mirrors :mod:`wdn_pipeline.faults.leak`:
the runner passes a single ``numpy.random.default_rng(seed)`` and the
injector consumes draws in spec declaration order so reproducibility is
deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from wntr.network import WaterNetworkModel

from wdn_pipeline.config import SensorFaultSpec
from wdn_pipeline.simulation import SimulationResults

logger = logging.getLogger("wdn_pipeline.faults.sensor")


@dataclass(frozen=True)
class ResolvedSensorFault:
    """Concrete parameters for one realised sensor fault.

    Mirrors :class:`wdn_pipeline.faults.leak.ResolvedLeak`: every random
    field of the spec has been resolved to a concrete value so this
    object is safe to serialise as the canonical record of what the
    fault actually was.

    Attributes:
        type: Fault type discriminator.
        quantity: Output table the fault targets.
        target: Concrete channel name (junction for pressure, link for
            flowrate). Never ``None``.
        start_time_seconds: Fault switch-on time (inclusive).
        end_time_seconds: Fault switch-off time (exclusive).
        bias_value: Bias offset (only meaningful for ``type == "bias"``).
        slope_per_second: Drift slope (only meaningful for
            ``type == "drift"``).
        intervals: Dropout sub-intervals (only meaningful for
            ``type == "dropout"``).
        fill_value: Dropout fill (``None`` means NaN).
        sigma: Noise std (only meaningful for ``type == "noise"``).
        rng_offset: Offset used to seed the per-fault noise RNG.
        gain_factor: Multiplicative factor (only meaningful for
            ``type == "gain"``).
        name: Optional human-readable label.
    """

    type: str
    quantity: str
    target: str
    start_time_seconds: int
    end_time_seconds: int
    bias_value: float | None
    slope_per_second: float | None
    intervals: list[tuple[int, int]] | None
    fill_value: float | None
    sigma: float | None
    rng_offset: int
    gain_factor: float | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        """Plain dict suitable for sidecar metadata serialisation."""

        d = asdict(self)
        if d["intervals"] is not None:
            d["intervals"] = [list(iv) for iv in d["intervals"]]
        return d


@dataclass
class SensorFaultApplyResult:
    """What :class:`SensorFaultInjector.apply` returns.

    Attributes:
        results: A fresh :class:`SimulationResults` with the corrupted
            ``pressure`` / ``flowrate`` frames replacing the originals.
            The ``*_clean`` companions still hold the uncorrupted
            simulator output.
        resolved: Concrete parameters for every realised fault, aligned
            with the input specs.
        masks: Mapping ``column_name -> boolean Series`` indicating
            where each fault was active. ``column_name`` follows the
            ``{fault_type}_mask_{target}`` convention (D26).
    """

    results: SimulationResults
    resolved: list[ResolvedSensorFault] = field(default_factory=list)
    masks: dict[str, pd.Series] = field(default_factory=dict)


class SensorFaultInjector:
    """Apply a list of :class:`SensorFaultSpec` to a SimulationResults.

    Stateless. The injector picks targets, constructs masks, builds
    corrupted DataFrames and returns a :class:`SensorFaultApplyResult`.
    The original ``SimulationResults`` is not mutated.
    """

    def apply(
        self,
        results: SimulationResults,
        fault_specs: list[SensorFaultSpec],
        rng: np.random.Generator,
        wn: WaterNetworkModel | None = None,
    ) -> SensorFaultApplyResult:
        """Apply every fault and return corrupted results plus diagnostics.

        Args:
            results: Simulator output to corrupt. Treated as immutable.
            fault_specs: Faults to apply in declaration order. RNG draws
                consume the supplied ``rng`` in that order so two runs
                with the same seed produce identical resolved faults.
            rng: Seeded NumPy RNG. Random target selection and noise
                samples both draw from this generator (the noise stream
                is forked into a sub-generator per fault for
                independence).
            wn: Optional network model. Required only if any fault has
                ``target=None``: the injector picks from
                ``wn.junction_name_list`` for pressure faults and
                ``wn.pipe_name_list`` for flowrate faults.

        Returns:
            A :class:`SensorFaultApplyResult`. The original
            ``results`` is left untouched.
        """

        if not fault_specs:
            return SensorFaultApplyResult(results=results)

        # Copy so we never mutate the simulator output. The clean
        # versions remain on the new SimulationResults.
        corrupted_pressure = results.pressure.copy()
        corrupted_flowrate = results.flowrate.copy()
        clean_pressure = results.pressure_clean
        clean_flowrate = results.flowrate_clean

        resolved_list: list[ResolvedSensorFault] = []
        masks: dict[str, pd.Series] = {}

        for idx, spec in enumerate(fault_specs):
            target = self._resolve_target(spec, wn, rng)
            self._validate_target(target, spec, results)

            if spec.quantity == "pressure":
                clean_col = clean_pressure[target].to_numpy()
                corrupted_col, mask = self._apply_one(
                    spec, clean_col, results.pressure.index, rng, idx
                )
                corrupted_pressure[target] = corrupted_col
                mask_series = pd.Series(
                    mask, index=results.pressure.index, dtype=bool
                )
            else:  # flowrate
                clean_col = clean_flowrate[target].to_numpy()
                corrupted_col, mask = self._apply_one(
                    spec, clean_col, results.flowrate.index, rng, idx
                )
                corrupted_flowrate[target] = corrupted_col
                mask_series = pd.Series(
                    mask, index=results.flowrate.index, dtype=bool
                )

            mask_col = f"{spec.type}_mask_{target}"
            masks[mask_col] = mask_series

            resolved_list.append(
                ResolvedSensorFault(
                    type=spec.type,
                    quantity=spec.quantity,
                    target=target,
                    start_time_seconds=int(spec.start_time_seconds),
                    end_time_seconds=int(spec.end_time_seconds),
                    bias_value=spec.bias_value,
                    slope_per_second=spec.slope_per_second,
                    intervals=(
                        [tuple(map(int, iv)) for iv in spec.intervals]
                        if spec.intervals is not None
                        else None
                    ),
                    fill_value=spec.fill_value,
                    sigma=spec.sigma,
                    rng_offset=int(spec.rng_offset),
                    gain_factor=spec.gain_factor,
                    name=spec.name,
                )
            )
            logger.info(
                "Applied sensor fault %d: type=%s quantity=%s target=%s window=[%d,%d)s",
                idx,
                spec.type,
                spec.quantity,
                target,
                spec.start_time_seconds,
                spec.end_time_seconds,
            )

        new_results = SimulationResults(
            pressure=corrupted_pressure,
            flowrate=corrupted_flowrate,
            demand=results.demand,
            leak_demand=results.leak_demand,
            elapsed_seconds=results.elapsed_seconds,
            pressure_clean=clean_pressure,
            flowrate_clean=clean_flowrate,
        )
        return SensorFaultApplyResult(
            results=new_results, resolved=resolved_list, masks=masks
        )

    @staticmethod
    def _resolve_target(
        spec: SensorFaultSpec,
        wn: WaterNetworkModel | None,
        rng: np.random.Generator,
    ) -> str:
        if spec.target is not None:
            return spec.target
        if wn is None:
            raise ValueError(
                "Random sensor target requested but no WaterNetworkModel "
                "was supplied to SensorFaultInjector.apply"
            )
        if spec.quantity == "pressure":
            candidates = list(wn.junction_name_list)
            kind = "junction"
        else:
            candidates = [
                name
                for name in wn.link_name_list
                if wn.get_link(name).link_type == "Pipe"
            ]
            kind = "pipe"
        if not candidates:
            raise ValueError(
                f"No {kind}s available for random sensor target selection"
            )
        return str(rng.choice(candidates))

    @staticmethod
    def _validate_target(
        target: str, spec: SensorFaultSpec, results: SimulationResults
    ) -> None:
        if spec.quantity == "pressure":
            if target not in results.pressure.columns:
                raise ValueError(
                    f"sensor fault target {target!r} is not a column in the "
                    f"pressure output (quantity=pressure expects a junction name)"
                )
        else:
            if target not in results.flowrate.columns:
                raise ValueError(
                    f"sensor fault target {target!r} is not a column in the "
                    f"flowrate output (quantity=flowrate expects a link name)"
                )

    @staticmethod
    def _window_mask(
        time_index: pd.Index, start: int, end: int
    ) -> np.ndarray:
        """Boolean array, True for indices in the half-open [start, end)."""

        times = time_index.to_numpy()
        return (times >= start) & (times < end)

    def _apply_one(
        self,
        spec: SensorFaultSpec,
        clean: np.ndarray,
        time_index: pd.Index,
        rng: np.random.Generator,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dispatch to the right strategy and return (corrupted, mask).

        ``clean`` is a 1-D NumPy array of the uncorrupted signal at the
        targeted channel, aligned with ``time_index``. The returned
        ``corrupted`` array is the same shape; ``mask`` is a boolean
        array True where the fault is active.
        """

        times = time_index.to_numpy()
        start = int(spec.start_time_seconds)
        end = int(spec.end_time_seconds)
        active = (times >= start) & (times < end)
        corrupted = clean.astype(float).copy()

        if spec.type == "bias":
            corrupted[active] = clean[active] + float(spec.bias_value)  # type: ignore[arg-type]
        elif spec.type == "drift":
            slope = float(spec.slope_per_second)  # type: ignore[arg-type]
            dt = times[active] - start
            corrupted[active] = clean[active] + slope * dt
        elif spec.type == "stuck":
            # Freeze at the value reported at the first active timestep.
            active_idx = np.where(active)[0]
            if active_idx.size == 0:
                return corrupted, active
            stuck_value = float(clean[active_idx[0]])
            corrupted[active] = stuck_value
        elif spec.type == "dropout":
            fill = (
                float("nan") if spec.fill_value is None else float(spec.fill_value)
            )
            # Dropout mask is the union of the configured sub-intervals,
            # not the outer window. The outer window is recorded as the
            # fault's authoritative window for labelling, but corruption
            # only happens inside the sub-intervals.
            sub_active = np.zeros_like(active)
            for a, b in spec.intervals or []:
                sub_active |= (times >= int(a)) & (times < int(b))
            corrupted[sub_active] = fill
            # The mask used by the validator is the sub-interval mask;
            # the outer window covers all sub-intervals by construction.
            return corrupted, sub_active
        elif spec.type == "noise":
            sigma = float(spec.sigma)  # type: ignore[arg-type]
            # Each noise fault forks its own sub-RNG. We mix the parent
            # RNG with rng_offset and the fault index so two noise faults
            # in the same scenario don't share samples.
            seed = int(rng.integers(0, 2**31 - 1)) ^ int(spec.rng_offset) ^ idx
            sub_rng = np.random.default_rng(seed)
            samples = sub_rng.normal(0.0, sigma, size=int(active.sum()))
            corrupted[active] = clean[active] + samples
        elif spec.type == "gain":
            gain = float(spec.gain_factor)  # type: ignore[arg-type]
            corrupted[active] = clean[active] * gain
        else:
            raise ValueError(f"Unknown sensor fault type: {spec.type}")

        return corrupted, active
