"""Pipe leak injection.

A leak is a hydraulic-model modification: the pipe is split, a new
junction is inserted, and a leak orifice is added at that junction
following WNTR's ``Q = Cd * A * sqrt(2 * g * h)`` model. Leaks must run
under :class:`wntr.sim.WNTRSimulator` with PDD enabled (D16); the
config validator rejects mismatched scenarios up front.

Three profiles:

- ``abrupt``: leak switches on at ``start_time`` with the configured
  area and switches off at ``end_time``. Implemented via the WNTR
  built-in ``junction.add_leak``.
- ``linear``: area grows linearly from ``target_area / n`` to
  ``target_area`` between ``start_time`` and ``end_time``. Rendered as
  ``n`` step controls, with ``n = profile_steps`` (default 30). At step
  ``k`` (0-indexed) the area is ``target_area * (k+1) / n``.
- ``step``: same step-function rendering as ``linear``; provided as a
  user-facing label so explicit "staircase" leaks remain semantically
  distinct in configs and metadata.

location: ``pipe`` and ``split_fraction`` may either
be set explicitly per leak or left ``None`` to draw a value from the
scenario RNG. Random draws are reproducible because the runner threads
``np.random.default_rng(config.seed)`` through every fault module.

Random pipe selection rejects any link whose ``link_type`` is not
``"Pipe"`` so pumps and valves are skipped (matches the LeakG3PD
convention). The split point is sampled from a uniform distribution on
``(0.05, 0.95)`` to avoid degenerate splits at pipe endpoints, which
WNTR rejects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np
import wntr
from wntr.network import WaterNetworkModel
from wntr.network.controls import Control, ControlAction

from wdn_pipeline.config import LeakSpec

logger = logging.getLogger("wdn_pipeline.faults.leak")

# Sampling bounds for random split fractions. Avoids degenerate splits at
# pipe endpoints which WNTR rejects.
_SPLIT_FRACTION_MIN = 0.05
_SPLIT_FRACTION_MAX = 0.95


@dataclass(frozen=True)
class ResolvedLeak:
    """Concrete parameters for one realised leak.

    Every field is a hard value, not ``None``. Random fields from the
    :class:`LeakSpec` are sampled before construction so this object is
    safe to serialise into metadata as the canonical record of what the
    leak actually was.

    Attributes:
        pipe: Original pipe name that was split.
        split_fraction: Position along the pipe at which the leak
            junction was inserted (0 = upstream, 1 = downstream).
        area_m2: Leak hole area in square metres.
        diameter_m: Hole diameter (m) if the spec was given by diameter,
            else ``None``.
        discharge_coeff: Leak discharge coefficient.
        start_time_seconds: Leak switch-on time.
        end_time_seconds: Leak switch-off time.
        profile: One of ``abrupt`` / ``linear`` / ``step``.
        profile_steps: Step count for incipient profiles (``1`` for
            abrupt; informational only).
        leak_node_name: Name of the new junction inserted at the leak
            location.
        new_pipe_name: Name of the new downstream pipe segment created
            by ``wntr.morph.split_pipe``.
        name: Optional human-readable label.
    """

    pipe: str
    split_fraction: float
    area_m2: float
    diameter_m: float | None
    discharge_coeff: float
    start_time_seconds: int
    end_time_seconds: int
    profile: str
    profile_steps: int
    leak_node_name: str
    new_pipe_name: str
    name: str | None

    def to_dict(self) -> dict:
        """Plain dict suitable for sidecar metadata serialisation."""

        return asdict(self)


class LeakInjector:
    """Apply a list of :class:`LeakSpec` to a WNTR network in place.

    Stateless; one instance can be reused, but typical callers
    instantiate a fresh injector per scenario for clarity.
    """

    def apply(
        self,
        wn: WaterNetworkModel,
        leak_specs: list[LeakSpec],
        rng: np.random.Generator,
    ) -> list[ResolvedLeak]:
        """Inject every leak and return their resolved parameters.

        Args:
            wn: Mutable :class:`WaterNetworkModel`. The model is mutated
                in place: each leak adds one junction, one pipe segment
                and one or more leak controls.
            leak_specs: Specs in declaration order. Random fields are
                resolved using ``rng`` in this same order so deterministic
                runs draw values reproducibly.
            rng: A seeded NumPy RNG. The runner builds this from
                ``config.seed``.

        Returns:
            A list of :class:`ResolvedLeak` aligned with ``leak_specs``.
        """

        resolved: list[ResolvedLeak] = []
        for idx, spec in enumerate(leak_specs):
            r = self._inject_one(wn, spec, rng, idx)
            logger.info(
                "Injected leak %d: pipe=%s split=%.3f area=%.3e profile=%s window=[%d,%d]s",
                idx,
                r.pipe,
                r.split_fraction,
                r.area_m2,
                r.profile,
                r.start_time_seconds,
                r.end_time_seconds,
            )
            resolved.append(r)
        return resolved

    def _inject_one(
        self,
        wn: WaterNetworkModel,
        spec: LeakSpec,
        rng: np.random.Generator,
        idx: int,
    ) -> ResolvedLeak:
        pipe = self._resolve_pipe(wn, spec, rng)
        split_fraction = self._resolve_split_fraction(spec, rng)
        area = self._resolve_area(spec)

        leak_node_name = self._unique_name(wn.junction_name_list, f"leak_{idx}_{pipe}")
        new_pipe_name = self._unique_name(wn.pipe_name_list, f"pipe_{idx}_{pipe}_B")

        wntr.morph.split_pipe(
            wn,
            pipe,
            new_pipe_name,
            leak_node_name,
            split_at_point=split_fraction,
            return_copy=False,
        )
        junction = wn.get_node(leak_node_name)

        if spec.profile == "abrupt":
            junction.add_leak(
                wn,
                area=area,
                discharge_coeff=spec.discharge_coeff,
                start_time=int(spec.start_time_seconds),
                end_time=int(spec.end_time_seconds),
            )
            steps_used = 1
        else:
            self._schedule_incipient_leak(
                wn=wn,
                junction=junction,
                target_area=area,
                discharge_coeff=spec.discharge_coeff,
                start_time=int(spec.start_time_seconds),
                end_time=int(spec.end_time_seconds),
                n_steps=spec.profile_steps,
                node_name=leak_node_name,
            )
            steps_used = spec.profile_steps

        return ResolvedLeak(
            pipe=pipe,
            split_fraction=split_fraction,
            area_m2=area,
            diameter_m=spec.diameter_m,
            discharge_coeff=spec.discharge_coeff,
            start_time_seconds=int(spec.start_time_seconds),
            end_time_seconds=int(spec.end_time_seconds),
            profile=spec.profile,
            profile_steps=steps_used,
            leak_node_name=leak_node_name,
            new_pipe_name=new_pipe_name,
            name=spec.name,
        )

    @staticmethod
    def _resolve_pipe(
        wn: WaterNetworkModel, spec: LeakSpec, rng: np.random.Generator
    ) -> str:
        if spec.pipe is not None:
            if spec.pipe not in wn.link_name_list:
                raise ValueError(
                    f"LeakSpec.pipe={spec.pipe!r} is not a link in the network"
                )
            link = wn.get_link(spec.pipe)
            if link.link_type != "Pipe":
                raise ValueError(
                    f"LeakSpec.pipe={spec.pipe!r} is a {link.link_type}; only pipes can leak"
                )
            return spec.pipe

        candidates = [
            name
            for name in wn.link_name_list
            if wn.get_link(name).link_type == "Pipe"
        ]
        if not candidates:
            raise ValueError("No pipes available for random leak placement")
        return str(rng.choice(candidates))

    @staticmethod
    def _resolve_split_fraction(spec: LeakSpec, rng: np.random.Generator) -> float:
        if spec.split_fraction is not None:
            return float(spec.split_fraction)
        return float(rng.uniform(_SPLIT_FRACTION_MIN, _SPLIT_FRACTION_MAX))

    @staticmethod
    def _resolve_area(spec: LeakSpec) -> float:
        if spec.area_m2 is not None:
            return float(spec.area_m2)
        # The config validator guarantees diameter_m is set if area_m2 is None.
        diameter = float(spec.diameter_m)  # type: ignore[arg-type]
        return math.pi * (diameter / 2.0) ** 2

    @staticmethod
    def _unique_name(existing: list[str], base: str) -> str:
        if base not in existing:
            return base
        suffix = 1
        while f"{base}_{suffix}" in existing:
            suffix += 1
        return f"{base}_{suffix}"

    @staticmethod
    def _schedule_incipient_leak(
        *,
        wn: WaterNetworkModel,
        junction,
        target_area: float,
        discharge_coeff: float,
        start_time: int,
        end_time: int,
        n_steps: int,
        node_name: str,
    ) -> None:
        """Wire up the step-function controls for a linear/step profile.

        ``add_leak`` already creates the on/off controls that flip
        ``leak_status``. We piggy-back on those and override the area
        trajectory: at ``start_time`` the leak begins at ``target/n`` and
        each subsequent step control raises it until it reaches
        ``target_area`` at the final step. The off control then flips
        ``leak_status`` to ``False`` at ``end_time``, which is what
        actually closes the leak.
        """

        junction.add_leak(
            wn,
            area=target_area,
            discharge_coeff=discharge_coeff,
            start_time=start_time,
            end_time=end_time,
        )
        # add_leak set _leak_area = target_area; reset to the first step value
        # so the leak starts small and ramps up.
        junction._leak_area = target_area * 1.0 / n_steps

        dt = (end_time - start_time) / n_steps
        for k in range(1, n_steps):
            t = int(round(start_time + k * dt))
            if t >= end_time:
                # Past the off control; would have no effect.
                break
            area_k = target_area * (k + 1) / n_steps
            action = ControlAction(junction, "_leak_area", area_k)
            ctrl = Control._time_control(wn, t, "SIM_TIME", False, action)
            wn.add_control(f"{node_name}_leak_step_{k}", ctrl)
