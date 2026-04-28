"""Pluggable demand-pattern strategies.

Three modes (selected by :attr:`wdn_pipeline.config.DemandConfig.mode`):

- ``default``: leave the .inp file's patterns untouched.
- ``scale``:  multiply every junction's base demand by a constant.
- ``fourier``: replace every junction's pattern with a synthetic
  diurnal pattern of the form
  ``m(t) = base + amplitude*sin(2*pi*t/period + phase) + N(0, sigma)``.

Adding a new strategy means adding a new :class:`DemandStrategy`
subclass and registering it in :data:`STRATEGIES`. No other module
needs to change.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from wntr.network import WaterNetworkModel

from wdn_pipeline.config import DemandConfig, FourierDemandParams, ScaleDemandParams

_FOURIER_PATTERN_NAME = "synthetic_fourier"


class DemandStrategy(ABC):
    """Abstract demand strategy.

    Concrete strategies mutate the supplied :class:`WaterNetworkModel`
    in place, so callers must pass a fresh model.
    """

    @abstractmethod
    def apply(self, wn: WaterNetworkModel, seed: int) -> None: ...


class DefaultDemand(DemandStrategy):
    """No-op: keep the patterns and base demands from the .inp file."""

    def apply(self, wn: WaterNetworkModel, seed: int) -> None:  # noqa: D401
        return None


class ScaleDemand(DemandStrategy):
    """Multiply every junction's base demand by a constant."""

    def __init__(self, params: ScaleDemandParams) -> None:
        self.multiplier = float(params.multiplier)

    def apply(self, wn: WaterNetworkModel, seed: int) -> None:
        for _, junction in wn.junctions():
            for ts in junction.demand_timeseries_list:
                ts.base_value = ts.base_value * self.multiplier


class FourierDemand(DemandStrategy):
    """Replace per-junction patterns with a single synthetic Fourier pattern.

    The pattern length covers the whole simulation duration at the
    network's pattern timestep. All junctions are pointed at this new
    pattern; their base demands are preserved.
    """

    def __init__(self, params: FourierDemandParams) -> None:
        self.params = params

    def apply(self, wn: WaterNetworkModel, seed: int) -> None:
        pattern_ts = int(wn.options.time.pattern_timestep)
        duration = int(wn.options.time.duration)
        if pattern_ts <= 0:
            raise ValueError("pattern_timestep must be > 0 for fourier demand mode")
        # +1 so the pattern covers the closed time interval [0, duration].
        n_steps = max(1, duration // pattern_ts + 1)

        rng = np.random.default_rng(seed)
        period_s = self.params.period_hours * 3600.0
        phase = 2.0 * math.pi * (self.params.phase_shift_hours * 3600.0) / period_s
        omega = 2.0 * math.pi / period_s

        t = np.arange(n_steps, dtype=float) * pattern_ts
        multipliers = (
            self.params.base
            + self.params.amplitude * np.sin(omega * t + phase)
            + rng.normal(0.0, self.params.noise_std, size=n_steps)
        )
        # Demand multipliers must be non-negative for physical realism.
        multipliers = np.clip(multipliers, 0.0, None)

        if _FOURIER_PATTERN_NAME in wn.pattern_name_list:
            wn.remove_pattern(_FOURIER_PATTERN_NAME)
        wn.add_pattern(_FOURIER_PATTERN_NAME, multipliers.tolist())

        for _, junction in wn.junctions():
            for ts in junction.demand_timeseries_list:
                ts.pattern_name = _FOURIER_PATTERN_NAME


STRATEGIES: dict[str, type[DemandStrategy]] = {
    "default": DefaultDemand,
    "scale": ScaleDemand,
    "fourier": FourierDemand,
}


def build_strategy(cfg: DemandConfig) -> DemandStrategy:
    """Construct the strategy implementation selected by ``cfg.mode``."""

    if cfg.mode == "default":
        return DefaultDemand()
    if cfg.mode == "scale":
        return ScaleDemand(cfg.scale)
    if cfg.mode == "fourier":
        return FourierDemand(cfg.fourier)
    raise ValueError(f"Unknown demand mode: {cfg.mode}")


def apply_demand(wn: WaterNetworkModel, cfg: DemandConfig, seed: int) -> None:
    """Apply the configured demand strategy to ``wn`` in place."""

    build_strategy(cfg).apply(wn, seed)
