"""Run WNTR simulations and return structured results.

We always use :class:`wntr.sim.WNTRSimulator` (decision D3): leak
support requires it in Phase 3 and using a single simulator across
scenario types avoids subtle numerical inconsistencies between normal
and faulty datasets. The trade-off is speed; this is acceptable for
the project's target scale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import wntr
from wntr.network import WaterNetworkModel


@dataclass(frozen=True)
class SimulationResults:
    """Time-series outputs of a single simulation.

    All DataFrames share the same index: time in seconds since
    simulation start, sampled at the report timestep.

    Attributes:
        pressure: rows = time, columns = node names. Pressure in metres.
        flowrate: rows = time, columns = link names. Flow in m^3/s.
        demand: rows = time, columns = node names. Actual demand
            delivered (matches base_demand * pattern under DDA).
        elapsed_seconds: Wall-clock time the simulation took to run.
    """

    pressure: pd.DataFrame
    flowrate: pd.DataFrame
    demand: pd.DataFrame
    elapsed_seconds: float


def run_simulation(wn: WaterNetworkModel) -> SimulationResults:
    """Execute the simulation and unpack pressure, flowrate and demand.

    The supplied model is consumed by the simulator; do not call this
    twice on the same model.
    """

    simulator = wntr.sim.WNTRSimulator(wn)
    started = time.perf_counter()
    results = simulator.run_sim()
    elapsed = time.perf_counter() - started

    return SimulationResults(
        pressure=results.node["pressure"].copy(),
        flowrate=results.link["flowrate"].copy(),
        demand=results.node["demand"].copy(),
        elapsed_seconds=elapsed,
    )
