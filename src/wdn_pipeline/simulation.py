"""Run WNTR simulations and return structured results.

We always use :class:`wntr.sim.WNTRSimulator` :
leak support requires it and using a single simulator across
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
        leak_demand: rows = time, columns = node names. Volumetric leak
            outflow at each junction in m^3/s. Empty (zero columns) for
            scenarios that did not inject leaks.
        elapsed_seconds: Wall-clock time the simulation took to run.
    """

    pressure: pd.DataFrame
    flowrate: pd.DataFrame
    demand: pd.DataFrame
    leak_demand: pd.DataFrame
    elapsed_seconds: float


def run_simulation(wn: WaterNetworkModel) -> SimulationResults:
    """Execute the simulation and unpack pressure, flowrate, demand and leak demand.

    The supplied model is consumed by the simulator; do not call this
    twice on the same model.
    """

    simulator = wntr.sim.WNTRSimulator(wn)
    started = time.perf_counter()
    results = simulator.run_sim()
    elapsed = time.perf_counter() - started

    pressure = results.node["pressure"].copy()
    flowrate = results.link["flowrate"].copy()
    demand = results.node["demand"].copy()
    if "leak_demand" in results.node:
        leak_demand = results.node["leak_demand"].copy()
    else:
        # Older WNTR releases or no-leak scenarios: emit a zero frame
        # aligned with the pressure index/columns so downstream code
        # never has to special-case absence.
        leak_demand = pd.DataFrame(0.0, index=pressure.index, columns=pressure.columns)

    return SimulationResults(
        pressure=pressure,
        flowrate=flowrate,
        demand=demand,
        leak_demand=leak_demand,
        elapsed_seconds=elapsed,
    )
