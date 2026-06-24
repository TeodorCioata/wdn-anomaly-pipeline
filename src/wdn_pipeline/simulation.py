"""Run WNTR simulations and return structured results.

We always use :class:`wntr.sim.WNTRSimulator` :
leak support requires it and using a single simulator across
scenario types avoids subtle numerical inconsistencies between normal
and faulty datasets. The trade-off is speed; this is acceptable for
the project's target scale.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import pandas as pd
import wntr
from wntr.network import WaterNetworkModel


@dataclass(frozen=True)
class SimulationResults:
    """Time-series outputs of a single simulation, plus clean baselines.

    All DataFrames share the same index: time in seconds since
    simulation start, sampled at the report timestep.

    ``pressure`` and ``flowrate`` are the signals that flow downstream
    to labelling and output. After sensor faults run they are the
    *corrupted* signals. ``pressure_clean`` and ``flowrate_clean``
    always carry the uncorrupted simulator output (Option A from
    `docs/phase4_plan.md` §3.5). For runs with no sensor faults the
    clean and corrupted frames are equal element-by-element.

    Attributes:
        pressure: rows = time, columns = node names. Pressure in metres.
            May be sensor-corrupted.
        flowrate: rows = time, columns = link names. Flow in m^3/s.
            May be sensor-corrupted.
        demand: rows = time, columns = node names. Actual demand
            delivered (matches base_demand * pattern under DDA).
        leak_demand: rows = time, columns = node names. Volumetric leak
            outflow at each junction in m^3/s. Empty (zero columns) for
            scenarios that did not inject leaks.
        elapsed_seconds: Wall-clock time the simulation took to run.
        pressure_clean: Uncorrupted pressure ground truth. Always
            present; equals ``pressure`` when no sensor faults run.
        flowrate_clean: Uncorrupted flowrate ground truth. Always
            present; equals ``flowrate`` when no sensor faults run.
        quality: Water quality time-series (rows = time, columns = node
            names) when a quality analysis ran through the
            ``EpanetSimulator``. ``None`` for the default WNTRSimulator
            path. Units depend on the quality parameter: mg/L for
            chemical, seconds for age, percent for trace.
        head: Total hydraulic head at every node (rows = time, columns =
            node names) in metres. ``head = pressure head + elevation``;
            this is WNTR's own reported head, used by the Hazen-Williams
            head loss validator. Never sensor-corrupted (sensor faults
            only touch ``pressure`` / ``flowrate``). ``None`` only on
            legacy results objects built without it.
        link_status: Per-timestep link status (rows = time, columns =
            link names): ``1`` = open, ``0`` = closed. Used by the head
            loss validator to skip closed pipes, on which WNTR enforces
            ``flow == 0`` instead of the head loss relationship. ``None``
            if the simulator did not report status.
    """

    pressure: pd.DataFrame
    flowrate: pd.DataFrame
    demand: pd.DataFrame
    leak_demand: pd.DataFrame
    elapsed_seconds: float
    pressure_clean: pd.DataFrame
    flowrate_clean: pd.DataFrame
    quality: pd.DataFrame | None = None
    head: pd.DataFrame | None = None
    link_status: pd.DataFrame | None = None


def run_simulation(
    wn: WaterNetworkModel, use_epanet_for_quality: bool = False
) -> SimulationResults:
    """Execute the simulation and unpack the time-series tables.

    The supplied model is consumed by the simulator; do not call this
    twice on the same model.

    Args:
        wn: The prepared water network model.
        use_epanet_for_quality: When true the scenario runs through the
            ``EpanetSimulator`` so a water quality analysis is produced
            (the default ``WNTRSimulator`` emits no quality table). Only
            valid for scenarios without leaks; the config validator
            enforces that. The whole scenario, including the hydraulics,
            is then solved by EPANET.

    Returns:
        A :class:`SimulationResults`. ``quality`` is populated only on
        the EpanetSimulator path.
    """

    started = time.perf_counter()
    if use_epanet_for_quality:
        simulator = wntr.sim.EpanetSimulator(wn)
        # Run in a private temp directory so the EPANET .inp/.rpt/.bin/.hyd
        # scratch files never land in the working directory or collide
        # between concurrent runs.
        with tempfile.TemporaryDirectory(prefix="wdn_epanet_") as tmp:
            results = simulator.run_sim(file_prefix=os.path.join(tmp, "run"))
    else:
        simulator = wntr.sim.WNTRSimulator(wn)
        results = simulator.run_sim()
    elapsed = time.perf_counter() - started

    # Genuine non-convergence (max trials exceeded or no solution) is
    # reported by the simulator as a non-None error_code: both the
    # WNTRSimulator and the EpanetSimulator reader set
    # ``ResultsStatus.error``, which equals 0 -- so the guard must test
    # ``is not None`` rather than truthiness, or a non-converged run would
    # be silently treated as valid. Benign EPANET warnings (negative
    # pressures under DDA, disconnected nodes) leave error_code at None and
    # still flow through to the validators as warnings rather than erroring
    # here. This catches a solve that returned rows but did not converge,
    # which the zero-timestep guard below would miss.
    error_code = getattr(results, "error_code", None)
    if error_code is not None:
        raise RuntimeError(
            "Simulation did not converge: the solver reported a non-success "
            f"status (error_code={error_code!r}), commonly a max-trials-exceeded "
            "or no-solution state under PDD on an uncalibrated or ill-posed "
            "network. Treat this network as incompatible with the current "
            "simulation settings."
        )

    pressure = results.node["pressure"].copy()
    # A WNTRSimulator run that fails to converge can return a results
    # object with zero reported timesteps instead of raising. Downstream
    # validators (np.nanmin etc.) then crash on the empty array with a
    # cryptic message. Surface it here as a clear, actionable failure so
    # the batch driver records an informative error per scenario.
    if pressure.shape[0] == 0:
        raise RuntimeError(
            "Simulation produced no reported timesteps: the WNTRSimulator "
            "returned an empty result, which indicates the hydraulics did "
            "not converge (commonly an uncalibrated or ill-posed network "
            "under PDD). Treat this network as incompatible with the current "
            "simulation settings."
        )
    flowrate = results.link["flowrate"].copy()
    demand = results.node["demand"].copy()
    if "leak_demand" in results.node:
        leak_demand = results.node["leak_demand"].copy()
    else:
        # Older WNTR releases, no-leak scenarios or the EpanetSimulator
        # path (no leak support): emit a zero frame aligned with the
        # pressure index/columns so downstream code never has to
        # special-case absence.
        leak_demand = pd.DataFrame(0.0, index=pressure.index, columns=pressure.columns)

    quality = results.node["quality"].copy() if "quality" in results.node else None

    # Total head (pressure head + elevation) and per-timestep link status
    # feed the Hazen-Williams head loss validator. Both simulators report
    # them; guard with membership in case a future/older release omits one.
    head = results.node["head"].copy() if "head" in results.node else None
    link_status = results.link["status"].copy() if "status" in results.link else None

    return SimulationResults(
        pressure=pressure,
        flowrate=flowrate,
        demand=demand,
        leak_demand=leak_demand,
        elapsed_seconds=elapsed,
        pressure_clean=pressure.copy(),
        flowrate_clean=flowrate.copy(),
        quality=quality,
        head=head,
        link_status=link_status,
    )
