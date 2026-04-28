"""Network loading and preparation.

A WaterNetworkModel is the mutable hydraulic state used by WNTR. Two
guarantees this module must hold:

1. **State isolation.** Every scenario gets a fresh model. WNTR's
   ``split_pipe`` and pattern-replacement APIs mutate the model in
   place, so reusing a model across scenarios is a recipe for silent
   contamination.
2. **Time settings come from config, not the .inp.** Different .inp
   files ship with wildly different default durations (Hanoi has
   ``duration=0``, Net3 ships with one week). The simulation block of
   the config is authoritative.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import wntr
from wntr.network import WaterNetworkModel

from wdn_pipeline.config import NetworkConfig, SimulationConfig

warnings.filterwarnings("ignore", category=UserWarning, module="wntr")


def load_network(network_cfg: NetworkConfig, simulation_cfg: SimulationConfig) -> WaterNetworkModel:
    """Load a WaterNetworkModel and apply simulation timing.

    The function returns a freshly constructed model every call. Callers
    must not cache or share the returned object across scenarios.

    Args:
        network_cfg: Network specification (path or bundled name).
        simulation_cfg: Simulation timing and demand model.

    Returns:
        A WNTR ``WaterNetworkModel`` with the time and hydraulic options
        from ``simulation_cfg`` applied.

    Raises:
        FileNotFoundError: If ``inp_path`` is neither a bundled WNTR
            network name nor a readable file path.
    """

    spec = network_cfg.inp_path
    path = Path(spec)
    if path.is_file():
        wn = wntr.network.WaterNetworkModel(str(path))
    else:
        # Try WNTR's bundled library; raises FileNotFoundError on miss.
        try:
            wn = wntr.network.WaterNetworkModel(spec)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not resolve network '{spec}' as a path or a bundled WNTR network. "
                f"Bundled names are case-sensitive (e.g. 'Net3')."
            ) from e

    wn.options.time.duration = simulation_cfg.duration_seconds
    wn.options.time.hydraulic_timestep = simulation_cfg.hydraulic_timestep_seconds
    wn.options.time.report_timestep = simulation_cfg.report_timestep_seconds
    if simulation_cfg.pattern_timestep_seconds is not None:
        wn.options.time.pattern_timestep = simulation_cfg.pattern_timestep_seconds
    wn.options.hydraulic.demand_model = simulation_cfg.demand_model
    return wn


def derive_network_name(network_cfg: NetworkConfig) -> str:
    """Return a filename-safe identifier for the network.

    Uses ``network_cfg.name`` when set, otherwise the ``.inp`` filename
    stem (or the bundled name as-is). Spaces are replaced with
    underscores.
    """

    if network_cfg.name:
        return network_cfg.name.replace(" ", "_")
    spec = network_cfg.inp_path
    path = Path(spec)
    stem = path.stem if path.suffix.lower() == ".inp" else spec
    return stem.replace(" ", "_")
