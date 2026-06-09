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
from dataclasses import dataclass
from pathlib import Path

import wntr
from wntr.network import WaterNetworkModel

from wdn_pipeline.config import NetworkConfig, SimulationConfig

warnings.filterwarnings("ignore", category=UserWarning, module="wntr")

# Calibration fields on SimulationConfig that map one-to-one onto a
# ``wn.options.hydraulic`` attribute of the same name (decision D33).
_CALIBRATION_FIELDS = (
    "viscosity",
    "specific_gravity",
    "headloss",
    "accuracy",
    "trials",
    "demand_multiplier",
    "minimum_pressure",
    "required_pressure",
    "pressure_exponent",
)


@dataclass(frozen=True)
class HydraulicOverride:
    """Record of one hydraulic option override, for calibration provenance.

    Attributes:
        option: The ``wn.options.hydraulic`` attribute name.
        inp_value: The value present after loading the .inp (the default
            the override replaced).
        config_value: The value applied from the config.
    """

    option: str
    inp_value: object
    config_value: object

    def to_dict(self) -> dict:
        return {
            "option": self.option,
            "inp_value": self.inp_value,
            "config_value": self.config_value,
        }


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


def apply_hydraulic_options(
    wn: WaterNetworkModel, simulation_cfg: SimulationConfig
) -> list[HydraulicOverride]:
    """Apply calibration overrides onto a loaded model (decision D33).

    Calibration is split out from :func:`load_network` so the override
    provenance can be returned to the runner and recorded in the run
    summary and metadata sidecar. The allowlisted fields map one-to-one
    onto ``wn.options.hydraulic`` attributes; ``extra_hydraulic_options``
    keys are validated against the live options object so a typo raises
    here (the network is not available at config-load time).

    Args:
        wn: A loaded ``WaterNetworkModel``. Mutated in place.
        simulation_cfg: The simulation config carrying the calibration
            fields.

    Returns:
        One :class:`HydraulicOverride` per option actually changed, in a
        stable order (allowlist first, then ``extra_hydraulic_options``
        in declaration order). Fields left at ``None`` are skipped so the
        .inp value is preserved.

    Raises:
        ValueError: If a key in ``extra_hydraulic_options`` is not a real
            ``wn.options.hydraulic`` attribute.
    """

    # WNTRSimulator (used always, decision D3) only implements
    # Hazen-Williams. Reject D-W/C-M headloss overrides loudly rather
    # than letting WNTRSimulator raise a cryptic NotImplementedError mid
    # simulation (decision D34).
    if simulation_cfg.headloss is not None and simulation_cfg.headloss != "H-W":
        raise ValueError(
            f"simulation.headloss='{simulation_cfg.headloss}' is not supported: "
            "the pipeline always uses WNTRSimulator (D3), which only implements "
            "Hazen-Williams ('H-W'). Darcy-Weisbach / Chezy-Manning would require "
            "the EpanetSimulator, which has no leak support (D34)."
        )

    overrides: list[HydraulicOverride] = []
    for field_name in _CALIBRATION_FIELDS:
        value = getattr(simulation_cfg, field_name)
        if value is None:
            continue
        inp_value = getattr(wn.options.hydraulic, field_name)
        setattr(wn.options.hydraulic, field_name, value)
        overrides.append(HydraulicOverride(field_name, inp_value, value))

    for key, value in simulation_cfg.extra_hydraulic_options.items():
        if not hasattr(wn.options.hydraulic, key):
            raise ValueError(
                f"Unknown WNTR hydraulic option '{key}' in "
                f"simulation.extra_hydraulic_options. It is not an attribute "
                f"of wn.options.hydraulic (D33: typos fail loudly)."
            )
        inp_value = getattr(wn.options.hydraulic, key)
        setattr(wn.options.hydraulic, key, value)
        overrides.append(HydraulicOverride(key, inp_value, value))

    return overrides


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
