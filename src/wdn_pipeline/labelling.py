"""Per-timestep labels and per-scenario metadata.

Two complementary artefacts (decision D10):

- A per-timestep ``label`` column attached to the wide pressure / flow
  DataFrames at output time. ``0`` means normal, ``1`` means anomalous.
  Phase 3 and 4 will populate this from leak / sensor-fault windows.
- A sidecar metadata dictionary describing the full scenario (network,
  seed, scenario type, fault parameters, time window). The runner
  serialises this next to each output file.

For Phase 2 every scenario is normal so ``label`` is all zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wdn_pipeline.config import PipelineConfig


@dataclass(frozen=True)
class Labels:
    """Per-timestep and scenario-level labels."""

    timestep_labels: pd.Series  # int 0/1 indexed by time-in-seconds
    metadata: dict


def build_labels(config: PipelineConfig, time_index: pd.Index, network_name: str) -> Labels:
    """Build the per-timestep label series and the scenario metadata dict.

    Args:
        config: The full pipeline configuration.
        time_index: The simulation time index (seconds since start).
        network_name: Filename-safe network identifier.

    Returns:
        :class:`Labels` with both artefacts. For Phase 2 normal
        scenarios ``timestep_labels`` is all zeros and ``metadata``
        captures the config plus the network name.
    """

    labels = pd.Series(0, index=time_index, name="label", dtype="int8")

    metadata = {
        "scenario_type": config.scenario.type,
        "scenario_label": config.scenario.label,
        "network_name": network_name,
        "network_inp": config.network.inp_path,
        "seed": config.seed,
        "duration_seconds": config.simulation.duration_seconds,
        "hydraulic_timestep_seconds": config.simulation.hydraulic_timestep_seconds,
        "report_timestep_seconds": config.simulation.report_timestep_seconds,
        "demand_model": config.simulation.demand_model,
        "demand_mode": config.demand.mode,
        "fault_summary": {
            "leaks": list(config.faults.leaks),
            "sensor_faults": list(config.faults.sensor_faults),
        },
    }
    return Labels(timestep_labels=labels, metadata=metadata)
