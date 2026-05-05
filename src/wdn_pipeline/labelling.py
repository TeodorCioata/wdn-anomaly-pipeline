"""Per-timestep labels and per-scenario metadata.

Two complementary artefacts (decision D10):

- A per-timestep ``label`` column attached to the wide pressure / flow
  DataFrames at output time. ``0`` means normal, ``1`` means anomalous.
  Current phase fills this in from leak windows; Next phase will extend it for
  sensor-fault windows.
- A sidecar metadata dictionary describing the full scenario (network,
  seed, scenario type, fault parameters, time window). The runner
  serialises this next to each output file.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.faults.leak import ResolvedLeak


@dataclass(frozen=True)
class Labels:
    """Per-timestep and scenario-level labels."""

    timestep_labels: pd.Series  # int 0/1 indexed by time-in-seconds
    metadata: dict


def _leak_label_series(
    time_index: pd.Index, resolved_leaks: Sequence[ResolvedLeak]
) -> pd.Series:
    labels = pd.Series(0, index=time_index, name="label", dtype="int8")
    if not resolved_leaks:
        return labels
    times = labels.index.to_numpy()
    for leak in resolved_leaks:
        mask = (times >= leak.start_time_seconds) & (times <= leak.end_time_seconds)
        labels.loc[mask] = 1
    return labels


def build_labels(
    config: PipelineConfig,
    time_index: pd.Index,
    network_name: str,
    resolved_leaks: Sequence[ResolvedLeak] | None = None,
) -> Labels:
    """Build the per-timestep label series and the scenario metadata dict.

    Args:
        config: The full pipeline configuration.
        time_index: The simulation time index (seconds since start).
        network_name: Filename-safe network identifier.
        resolved_leaks: Concrete leak parameters realised by the leak
            injector. ``None`` (or empty) means a no-leak run.

    Returns:
        :class:`Labels` with both artefacts. ``timestep_labels`` is ``1``
        at every timestep covered by at least one leak window and ``0``
        elsewhere; ``metadata`` includes the resolved leaks alongside
        the original config summary.
    """

    leaks = list(resolved_leaks) if resolved_leaks else []
    labels = _leak_label_series(time_index, leaks)

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
            "leaks": [leak.to_dict() for leak in leaks],
            "sensor_faults": list(config.faults.sensor_faults),
        },
    }
    return Labels(timestep_labels=labels, metadata=metadata)
