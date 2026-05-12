"""Per-timestep labels and per-scenario metadata.

Two complementary artefacts (decision D10):

- A per-timestep ``label`` column attached to the wide pressure / flow
  DataFrames at output time. ``0`` means normal, ``1`` means anomalous.
  The label is the **union** of every active fault window: leak windows
  from Phase 3 plus sensor-fault windows from Phase 4.
- A sidecar metadata dictionary describing the full scenario (network,
  seed, scenario type, fault parameters, time window). The runner
  serialises this next to each output file.

Per-channel masks (D22) are not stored in :class:`Labels` itself; the
runner attaches them directly to the corresponding output table via
:mod:`wdn_pipeline.output` to keep this module's responsibilities
narrow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from wdn_pipeline.config import PipelineConfig
from wdn_pipeline.faults.leak import ResolvedLeak
from wdn_pipeline.faults.sensor import ResolvedSensorFault


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
        # Half-open window [start, end). WNTR's end control flips
        # leak_status to False at end_time_seconds, so the reported
        # leak_demand at that exact timestep is already zero. Including
        # end_time_seconds in the label window would mark a normal frame
        # as anomalous and disagree with leak_demand at every leak end.
        mask = (times >= leak.start_time_seconds) & (times < leak.end_time_seconds)
        labels.loc[mask] = 1
    return labels


def _apply_sensor_fault_windows(
    labels: pd.Series, resolved_sensor_faults: Sequence[ResolvedSensorFault]
) -> pd.Series:
    if not resolved_sensor_faults:
        return labels
    times = labels.index.to_numpy()
    for fault in resolved_sensor_faults:
        # Same half-open convention as leaks.
        mask = (times >= fault.start_time_seconds) & (times < fault.end_time_seconds)
        labels.loc[mask] = 1
    return labels


def build_labels(
    config: PipelineConfig,
    time_index: pd.Index,
    network_name: str,
    resolved_leaks: Sequence[ResolvedLeak] | None = None,
    resolved_sensor_faults: Sequence[ResolvedSensorFault] | None = None,
) -> Labels:
    """Build the per-timestep label series and the scenario metadata dict.

    Args:
        config: The full pipeline configuration.
        time_index: The simulation time index (seconds since start).
        network_name: Filename-safe network identifier.
        resolved_leaks: Concrete leak parameters realised by the leak
            injector. ``None`` (or empty) means no leaks ran.
        resolved_sensor_faults: Concrete sensor-fault parameters from
            the sensor injector. ``None`` (or empty) means no sensor
            faults ran.

    Returns:
        :class:`Labels` with both artefacts. ``timestep_labels`` is
        ``1`` at every timestep covered by at least one leak or sensor
        fault window and ``0`` elsewhere; ``metadata`` includes the
        resolved leaks and sensor faults alongside the original config
        summary.
    """

    leaks = list(resolved_leaks) if resolved_leaks else []
    sensor_faults = (
        list(resolved_sensor_faults) if resolved_sensor_faults else []
    )
    labels = _leak_label_series(time_index, leaks)
    labels = _apply_sensor_fault_windows(labels, sensor_faults)

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
            "sensor_faults": [fault.to_dict() for fault in sensor_faults],
        },
    }
    return Labels(timestep_labels=labels, metadata=metadata)
