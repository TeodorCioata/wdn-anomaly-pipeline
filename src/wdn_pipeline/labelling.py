"""Per-timestep labels and per-scenario metadata.

Two complementary artefacts:

- A per-timestep ``label`` column attached to the wide pressure / flow
  DataFrames at output time. ``0`` means normal, ``1`` means anomalous.
  The label is the **union** of every active fault window: leak windows
  from Phase 3 plus sensor-fault windows from Phase 4.
- A sidecar metadata dictionary describing the full scenario (network,
  seed, scenario type, fault parameters, time window). The runner
  serialises this next to each output file.

Per-channel masks are not stored in :class:`Labels` itself; the
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


def _detect_interactions(
    resolved_leaks: Sequence[ResolvedLeak],
    resolved_sensor_faults: Sequence[ResolvedSensorFault],
) -> list[dict]:
    """Record cumulative interactions where a sensor fault sits on a leak.

    An interaction is a sensor fault whose target channel coincides with
    a leak's hydraulic footprint:

    - a ``pressure`` fault on the inserted leak junction, or
    - a ``flowrate`` fault on either segment of the split pipe.

    The result is **informational only** (decision D24 / Week 5 plan):
    it is surfaced in the sidecar metadata so a downstream consumer can
    see that the leak signature and the sensor fault overlap on the same
    channel, but it never gates the pipeline. The structural leak check
    ``leak_demand_active`` reads from the uncorrupted ``leak_demand``
    table, so leak correctness is unaffected by the overlap.
    """

    interactions: list[dict] = []
    for fault in resolved_sensor_faults:
        for leak in resolved_leaks:
            shared: str | None = None
            if fault.quantity == "pressure" and fault.target == leak.leak_node_name:
                shared = "pressure_sensor_on_leak_node"
            elif fault.quantity == "flowrate" and fault.target in (
                leak.pipe,
                leak.new_pipe_name,
            ):
                shared = "flowrate_sensor_on_leaked_pipe"
            if shared is not None:
                interactions.append(
                    {
                        "kind": shared,
                        "leak_node": leak.leak_node_name,
                        "leak_pipe": leak.pipe,
                        "leak_name": leak.name,
                        "sensor_fault_type": fault.type,
                        "sensor_target": fault.target,
                        "sensor_fault_name": fault.name,
                    }
                )
    return interactions


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
            "interactions": _detect_interactions(leaks, sensor_faults),
        },
    }
    return Labels(timestep_labels=labels, metadata=metadata)
