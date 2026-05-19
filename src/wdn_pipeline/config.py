"""Pydantic v2 schema for scenario configuration files.

The pipeline is driven by YAML configs that describe a single scenario:
which network to load, simulation timing, demand-pattern strategy, the
random seed, fault injections (Phase 3 leaks, Phase 4 sensor faults)
and output settings. This module loads, validates and exposes those
settings as immutable typed objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    """Base for all config models. Immutable, strict, forbids extra keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=False)


class NetworkConfig(_Frozen):
    """Network specification.

    Attributes:
        inp_path: Either a bundled WNTR network name (e.g. ``Net3``) or a
            filesystem path to an EPANET ``.inp`` file. WNTR resolves bare
            names against its built-in network library.
        name: Short, filename-safe identifier for this network. Used in
            output filenames and metadata. If omitted it is derived from
            ``inp_path``.
    """

    inp_path: str
    name: str | None = None


class SimulationConfig(_Frozen):
    """Simulation timing and hydraulic options.

    Attributes:
        duration_seconds: Total simulation length.
        hydraulic_timestep_seconds: Solver step size.
        report_timestep_seconds: Step size at which results are reported.
            Must be a multiple of the hydraulic timestep.
        pattern_timestep_seconds: Step size at which demand patterns
            advance. If ``None`` the .inp default is kept.
        demand_model: ``DDA`` (demand-driven, the EPANET default) or
            ``PDD`` (pressure-dependent, required when leaks are present).
    """

    duration_seconds: Annotated[int, Field(gt=0)]
    hydraulic_timestep_seconds: Annotated[int, Field(gt=0)] = 3600
    report_timestep_seconds: Annotated[int, Field(gt=0)] = 3600
    pattern_timestep_seconds: Annotated[int, Field(gt=0)] | None = None
    demand_model: Literal["DDA", "PDD"] = "DDA"

    @model_validator(mode="after")
    def _validate_timesteps(self) -> SimulationConfig:
        if self.report_timestep_seconds % self.hydraulic_timestep_seconds != 0:
            raise ValueError(
                "report_timestep_seconds must be a multiple of hydraulic_timestep_seconds"
            )
        if self.duration_seconds % self.hydraulic_timestep_seconds != 0:
            raise ValueError(
                "duration_seconds must be a multiple of hydraulic_timestep_seconds"
            )
        return self


class ScaleDemandParams(_Frozen):
    """Parameters for the ``scale`` demand mode.

    Multiplies every junction's base demand by ``multiplier``.
    """

    multiplier: Annotated[float, Field(gt=0)] = 1.0


class FourierDemandParams(_Frozen):
    """Parameters for the ``fourier`` demand mode.

    The pattern multiplier at time ``t`` is::

        m(t) = base + amplitude * sin(2*pi*t / period + phase) + N(0, noise_std)

    The pattern is applied to every junction, replacing whatever pattern
    is in the .inp file. A new pattern named ``synthetic_fourier`` is
    added to the network.
    """

    base: Annotated[float, Field(gt=0)] = 1.0
    amplitude: Annotated[float, Field(ge=0)] = 0.3
    period_hours: Annotated[float, Field(gt=0)] = 24.0
    phase_shift_hours: float = 0.0
    noise_std: Annotated[float, Field(ge=0)] = 0.0


class DemandConfig(_Frozen):
    """Demand pattern strategy.

    ``mode`` selects one of:

    - ``default``: leave the .inp patterns untouched.
    - ``scale``: scale every junction base demand by ``scale.multiplier``.
    - ``fourier``: replace patterns with a synthetic diurnal pattern.
    """

    mode: Literal["default", "scale", "fourier"] = "default"
    scale: ScaleDemandParams = ScaleDemandParams()
    fourier: FourierDemandParams = FourierDemandParams()


class LeakSpec(_Frozen):
    """Specification for a single pipe leak.

    Per-leak validation enforces the
    geometry mutual exclusion (area xor diameter), value ranges and
    timing ordering. Cross-config invariants (PDD requirement, leak
    window inside the simulation duration) live on
    :class:`PipelineConfig`.

    Attributes:
        pipe: Name of the pipe to split, or ``None`` to pick at random.
            Random selection is reproducible: it consumes the scenario
            ``seed`` via the RNG threaded through the runner.
        split_fraction: Position along the pipe (0 = upstream, 1 =
            downstream) at which the new junction is inserted, or
            ``None`` for a random fraction in (0, 1).
        area_m2: Leak hole area in square metres. Mutually exclusive with
            ``diameter_m``.
        diameter_m: Leak hole diameter in metres. Mutually exclusive with
            ``area_m2``. Internally converted to area via
            ``area = pi * (d/2)**2``.
        discharge_coeff: Leak discharge coefficient ``Cd`` in the WNTR
            leak equation ``Q = Cd * A * sqrt(2*g*h)``. Default 0.75
            mirrors the WNTR default and the LeakG3PD convention.
        start_time_seconds: Simulation time at which the leak switches
            on.
        end_time_seconds: Simulation time at which the leak switches off.
            Must be strictly greater than ``start_time_seconds`` and
            no greater than ``simulation.duration_seconds``.
        profile: ``abrupt`` (instantaneous full-area leak from start to
            end), ``linear`` (linearly growing area, rendered as a
            step-function approximation, default Phase 3 incipient
            profile) or ``step`` (mathematically equivalent stepwise
            growth; user-facing label for staircase-style incipient
            leaks).
        profile_steps: Step count for the linear/step approximation.
            Ignored for abrupt leaks. Default 30.
        name: Optional human-readable label for the leak. Surfaced in
            the resolved metadata to make multi-leak scenarios easier
            to read.
    """

    pipe: str | None = None
    split_fraction: float | None = None
    area_m2: float | None = None
    diameter_m: float | None = None
    discharge_coeff: float = 0.75
    start_time_seconds: Annotated[int, Field(ge=0)]
    end_time_seconds: Annotated[int, Field(gt=0)]
    profile: Literal["abrupt", "linear", "step"] = "abrupt"
    profile_steps: Annotated[int, Field(ge=2)] = 30
    name: str | None = None

    @model_validator(mode="after")
    def _validate_leak(self) -> LeakSpec:
        if (self.area_m2 is None) == (self.diameter_m is None):
            raise ValueError(
                "Exactly one of 'area_m2' or 'diameter_m' must be set per leak"
            )
        if self.area_m2 is not None and self.area_m2 <= 0:
            raise ValueError("area_m2 must be > 0")
        if self.diameter_m is not None and self.diameter_m <= 0:
            raise ValueError("diameter_m must be > 0")
        if self.split_fraction is not None and not (0.0 <= self.split_fraction <= 1.0):
            raise ValueError("split_fraction must be in [0, 1]")
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be > start_time_seconds")
        if not (0.0 < self.discharge_coeff <= 1.0):
            raise ValueError("discharge_coeff must be in (0, 1]")
        return self


class SensorFaultSpec(_Frozen):
    """Specification for a single sensor fault (Phase 4, D18-D25).

    Sensor faults are applied **post-simulation** (D5): they corrupt the
    reported pressure or flowrate at a single channel without modifying
    the hydraulic model. Six fault types are supported, discriminated
    by ``type``. Each fault uses the half-open window convention
    ``[start_time_seconds, end_time_seconds)`` consistent with Phase 3
    leaks.

    Attributes:
        type: One of ``bias`` / ``drift`` / ``stuck`` / ``dropout`` /
            ``noise`` / ``gain``. Acts as the type discriminator for
            downstream dispatch in :mod:`wdn_pipeline.faults.sensor`.
        quantity: Which output table to corrupt. ``pressure`` targets
            a junction; ``flowrate`` targets a pipe.
        target: Channel name (junction for pressure, link for flowrate)
            or ``None`` to draw a target from the scenario RNG (D19).
            The injector validates the resolved name against the loaded
            network; config-load time does not see the network yet.
        start_time_seconds: Fault switches on at this time (inclusive).
        end_time_seconds: Fault switches off at this time (exclusive).
            Strictly greater than ``start_time_seconds`` and no greater
            than ``simulation.duration_seconds``.
        name: Optional human-readable label, surfaced in resolved
            metadata.
        bias_value: Constant offset for the ``bias`` fault. Non-zero
            (a zero bias is a no-op and is rejected as a likely config
            mistake).
        slope_per_second: Linear drift rate for the ``drift`` fault, in
            units of the quantity per second. Sign determines drift
            direction.
        intervals: For the ``dropout`` fault, a list of
            ``[start, end)`` sub-intervals within the outer fault
            window. Each interval is independently validated.
        fill_value: For the ``dropout`` fault, the value written into
            corrupted samples. ``None`` (default) means NaN, per D21.
        sigma: Standard deviation of additive Gaussian noise for the
            ``noise`` fault. Must be strictly positive.
        rng_offset: Optional per-fault offset for the noise RNG. Lets
            two ``noise`` faults in the same scenario draw independent
            sample streams without changing the scenario seed.
        gain_factor: Multiplicative factor for the ``gain`` fault
            (``y'(t) = gain_factor * y(t)``). A factor of ``1.0`` is a
            no-op and ``0.0`` zeroes the signal; both are rejected as
            likely config mistakes. Detectability of small gains is
            checked at runtime (D27).
    """

    type: Literal["bias", "drift", "stuck", "dropout", "noise", "gain"]
    quantity: Literal["pressure", "flowrate"] = "pressure"
    target: str | None = None
    start_time_seconds: Annotated[int, Field(ge=0)]
    end_time_seconds: Annotated[int, Field(gt=0)]
    name: str | None = None

    # Per-type fields. All optional at the schema level; the
    # _validate_sensor_fault model_validator enforces presence/absence
    # based on ``type`` so the YAML stays simple to write.
    bias_value: float | None = None
    slope_per_second: float | None = None
    intervals: list[tuple[int, int]] | None = None
    fill_value: float | None = None
    sigma: float | None = None
    rng_offset: int = 0
    gain_factor: float | None = None

    @model_validator(mode="after")
    def _validate_sensor_fault(self) -> SensorFaultSpec:
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be > start_time_seconds")

        if self.type == "bias":
            if self.bias_value is None:
                raise ValueError("bias fault requires 'bias_value'")
            if self.bias_value == 0.0:
                raise ValueError(
                    "bias_value must be non-zero (a zero bias is a no-op)"
                )
        elif self.type == "drift":
            if self.slope_per_second is None:
                raise ValueError("drift fault requires 'slope_per_second'")
        elif self.type == "stuck":
            # No extra fields.
            pass
        elif self.type == "dropout":
            if self.intervals is None or not self.intervals:
                raise ValueError(
                    "dropout fault requires a non-empty 'intervals' list"
                )
            for i, iv in enumerate(self.intervals):
                if len(iv) != 2:
                    raise ValueError(
                        f"dropout intervals[{i}] must be a (start, end) pair"
                    )
                a, b = int(iv[0]), int(iv[1])
                if b <= a:
                    raise ValueError(
                        f"dropout intervals[{i}] end ({b}) must be > start ({a})"
                    )
                if a < self.start_time_seconds or b > self.end_time_seconds:
                    raise ValueError(
                        f"dropout intervals[{i}] [{a}, {b}) is outside the fault "
                        f"window [{self.start_time_seconds}, {self.end_time_seconds})"
                    )
        elif self.type == "noise":
            if self.sigma is None:
                raise ValueError("noise fault requires 'sigma'")
            if self.sigma <= 0:
                raise ValueError("sigma must be > 0")
        elif self.type == "gain":
            if self.gain_factor is None:
                raise ValueError("gain fault requires 'gain_factor'")
            if self.gain_factor == 1.0:
                raise ValueError(
                    "gain_factor must not be 1.0 (a unit gain is a no-op)"
                )
            if self.gain_factor == 0.0:
                raise ValueError(
                    "gain_factor must not be 0.0 (a zero gain zeroes the signal)"
                )

        return self


class FaultsConfig(_Frozen):
    """Fault injection plan for one scenario.

    ``leaks`` is a list of typed :class:`LeakSpec` (Phase 3). Concurrent
    leaks are supported: each LeakSpec is independently realised on a
    fresh node inserted via ``wntr.morph.split_pipe``. ``sensor_faults``
    is a list of typed :class:`SensorFaultSpec` (Phase 4); concurrent
    faults across distinct channels are supported.
    """

    leaks: list[LeakSpec] = Field(default_factory=list)
    sensor_faults: list[SensorFaultSpec] = Field(default_factory=list)


class ScenarioConfig(_Frozen):
    """Identifies what kind of scenario this is.

    ``label`` flows into both output filenames and the per-timestep label
    column. ``type`` is informational; the actual label values come from
    the resolved fault windows.
    """

    type: Literal["normal", "leak", "sensor_fault", "cumulative"] = "normal"
    label: str = "normal"


class OutputConfig(_Frozen):
    """Where and how to write outputs.

    Attributes:
        directory: Output root directory; created on demand.
        formats: One or more of ``parquet`` / ``csv``. The writer registry
            in :mod:`wdn_pipeline.output` is open to additional backends
            (e.g. DuckDB) without changes to this schema.
        write_metadata_sidecar: When true a ``.meta.yaml`` file is
            written next to each data file describing the scenario.
        remove_leak_nodes: When true a post-processing step (D28) reverts
            the leak-node split artefacts before serialisation: leak-node
            columns are dropped and split pipe segments are collapsed
            back to the original pipe name so the output schema matches
            the original network. Default false (explicit opt-in). The
            ``leak_demand`` diagnostic table is never affected.
    """

    directory: Path = Path("outputs")
    formats: list[Literal["parquet", "csv"]] = Field(default_factory=lambda: ["parquet", "csv"])
    write_metadata_sidecar: bool = True
    remove_leak_nodes: bool = False

    @model_validator(mode="after")
    def _validate_formats(self) -> OutputConfig:
        if not self.formats:
            raise ValueError("output.formats must list at least one format")
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("output.formats must not contain duplicates")
        return self


class ValidationConfig(_Frozen):
    """Tolerances for the physics-based validation checks.

    Validation reports three severities (decision D12):

    - ``ok``: value in the strict-OK range.
    - ``warning``: value drifts outside the strict range but stays
      within the configured tolerance band. Logged but does not fail
      the pipeline. Useful for known physical artefacts (e.g. Net3
      node "10" under both WNTRSimulator and the EpanetSimulator
      reference produces ~-0.66 m).
    - ``fail``: value exceeds the tolerance band. Pipeline returns
      a non-zero exit code.

    Attributes:
        pressure_min_m: Strict lower bound for "no warning".
        pressure_min_warning_tolerance_m: How far below
            ``pressure_min_m`` is still tolerated as a warning. Below
            (``pressure_min_m - tolerance``) is a hard failure.
        pressure_max_m: Strict upper bound. Above is a hard failure.
        mass_balance_tol_m3s: Maximum allowed absolute residual
            (m^3/s) for net inflow minus demand at any junction.
            Phase 3 subtracts ``leak_demand`` from the demand side so
            this tolerance also applies to leak scenarios.
        leak_pressure_drop_min_m: Minimum mean pressure drop
            (baseline minus during-leak) at the leak junction needed
            to register as a clear pressure response. If the observed
            drop is smaller the leak-pressure-drop check fires a
            warning (never a hard failure).
        gain_detectability_min_ratio: Minimum ratio of the gain-fault
            residual standard deviation to the clean-signal standard
            deviation for the fault to be considered detectable (D27).
            Below this ratio the ``sensor_fault_signal_applied`` check
            emits a warning (never a hard failure): a small gain on a
            low-variance channel is statistically indistinguishable
            from normal sensor noise and would mislabel ML training
            data.
    """

    pressure_min_m: float = 0.0
    pressure_min_warning_tolerance_m: Annotated[float, Field(ge=0)] = 1.0
    pressure_max_m: Annotated[float, Field(gt=0)] = 150.0
    mass_balance_tol_m3s: Annotated[float, Field(gt=0)] = 1e-3
    leak_pressure_drop_min_m: Annotated[float, Field(ge=0)] = 0.01
    gain_detectability_min_ratio: Annotated[float, Field(ge=0)] = 0.05


class PipelineConfig(_Frozen):
    """Top-level pipeline configuration.

    Maps directly onto the YAML structure consumed by
    :func:`wdn_pipeline.runner.run_from_config_file`.
    """

    network: NetworkConfig
    simulation: SimulationConfig
    seed: int = 0
    demand: DemandConfig = DemandConfig()
    faults: FaultsConfig = FaultsConfig()
    scenario: ScenarioConfig = ScenarioConfig()
    validation: ValidationConfig = ValidationConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def _validate_leak_invariants(self) -> PipelineConfig:
        if self.faults.leaks:
            if self.simulation.demand_model != "PDD":
                raise ValueError(
                    "simulation.demand_model must be 'PDD' when faults.leaks is non-empty "
                    "(D16: explicit PDD requirement, no silent auto-promotion)"
                )
            for i, spec in enumerate(self.faults.leaks):
                if spec.end_time_seconds > self.simulation.duration_seconds:
                    raise ValueError(
                        f"faults.leaks[{i}].end_time_seconds ({spec.end_time_seconds}) "
                        f"exceeds simulation.duration_seconds ({self.simulation.duration_seconds})"
                    )
        for i, spec in enumerate(self.faults.sensor_faults):
            if spec.end_time_seconds > self.simulation.duration_seconds:
                raise ValueError(
                    f"faults.sensor_faults[{i}].end_time_seconds ({spec.end_time_seconds}) "
                    f"exceeds simulation.duration_seconds ({self.simulation.duration_seconds})"
                )
        return self


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline config.

    Args:
        path: Filesystem path to a YAML file.

    Returns:
        A validated :class:`PipelineConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the YAML does not match the schema.
    """

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return PipelineConfig.model_validate(raw)
