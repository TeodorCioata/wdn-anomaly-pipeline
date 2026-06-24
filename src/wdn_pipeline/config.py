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


class QualitySourceSpec(_Frozen):
    """A water quality source injected at a node (chemical quality only).

    Maps onto ``wn.add_source(name, node, source_type, strength, pattern)``.
    Sources are only meaningful for ``parameter: chemical``; the
    :class:`WaterQualityConfig` validator rejects them under ``age`` /
    ``trace`` / ``none``.

    Attributes:
        node: Injection node name. Validated against the loaded network
            at network preparation (the network is not available at
            config-load time).
        source_type: EPANET source type. ``CONCEN`` sets the
            concentration of external inflow, ``MASS`` a mass injection
            rate, ``FLOWPACED`` a concentration added to each unit of
            inflow and ``SETPOINT`` a fixed downstream concentration.
        strength: Source strength. Units depend on ``source_type``
            (mass/volume for CONCEN/FLOWPACED/SETPOINT, mass/time for
            MASS).
        pattern: Optional list of time multipliers applied to ``strength``
            (registered as a network pattern and attached to the source).
            ``None`` means a constant source.
    """

    node: str
    source_type: Literal["CONCEN", "MASS", "FLOWPACED", "SETPOINT"] = "CONCEN"
    strength: Annotated[float, Field(ge=0)]
    pattern: list[Annotated[float, Field(ge=0)]] | None = None


class WaterQualityConfig(_Frozen):
    """Water quality analysis options.

    Exposes WNTR's ``wn.options.quality`` / ``wn.options.reaction``
    surface so a scenario can run a chemical-decay, water-age or
    source-trace analysis. Water quality runs through the
    ``EpanetSimulator`` (the ``WNTRSimulator`` emits no quality table),
    so it is supported only for scenarios without leaks; the
    :class:`PipelineConfig` validator rejects a leak + quality
    combination. Quality settings are applied in
    :func:`wdn_pipeline.network.apply_quality_options` and logged into
    the run summary and metadata sidecar for provenance.

    All fields default to a no-quality state (``parameter: none``),
    preserving the existing pipeline behaviour exactly.

    Attributes:
        parameter: ``none`` (default, no quality analysis), ``chemical``
            (track a reactive species), ``age`` (water age in seconds)
            or ``trace`` (percentage of flow originating at
            ``trace_node``).
        trace_node: Source node for a ``trace`` analysis. Required for
            ``trace`` and rejected otherwise.
        chemical_name: Display name of the tracked species (``chemical``
            only).
        diffusivity: Molecular diffusivity relative to chlorine in water
            (``chemical`` only, EPANET default ``1.0``).
        bulk_coeff: Bulk reaction coefficient in WNTR units (per second;
            WNTR converts to/from EPANET's 1/day on I/O). Negative for
            decay. ``chemical`` only.
        wall_coeff: Wall reaction coefficient in WNTR units (per second).
            ``chemical`` only.
        sources: Chemical injection sources. ``chemical`` only.
        quality_timestep_seconds: Water quality timestep. If ``None`` the
            .inp default (300 s) is kept. Invalid when ``parameter`` is
            ``none``.
    """

    parameter: Literal["none", "chemical", "age", "trace"] = "none"
    trace_node: str | None = None
    chemical_name: str | None = None
    diffusivity: Annotated[float, Field(gt=0)] | None = None
    bulk_coeff: float | None = None
    wall_coeff: float | None = None
    sources: list[QualitySourceSpec] = Field(default_factory=list)
    quality_timestep_seconds: Annotated[int, Field(gt=0)] | None = None

    @property
    def enabled(self) -> bool:
        """True when a quality analysis is requested."""

        return self.parameter != "none"

    @model_validator(mode="after")
    def _validate_quality(self) -> WaterQualityConfig:
        chemical_only = {
            "chemical_name": self.chemical_name,
            "diffusivity": self.diffusivity,
            "bulk_coeff": self.bulk_coeff,
            "wall_coeff": self.wall_coeff,
        }
        set_chemical = [k for k, v in chemical_only.items() if v is not None]
        if self.sources:
            set_chemical.append("sources")

        if self.parameter == "none":
            offenders = list(set_chemical)
            if self.trace_node is not None:
                offenders.append("trace_node")
            if self.quality_timestep_seconds is not None:
                offenders.append("quality_timestep_seconds")
            if offenders:
                raise ValueError(
                    "water quality fields "
                    f"{sorted(offenders)} require simulation.quality.parameter "
                    "to be set (chemical/age/trace), but it is 'none'"
                )
        elif self.parameter == "trace":
            if self.trace_node is None:
                raise ValueError("simulation.quality.parameter='trace' requires 'trace_node'")
            if set_chemical:
                raise ValueError(
                    f"chemical-only fields {sorted(set_chemical)} are not valid "
                    "for a 'trace' analysis"
                )
        elif self.parameter == "age":
            offenders = list(set_chemical)
            if self.trace_node is not None:
                offenders.append("trace_node")
            if offenders:
                raise ValueError(
                    f"fields {sorted(offenders)} are not valid for an 'age' "
                    "analysis (age tracks residence time, no species or source)"
                )
        elif self.parameter == "chemical":
            if self.trace_node is not None:
                raise ValueError(
                    "'trace_node' is only valid for a 'trace' analysis, not 'chemical'"
                )
        return self


class SimulationConfig(_Frozen):
    """Simulation timing and hydraulic options.

    The calibration fields expose WNTR's
    ``wn.options.hydraulic`` surface so simulations can be calibrated
    per network. Every calibration field defaults to ``None`` meaning
    "leave the .inp value untouched" — important because some LeakG3PD
    networks ship their own calibrated values. Overrides are applied in
    :func:`wdn_pipeline.network.apply_hydraulic_options` and logged into
    the run summary and metadata sidecar for provenance.

    Attributes:
        duration_seconds: Total simulation length.
        hydraulic_timestep_seconds: Solver step size.
        report_timestep_seconds: Step size at which results are reported.
            Must be a multiple of the hydraulic timestep.
        pattern_timestep_seconds: Step size at which demand patterns
            advance. If ``None`` the .inp default is kept.
        demand_model: ``DDA`` (demand-driven, the EPANET default) or
            ``PDD`` (pressure-dependent, required when leaks are present).
        viscosity: Kinematic viscosity relative to water at 20 C
            (WNTR/EPANET default ``1.0``). Affects Darcy-Weisbach
            headloss.
        specific_gravity: Fluid specific gravity relative to water
            (default ``1.0``).
        headloss: Headloss formula: ``H-W`` (Hazen-Williams), ``D-W``
            (Darcy-Weisbach) or ``C-M`` (Chezy-Manning).
        accuracy: Solver convergence accuracy (default ``0.001``).
        trials: Maximum solver trials per timestep (default ``40``).
        demand_multiplier: Global multiplier applied to all demands
            (default ``1.0``).
        minimum_pressure: PDD lower pressure bound in metres below which
            demand is zero (default ``0.0``). PDD only.
        required_pressure: PDD pressure in metres at and above which full
            demand is delivered (default ``0.07``). PDD only.
        pressure_exponent: PDD demand curve exponent (default ``0.5``).
            PDD only.
        extra_hydraulic_options: Escape hatch for WNTR hydraulic options
            not in the allowlist above. Each key is applied with
            ``setattr`` only after verifying it exists on
            ``wn.options.hydraulic``; unknown keys raise at network prep,
            so typos fail loudly rather than being silently ignored.
        quality: Water quality analysis options. Defaults to a
            no-quality state. When enabled the scenario runs through the
            ``EpanetSimulator`` and must not have leaks.
    """

    duration_seconds: Annotated[int, Field(gt=0)]
    hydraulic_timestep_seconds: Annotated[int, Field(gt=0)] = 3600
    report_timestep_seconds: Annotated[int, Field(gt=0)] = 3600
    pattern_timestep_seconds: Annotated[int, Field(gt=0)] | None = None
    demand_model: Literal["DDA", "PDD"] = "DDA"

    # --- Hydraulic calibration allowlist. None = untouched. ---
    viscosity: Annotated[float, Field(gt=0)] | None = None
    specific_gravity: Annotated[float, Field(gt=0)] | None = None
    headloss: Literal["H-W", "D-W", "C-M"] | None = None
    accuracy: Annotated[float, Field(gt=0)] | None = None
    trials: Annotated[int, Field(gt=0)] | None = None
    demand_multiplier: Annotated[float, Field(gt=0)] | None = None
    minimum_pressure: float | None = None
    required_pressure: Annotated[float, Field(gt=0)] | None = None
    pressure_exponent: Annotated[float, Field(gt=0)] | None = None
    extra_hydraulic_options: dict[str, float | int | str] = Field(default_factory=dict)

    # --- Water quality (closeout pass). Default = no quality analysis. ---
    quality: WaterQualityConfig = WaterQualityConfig()

    @model_validator(mode="after")
    def _validate_timesteps(self) -> SimulationConfig:
        if self.report_timestep_seconds % self.hydraulic_timestep_seconds != 0:
            raise ValueError(
                "report_timestep_seconds must be a multiple of hydraulic_timestep_seconds"
            )
        if self.duration_seconds % self.hydraulic_timestep_seconds != 0:
            raise ValueError("duration_seconds must be a multiple of hydraulic_timestep_seconds")
        # PDD-only calibration fields: setting any under DDA is a likely
        # config mistake (EPANET silently ignores them otherwise).
        if self.demand_model != "PDD":
            for field_name in ("minimum_pressure", "required_pressure", "pressure_exponent"):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"simulation.{field_name} is a pressure-dependent (PDD) "
                        f"option and must not be set when demand_model is "
                        f"'{self.demand_model}'. Set demand_model: PDD or remove it."
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
            raise ValueError("Exactly one of 'area_m2' or 'diameter_m' must be set per leak")
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


class _SensorFaultBase(_Frozen):
    """Fields shared by every sensor fault type (Phase 4).

    Sensor faults are applied **post-simulation**: they corrupt the
    reported pressure or flowrate at a single channel without modifying
    the hydraulic model. Six fault types are supported as a discriminated
    union (:data:`SensorFaultSpec`), each a concrete subclass declaring
    only its own parameters and keyed on the ``type`` discriminator.
    Every fault uses the half-open window convention
    ``[start_time_seconds, end_time_seconds)`` consistent with Phase 3
    leaks.

    Attributes:
        quantity: Which output table to corrupt. ``pressure`` targets a
            junction; ``flowrate`` targets a pipe.
        target: Channel name (junction for pressure, link for flowrate)
            or ``None`` to draw a target from the scenario RNG. The
            injector validates the resolved name against the loaded
            network; config-load time does not see the network yet.
        start_time_seconds: Fault switches on at this time (inclusive).
        end_time_seconds: Fault switches off at this time (exclusive).
            Strictly greater than ``start_time_seconds`` and no greater
            than ``simulation.duration_seconds``.
        name: Optional human-readable label, surfaced in resolved
            metadata.
    """

    quantity: Literal["pressure", "flowrate"] = "pressure"
    target: str | None = None
    start_time_seconds: Annotated[int, Field(ge=0)]
    end_time_seconds: Annotated[int, Field(gt=0)]
    name: str | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> _SensorFaultBase:
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be > start_time_seconds")
        return self


class BiasFault(_SensorFaultBase):
    """Constant additive offset: ``y'(t) = y(t) + bias_value``.

    Attributes:
        bias_value: Constant offset. Non-zero (a zero bias is a no-op and
            is rejected as a likely config mistake).
    """

    type: Literal["bias"] = "bias"
    bias_value: float

    @model_validator(mode="after")
    def _validate_bias(self) -> BiasFault:
        if self.bias_value == 0.0:
            raise ValueError("bias_value must be non-zero (a zero bias is a no-op)")
        return self


class DriftFault(_SensorFaultBase):
    """Linear ramp: ``y'(t) = y(t) + slope_per_second * (t - start)``.

    Attributes:
        slope_per_second: Drift rate in units of the quantity per second.
            The sign determines the drift direction.
    """

    type: Literal["drift"] = "drift"
    slope_per_second: float


class StuckFault(_SensorFaultBase):
    """Freeze at the first in-window value: ``y'(t) = y(start)``."""

    type: Literal["stuck"] = "stuck"


class DropoutFault(_SensorFaultBase):
    """Drop samples to NaN (or ``fill_value``) on configured sub-intervals.

    Attributes:
        intervals: Non-empty list of ``[start, end)`` sub-intervals
            within the outer fault window. Each interval is independently
            validated.
        fill_value: Value written into corrupted samples. ``None``
            (default) means NaN.
    """

    type: Literal["dropout"] = "dropout"
    intervals: list[tuple[int, int]]
    fill_value: float | None = None

    @model_validator(mode="after")
    def _validate_dropout(self) -> DropoutFault:
        if not self.intervals:
            raise ValueError("dropout fault requires a non-empty 'intervals' list")
        for i, iv in enumerate(self.intervals):
            if len(iv) != 2:
                raise ValueError(f"dropout intervals[{i}] must be a (start, end) pair")
            a, b = int(iv[0]), int(iv[1])
            if b <= a:
                raise ValueError(f"dropout intervals[{i}] end ({b}) must be > start ({a})")
            if a < self.start_time_seconds or b > self.end_time_seconds:
                raise ValueError(
                    f"dropout intervals[{i}] [{a}, {b}) is outside the fault "
                    f"window [{self.start_time_seconds}, {self.end_time_seconds})"
                )
        return self


class NoiseFault(_SensorFaultBase):
    """Additive Gaussian noise: ``y'(t) = y(t) + N(0, sigma**2)``.

    Attributes:
        sigma: Standard deviation of the additive noise. Strictly
            positive.
        rng_offset: Per-fault offset for the noise RNG so two ``noise``
            faults in one scenario draw independent sample streams
            without changing the scenario seed.
    """

    type: Literal["noise"] = "noise"
    sigma: Annotated[float, Field(gt=0)]
    rng_offset: int = 0


class GainFault(_SensorFaultBase):
    """Multiplicative scale error: ``y'(t) = gain_factor * y(t)``.

    Attributes:
        gain_factor: Multiplicative factor. A factor of ``1.0`` is a
            no-op and ``0.0`` zeroes the signal; both are rejected as
            likely config mistakes. Detectability of small gains is
            checked at runtime.
    """

    type: Literal["gain"] = "gain"
    gain_factor: float

    @model_validator(mode="after")
    def _validate_gain(self) -> GainFault:
        if self.gain_factor == 1.0:
            raise ValueError("gain_factor must not be 1.0 (a unit gain is a no-op)")
        if self.gain_factor == 0.0:
            raise ValueError("gain_factor must not be 0.0 (a zero gain zeroes the signal)")
        return self


# Discriminated union of the six sensor fault types, dispatched on the
# ``type`` literal. Pydantic routes a config dict to the matching subclass
# and rejects an unknown ``type`` or a field foreign to the matched type
# (``extra="forbid"`` is inherited from ``_Frozen``).
SensorFaultSpec = Annotated[
    BiasFault | DriftFault | StuckFault | DropoutFault | NoiseFault | GainFault,
    Field(discriminator="type"),
]


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


def _default_formats() -> list[Literal["parquet", "csv"]]:
    """Default output formats for :class:`OutputConfig.formats`."""

    return ["parquet", "csv"]


class OutputConfig(_Frozen):
    """Where and how to write outputs.

    Attributes:
        directory: Output root directory; created on demand.
        formats: One or more of ``parquet`` / ``csv``. The writer registry
            in :mod:`wdn_pipeline.output` is open to additional backends
            (e.g. DuckDB) without changes to this schema.
        write_metadata_sidecar: When true a ``.meta.yaml`` file is
            written next to each data file describing the scenario.
        remove_leak_nodes: When true a post-processing step reverts
            the leak-node split artefacts before serialisation: leak-node
            columns are dropped and split pipe segments are collapsed
            back to the original pipe name so the output schema matches
            the original network. Default false (explicit opt-in). The
            ``leak_demand`` diagnostic table is never affected.
        duckdb: When true the scenario's tables are also inserted into a
            consolidated DuckDB file at ``duckdb_path``.
            Default false. The batch CLI (``wdn-pipeline batch --duckdb
            PATH``) sets this implicitly for every scenario in the
            batch, overriding ``duckdb_path`` with the CLI value.
        duckdb_path: Target DuckDB file when ``duckdb`` is true. Parent
            directory is created on demand. Required when ``duckdb`` is
            true; the model validator enforces this.
        duckdb_wide_tables: When true (default) the DuckDB
            file also materialises the per-scenario wide tables
            (``{basename}_{table}``). When false the file holds only the
            consolidated long tables (``pressure_long`` etc.) plus the
            ``scenarios`` catalogue. The long tables are
            what the query layer slices; the per-scenario wide tables
            cost DuckDB a storage block each, so a large consolidated
            batch is far smaller and faster to write long-only.
            Whole-scenario flat export stays available via the Parquet
            and CSV writers.
    """

    directory: Path = Path("outputs")
    formats: list[Literal["parquet", "csv"]] = Field(default_factory=_default_formats)
    write_metadata_sidecar: bool = True
    remove_leak_nodes: bool = False
    duckdb: bool = False
    duckdb_path: Path | None = None
    duckdb_wide_tables: bool = True

    @model_validator(mode="after")
    def _validate_formats(self) -> OutputConfig:
        if not self.formats:
            raise ValueError("output.formats must list at least one format")
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("output.formats must not contain duplicates")
        if self.duckdb and self.duckdb_path is None:
            raise ValueError("output.duckdb_path is required when output.duckdb is true")
        return self


class ValidationConfig(_Frozen):
    """Tolerances for the physics-based validation checks.

    Validation reports three severities:

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
            deviation for the fault to be considered detectable.
            Below this ratio the ``sensor_fault_signal_applied`` check
            emits a warning (never a hard failure): a small gain on a
            low-variance channel is statistically indistinguishable
            from normal sensor noise and would mislabel ML training
            data.
        headloss_tol_m: Strict OK ceiling (metres) for the
            Hazen-Williams head loss residual ``|dh_observed -
            dh_predicted|`` aggregated over every open pipe and timestep.
            Default ``1e-3`` m comfortably accommodates the WNTRSimulator
            near-zero-flow regularization term (~6e-5 m on the corpus)
            and the EpanetSimulator convergence precision (~8e-4 m) while
            staying far below any real energy violation.
        headloss_warning_tol_m: WARNING ceiling (metres). A residual
            above ``headloss_tol_m`` but at or below this value is a
            warning (a near-miss); above it is a hard failure (a unit
            error, a head reporting bug or non-convergence that slipped
            through). Must be >= ``headloss_tol_m``.
    """

    pressure_min_m: float = 0.0
    pressure_min_warning_tolerance_m: Annotated[float, Field(ge=0)] = 1.0
    pressure_max_m: Annotated[float, Field(gt=0)] = 150.0
    mass_balance_tol_m3s: Annotated[float, Field(gt=0)] = 1e-3
    leak_pressure_drop_min_m: Annotated[float, Field(ge=0)] = 0.01
    gain_detectability_min_ratio: Annotated[float, Field(ge=0)] = 0.05
    headloss_tol_m: Annotated[float, Field(gt=0)] = 1e-3
    headloss_warning_tol_m: Annotated[float, Field(gt=0)] = 1e-1

    @model_validator(mode="after")
    def _validate_headloss_bands(self) -> ValidationConfig:
        if self.headloss_warning_tol_m < self.headloss_tol_m:
            raise ValueError(
                "validation.headloss_warning_tol_m must be >= validation.headloss_tol_m "
                f"(got warning={self.headloss_warning_tol_m}, ok={self.headloss_tol_m})"
            )
        return self


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
            # Water quality runs through the EpanetSimulator, which has no
            # leak support; leaks run through the WNTRSimulator, which has
            # no quality support. The two are mutually exclusive rather
            # than silently approximated.
            if self.simulation.quality.enabled:
                raise ValueError(
                    "simulation.quality cannot be combined with faults.leaks: "
                    "water quality runs through the EpanetSimulator (no leak "
                    "support) while leaks run through the WNTRSimulator (no "
                    "quality support). Run quality and leak scenarios separately."
                )
            if self.simulation.demand_model != "PDD":
                raise ValueError(
                    "simulation.demand_model must be 'PDD' when faults.leaks is non-empty "
                    "(explicit PDD requirement, no silent auto-promotion)"
                )
            for i, spec in enumerate(self.faults.leaks):
                if spec.end_time_seconds > self.simulation.duration_seconds:
                    raise ValueError(
                        f"faults.leaks[{i}].end_time_seconds ({spec.end_time_seconds}) "
                        f"exceeds simulation.duration_seconds ({self.simulation.duration_seconds})"
                    )
        for i, sensor_spec in enumerate(self.faults.sensor_faults):
            if sensor_spec.end_time_seconds > self.simulation.duration_seconds:
                raise ValueError(
                    f"faults.sensor_faults[{i}].end_time_seconds "
                    f"({sensor_spec.end_time_seconds}) exceeds "
                    f"simulation.duration_seconds ({self.simulation.duration_seconds})"
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
