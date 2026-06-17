"""Tests for the water quality options (chemical / age / trace).

Water quality runs through the EpanetSimulator (the WNTRSimulator emits
no quality table) and is supported only for scenarios without leaks. The
whole scenario is solved by EPANET when quality is requested; the
existing WNTRSimulator path is untouched when it is not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from wdn_pipeline.config import (
    FaultsConfig,
    LeakSpec,
    NetworkConfig,
    OutputConfig,
    PipelineConfig,
    QualitySourceSpec,
    ScenarioConfig,
    SimulationConfig,
    WaterQualityConfig,
)
from wdn_pipeline.network import apply_quality_options, load_network
from wdn_pipeline.query import DatasetQuery
from wdn_pipeline.runner import run
from wdn_pipeline.simulation import run_simulation


def _sim_cfg(quality: WaterQualityConfig, duration: int = 12 * 3600) -> SimulationConfig:
    return SimulationConfig(
        duration_seconds=duration,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
        demand_model="DDA",
        quality=quality,
    )


def _pipeline_cfg(
    tmp_path: Path,
    quality: WaterQualityConfig,
    duration: int = 12 * 3600,
    label: str = "q",
    **output_overrides: object,
) -> PipelineConfig:
    return PipelineConfig(
        network=NetworkConfig(inp_path="Net3", name="net3"),
        simulation=_sim_cfg(quality, duration),
        seed=42,
        scenario=ScenarioConfig(type="normal", label=label),
        output=OutputConfig(
            directory=tmp_path / "outputs",
            formats=["parquet"],
            **output_overrides,  # type: ignore[arg-type]
        ),
    )


def _run_quality(quality: WaterQualityConfig, duration: int = 12 * 3600):
    """Run only the simulation stage on the EpanetSimulator path."""

    cfg = _sim_cfg(quality, duration)
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    apply_quality_options(wn, cfg)
    return wn, run_simulation(wn, use_epanet_for_quality=True)


# -- physics / simulator routing ------------------------------------------


def test_wntr_path_has_no_quality_table() -> None:
    # The default path (no quality) keeps using WNTRSimulator, which emits
    # no quality table.
    cfg = _sim_cfg(WaterQualityConfig())
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    results = run_simulation(wn, use_epanet_for_quality=False)
    assert results.quality is None


def test_age_quality_grows_from_source() -> None:
    _wn, results = _run_quality(WaterQualityConfig(parameter="age"), duration=24 * 3600)
    q = results.quality
    assert q is not None
    assert not q.empty
    # Age is a non-negative residence time everywhere.
    assert float(np.nanmin(q.to_numpy())) >= 0.0
    final = q.iloc[-1]
    # The reservoir source carries the freshest water (age ~0); the rest of
    # the network is strictly older, so the mean exceeds the source age.
    assert float(final["River"]) == pytest.approx(0.0, abs=1.0)
    assert float(final.mean()) > float(final["River"])


def test_chemical_quality_decays_from_source() -> None:
    quality = WaterQualityConfig(
        parameter="chemical",
        chemical_name="chlorine",
        bulk_coeff=-1.157e-05,  # ~ -1.0/day, decay
        sources=[
            QualitySourceSpec(node="River", source_type="CONCEN", strength=1.0),
            QualitySourceSpec(node="Lake", source_type="CONCEN", strength=1.0),
        ],
    )
    _wn, results = _run_quality(quality, duration=24 * 3600)
    q = results.quality
    assert q is not None and not q.empty
    values = q.to_numpy()
    # Concentration stays within [0, source strength].
    assert float(np.nanmin(values)) >= -1e-9
    assert float(np.nanmax(values)) <= 1.0 + 1e-6
    # Decay means at least one node ends below the 1.0 mg/L source.
    assert float(q.iloc[-1].min()) < 1.0


def test_trace_quality_is_percentage() -> None:
    _wn, results = _run_quality(
        WaterQualityConfig(parameter="trace", trace_node="Lake"), duration=24 * 3600
    )
    q = results.quality
    assert q is not None and not q.empty
    values = q.to_numpy()
    # Trace is a percentage of flow originating at the trace node.
    assert float(np.nanmin(values)) >= -1e-9
    assert float(np.nanmax(values)) <= 100.0 + 1e-6
    # The trace node itself is 100% its own water.
    assert float(q.iloc[-1]["Lake"]) == pytest.approx(100.0, abs=1e-3)


def test_source_pattern_is_registered() -> None:
    quality = WaterQualityConfig(
        parameter="chemical",
        sources=[QualitySourceSpec(node="River", strength=1.0, pattern=[1.0, 0.5, 1.0, 0.5])],
    )
    cfg = _sim_cfg(quality)
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    applied = apply_quality_options(wn, cfg)
    assert "wq_source_0_pattern" in wn.pattern_name_list
    assert applied["sources"][0]["pattern"] == "wq_source_0_pattern"


# -- config validation ----------------------------------------------------


def test_trace_requires_trace_node() -> None:
    with pytest.raises(ValidationError, match="requires 'trace_node'"):
        WaterQualityConfig(parameter="trace")


def test_chemical_fields_rejected_under_age() -> None:
    with pytest.raises(ValidationError, match="age"):
        WaterQualityConfig(parameter="age", bulk_coeff=-1e-5)


def test_chemical_fields_rejected_under_trace() -> None:
    with pytest.raises(ValidationError, match="not valid for a 'trace'"):
        WaterQualityConfig(parameter="trace", trace_node="Lake", chemical_name="chlorine")


def test_quality_fields_rejected_under_none() -> None:
    with pytest.raises(ValidationError, match="parameter"):
        WaterQualityConfig(parameter="none", bulk_coeff=-1e-5)


def test_trace_node_rejected_under_chemical() -> None:
    with pytest.raises(ValidationError, match="only valid for a 'trace'"):
        WaterQualityConfig(parameter="chemical", trace_node="Lake")


def test_leak_plus_quality_rejected(tmp_path: Path) -> None:
    quality = WaterQualityConfig(parameter="age")
    with pytest.raises(ValidationError, match="quality cannot be combined with"):
        PipelineConfig(
            network=NetworkConfig(inp_path="Net3", name="net3"),
            simulation=SimulationConfig(
                duration_seconds=12 * 3600,
                hydraulic_timestep_seconds=3600,
                report_timestep_seconds=3600,
                demand_model="PDD",
                quality=quality,
            ),
            faults=FaultsConfig(
                leaks=[
                    LeakSpec(
                        pipe=None,
                        area_m2=0.01,
                        start_time_seconds=3600,
                        end_time_seconds=7200,
                    )
                ]
            ),
            scenario=ScenarioConfig(type="leak", label="leak"),
            output=OutputConfig(directory=tmp_path / "outputs", formats=["parquet"]),
        )


# -- network-prep node validation -----------------------------------------


def test_bad_trace_node_raises_at_network_prep() -> None:
    cfg = _sim_cfg(WaterQualityConfig(parameter="trace", trace_node="not_a_node"))
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    with pytest.raises(ValueError, match="trace_node"):
        apply_quality_options(wn, cfg)


def test_bad_source_node_raises_at_network_prep() -> None:
    quality = WaterQualityConfig(
        parameter="chemical",
        sources=[QualitySourceSpec(node="not_a_node", strength=1.0)],
    )
    cfg = _sim_cfg(quality)
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    with pytest.raises(ValueError, match="is not a"):
        apply_quality_options(wn, cfg)


# -- integration: output, query, provenance, determinism ------------------


def test_quality_table_written_and_queryable(tmp_path: Path) -> None:
    db_path = tmp_path / "q.duckdb"
    cfg = _pipeline_cfg(
        tmp_path,
        WaterQualityConfig(parameter="age"),
        duration=24 * 3600,
        label="age",
        duckdb=True,
        duckdb_path=db_path,
    )
    summary = run(cfg)
    assert any(p.name.endswith("_quality.parquet") for p in summary.output_paths)

    with DatasetQuery(db_path) as q:
        df = q.quality(scenario="net3_age_42")
        assert not df.empty
        wide = q.quality(scenario="net3_age_42", nodes=["River"], wide=True)
        assert float(wide["River"].iloc[-1]) == pytest.approx(0.0, abs=1.0)


def test_quality_provenance_in_summary_and_sidecar(tmp_path: Path) -> None:
    cfg = _pipeline_cfg(
        tmp_path,
        WaterQualityConfig(parameter="age", quality_timestep_seconds=600),
        label="age",
        write_metadata_sidecar=True,
    )
    summary = run(cfg)
    assert summary.quality_settings["parameter"] == "age"
    assert summary.quality_settings["quality_timestep_seconds"] == 600

    meta = yaml.safe_load(summary.metadata_path.read_text())
    assert meta["water_quality"]["parameter"] == "age"


def test_none_default_produces_no_quality_table(tmp_path: Path) -> None:
    # Regression guard: a no-quality scenario produces the original table
    # set and no water_quality provenance key.
    cfg = _pipeline_cfg(tmp_path, WaterQualityConfig(), label="normal")
    summary = run(cfg)
    assert not any(p.name.endswith("_quality.parquet") for p in summary.output_paths)
    assert summary.quality_settings == {}
    meta = yaml.safe_load(summary.metadata_path.read_text())
    assert "water_quality" not in meta


def test_quality_is_deterministic(tmp_path: Path) -> None:
    quality = WaterQualityConfig(parameter="age")
    _wn1, r1 = _run_quality(quality, duration=24 * 3600)
    _wn2, r2 = _run_quality(quality, duration=24 * 3600)
    assert r1.quality is not None and r2.quality is not None
    np.testing.assert_allclose(r1.quality.to_numpy(), r2.quality.to_numpy(), rtol=0, atol=1e-9)


# -- physics plausibility validator ---------------------------------------


def test_quality_plausible_runs_and_passes_for_age(tmp_path: Path) -> None:
    cfg = _pipeline_cfg(
        tmp_path, WaterQualityConfig(parameter="age"), duration=24 * 3600, label="age"
    )
    summary = run(cfg)
    qp = next(c for c in summary.validation.checks if c.name == "quality_plausible")
    assert qp.severity == "ok"


def test_quality_plausible_passes_for_chemical(tmp_path: Path) -> None:
    quality = WaterQualityConfig(
        parameter="chemical",
        chemical_name="chlorine",
        bulk_coeff=-1.157e-05,
        sources=[
            QualitySourceSpec(node="River", source_type="CONCEN", strength=1.0),
            QualitySourceSpec(node="Lake", source_type="CONCEN", strength=1.0),
        ],
    )
    cfg = _pipeline_cfg(tmp_path, quality, duration=24 * 3600, label="chem")
    summary = run(cfg)
    qp = next(c for c in summary.validation.checks if c.name == "quality_plausible")
    assert qp.severity == "ok"


def test_quality_plausible_fails_on_negative_chemical() -> None:
    from dataclasses import replace

    from wdn_pipeline.validation import _check_quality_plausible

    quality = WaterQualityConfig(
        parameter="chemical",
        sources=[QualitySourceSpec(node="River", strength=1.0)],
    )
    _wn, results = _run_quality(quality, duration=6 * 3600)
    bad = results.quality.copy()
    bad.iloc[0, 0] = -5.0
    tampered = replace(results, quality=bad)
    check = _check_quality_plausible(tampered, quality)
    assert check.severity == "fail"
    assert "negative" in check.detail


def test_quality_plausible_warns_on_chemical_above_source() -> None:
    from dataclasses import replace

    from wdn_pipeline.validation import _check_quality_plausible

    quality = WaterQualityConfig(
        parameter="chemical",
        sources=[QualitySourceSpec(node="River", strength=1.0)],
    )
    _wn, results = _run_quality(quality, duration=6 * 3600)
    bad = results.quality.copy()
    bad.iloc[0, 0] = 5.0  # well above the 1.0 source strength
    tampered = replace(results, quality=bad)
    check = _check_quality_plausible(tampered, quality)
    assert check.severity == "warning"


def test_quality_plausible_fails_on_out_of_range_trace() -> None:
    from dataclasses import replace

    from wdn_pipeline.validation import _check_quality_plausible

    quality = WaterQualityConfig(parameter="trace", trace_node="Lake")
    _wn, results = _run_quality(quality, duration=6 * 3600)
    bad = results.quality.copy()
    bad.iloc[0, 0] = 150.0  # a percentage can never exceed 100
    tampered = replace(results, quality=bad)
    check = _check_quality_plausible(tampered, quality)
    assert check.severity == "fail"


def test_quality_plausible_ok_without_quality_table() -> None:
    """The check is a safe no-op on a results object with no quality table."""

    from wdn_pipeline.validation import _check_quality_plausible

    cfg = _sim_cfg(WaterQualityConfig())
    wn = load_network(NetworkConfig(inp_path="Net3"), cfg)
    results = run_simulation(wn, use_epanet_for_quality=False)
    check = _check_quality_plausible(results, WaterQualityConfig())
    assert check.severity == "ok"
    assert "no quality table" in check.detail
