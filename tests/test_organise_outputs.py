"""Tests for ``scripts/organise_outputs.py`` (Phase 5 Week 7)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "organise_outputs.py"


@pytest.fixture(scope="module")
def organise_module():
    """Load the organise_outputs script as an importable module."""

    spec = importlib.util.spec_from_file_location("organise_outputs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["organise_outputs"] = module
    spec.loader.exec_module(module)
    return module


def _write_normal_config(
    config_dir: Path,
    stem: str,
    network_name: str = "net3",
    label: str = "normal",
) -> Path:
    cfg = {
        "network": {"inp_path": "Net3", "name": network_name},
        "simulation": {
            "duration_seconds": 24 * 3600,
            "hydraulic_timestep_seconds": 3600,
            "report_timestep_seconds": 3600,
        },
        "seed": 1,
        "scenario": {"type": "normal", "label": label},
        "output": {
            "directory": str(config_dir / "_scratch"),
            "formats": ["parquet", "csv"],
            "write_metadata_sidecar": True,
        },
    }
    path = config_dir / f"{stem}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_categorise_explicit_map_hits(organise_module) -> None:
    cat, net, var = organise_module.EXPLICIT_TREE_MAP["normal_net3_pdd"]
    assert (cat, net, var) == ("normal", "net3", "pdd")
    cat, net, var = organise_module.EXPLICIT_TREE_MAP["cumulative_leak_bias_net3"]
    assert (cat, net, var) == ("cumulative", "net3", "leak_bias")
    cat, net, var = organise_module.EXPLICIT_TREE_MAP["sensor_gain_net3"]
    assert (cat, net, var) == ("sensor_fault", "net3", "gain")


def test_categorise_unknown_falls_back_to_config(organise_module, tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = _write_normal_config(
        config_dir, "unknown_stem_xyz", network_name="net3", label="custom_label"
    )
    cat, net, var = organise_module.categorise_config(path)
    assert (cat, net, var) == ("normal", "net3", "custom_label")


def test_organise_one_produces_expected_tree(organise_module, tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfg_path = _write_normal_config(config_dir, "normal_net3")
    organised_root = tmp_path / "organised"
    scratch_root = tmp_path / "scratch"
    record = organise_module.organise_one(cfg_path, organised_root, scratch_root)
    assert record["category"] == "normal"
    assert record["network"] == "net3"
    assert record["variant"] == "dda"
    scenario_dir = organised_root / "normal" / "net3" / "dda"
    assert scenario_dir.is_dir()
    for table in ("pressure", "flowrate", "demand"):
        f = scenario_dir / f"{table}.csv"
        assert f.is_file()
        assert f.stat().st_size > 0
    assert (scenario_dir / "metadata.yaml").is_file()


def test_organise_one_strips_clean_tables_for_normal(
    organise_module, tmp_path: Path
) -> None:
    """A normal scenario must not carry ``*_clean`` siblings into the tree.

    They are present in the parquet outputs because the pipeline always
    emits both clean and corrupted tables, but for a normal scenario
    they are identical to the corrupted siblings and just inflate the
    Drive upload.
    """

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfg_path = _write_normal_config(config_dir, "normal_net3")
    organised_root = tmp_path / "organised"
    scratch_root = tmp_path / "scratch"
    organise_module.organise_one(cfg_path, organised_root, scratch_root)
    scenario_dir = organised_root / "normal" / "net3" / "dda"
    files = {p.name for p in scenario_dir.iterdir()}
    assert "pressure_clean.csv" not in files
    assert "flowrate_clean.csv" not in files


def test_render_readme_contains_directory_layout(organise_module) -> None:
    records = [
        {
            "config_path": "configs/normal_net3.yaml",
            "category": "normal",
            "network": "net3",
            "variant": "dda",
            "tables": ["pressure", "flowrate", "demand"],
            "scenario_dir": "organised/normal/net3/dda",
            "severity": "warning",
        }
    ]
    text = organise_module.render_readme(records)
    assert "# Organised Pipeline Outputs" in text
    assert "configs/normal_net3.yaml" in text
    assert "organised/normal/net3/dda" in text
    assert "warning" in text
    assert "## Conventions" in text
