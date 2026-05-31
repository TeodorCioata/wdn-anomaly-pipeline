"""Phase 4 reproducibility audit.

Runs every YAML config in ``configs/`` twice into separate output
directories, then compares the ``pressure``, ``flowrate``, ``demand``
and ``leak_demand`` Parquet tables. The WNTRSimulator Newton solver is
not bit-exact across calls but the documented residual floor is
~1e-12 m. Any config whose two runs disagree above that threshold is
flagged as a non-determinism failure.

Writes ``outputs/reproducibility_audit.txt`` with one line per config
(severity + worst residual + elapsed time) and a final summary block.

Usage::

    python scripts/reproducibility_audit.py
    python scripts/reproducibility_audit.py --configs-dir configs/ \
        --output-dir outputs/reproducibility_audit/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT / "src"))

from wdn_pipeline.config import load_config  # noqa: E402
from wdn_pipeline.runner import run  # noqa: E402

# WNTR documented Newton-solver residual floor.
NOISE_FLOOR_M = 1e-12

# Tables to diff. The clean tables are excluded because they are copies
# of pressure / flowrate at write time; if those two match, the clean
# siblings match by construction.
DIFF_TABLES = ("pressure", "flowrate", "demand", "leak_demand")


def _patched_config_text(src: Path, override_directory: Path) -> str:
    """Load ``src`` as YAML and rewrite ``output.directory`` to ``override_directory``."""

    with src.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.setdefault("output", {})
    raw["output"]["directory"] = str(override_directory)
    # Force parquet so the diff is independent of any csv flag in the
    # source config.
    raw["output"]["formats"] = ["parquet"]
    raw["output"]["write_metadata_sidecar"] = False
    return yaml.safe_dump(raw, sort_keys=True)


def _run_to_dir(config_src: Path, out_dir: Path) -> Path:
    """Run one config with the output directory overridden to ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    patched = out_dir / "config.yaml"
    patched.write_text(_patched_config_text(config_src, out_dir))
    cfg = load_config(patched)
    run(cfg)
    return out_dir


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    if "time_seconds" in df.columns:
        df = df.set_index("time_seconds")
    return df


def _compare_dirs(
    dir_a: Path, dir_b: Path
) -> tuple[float, str]:
    """Return (max_abs_residual, detail string) across the diff tables."""

    worst: float = 0.0
    worst_table = ""
    details: list[str] = []
    for table in DIFF_TABLES:
        # Match the per-table parquet exactly; the "*_demand.parquet"
        # glob would otherwise also match "*_leak_demand.parquet".
        suffix = f"_{table}.parquet"
        a_files = sorted(
            p
            for p in dir_a.glob("*.parquet")
            if p.name.endswith(suffix)
            and not p.name.endswith(f"_leak{suffix}")
            and not p.name.endswith(f"_clean{suffix.replace('.parquet', '')}.parquet")
        )
        b_files = sorted(
            p
            for p in dir_b.glob("*.parquet")
            if p.name.endswith(suffix)
            and not p.name.endswith(f"_leak{suffix}")
            and not p.name.endswith(f"_clean{suffix.replace('.parquet', '')}.parquet")
        )
        if table == "leak_demand":
            # The leak_demand table has its own dedicated name.
            a_files = sorted(p for p in dir_a.glob("*_leak_demand.parquet"))
            b_files = sorted(p for p in dir_b.glob("*_leak_demand.parquet"))
        if len(a_files) != 1 or len(b_files) != 1:
            details.append(f"{table}: missing or duplicate parquet")
            continue
        df_a = _load_parquet(a_files[0])
        df_b = _load_parquet(b_files[0])
        # Drop the label / mask columns; they are bool/int8 and would
        # poison the abs() max if cast to float. The numeric measure
        # columns are what we care about for determinism.
        numeric_a = df_a.select_dtypes(include=[np.number]).drop(
            columns=[c for c in df_a.columns if c == "label"], errors="ignore"
        )
        numeric_b = df_b.select_dtypes(include=[np.number]).drop(
            columns=[c for c in df_b.columns if c == "label"], errors="ignore"
        )
        common = sorted(set(numeric_a.columns) & set(numeric_b.columns))
        if not common:
            details.append(f"{table}: no shared numeric columns")
            continue
        arr_a = numeric_a[common].to_numpy()
        arr_b = numeric_b[common].to_numpy()
        # Pairs where both sides are NaN are considered equal (dropout).
        both_nan = np.isnan(arr_a) & np.isnan(arr_b)
        diff = np.where(both_nan, 0.0, np.abs(arr_a - arr_b))
        max_diff = float(np.nanmax(diff)) if diff.size else 0.0
        details.append(f"{table}: max|diff|={max_diff:.3e}")
        if max_diff > worst:
            worst = max_diff
            worst_table = table
    detail = "; ".join(details)
    if worst_table:
        detail = f"worst {worst_table}={worst:.3e} | " + detail
    return worst, detail


def audit(
    configs_dir: Path, output_dir: Path, glob_pattern: str = "*.yaml"
) -> int:
    """Run the audit across every matching config. Returns process exit code."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "outputs" / "reproducibility_audit.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    configs = sorted(configs_dir.glob(glob_pattern))
    if not configs:
        print(f"no configs matched {configs_dir / glob_pattern}", file=sys.stderr)
        return 2

    lines: list[str] = [
        f"# Reproducibility audit ({time.strftime('%Y-%m-%d %H:%M:%S')})",
        f"# Source: {configs_dir}",
        f"# Noise floor: {NOISE_FLOOR_M:.0e} m",
        "",
    ]
    n_total = 0
    n_pass = 0
    n_fail = 0
    n_error = 0

    for cfg_path in configs:
        n_total += 1
        name = cfg_path.stem
        run_a_dir = output_dir / name / "run_a"
        run_b_dir = output_dir / name / "run_b"
        try:
            started = time.perf_counter()
            _run_to_dir(cfg_path, run_a_dir)
            _run_to_dir(cfg_path, run_b_dir)
            elapsed = time.perf_counter() - started
            worst, detail = _compare_dirs(run_a_dir, run_b_dir)
            status = "PASS" if worst <= NOISE_FLOOR_M else "FAIL"
            if status == "PASS":
                n_pass += 1
            else:
                n_fail += 1
            lines.append(
                f"{status:>5} {name:<40s} elapsed={elapsed:5.1f}s {detail}"
            )
            print(lines[-1])
        except Exception as exc:  # noqa: BLE001 - audit isolates failures
            n_error += 1
            lines.append(f"ERROR {name:<40s} {type(exc).__name__}: {exc}")
            print(lines[-1], file=sys.stderr)

    lines.append("")
    lines.append(
        f"Total: {n_total}, pass: {n_pass}, fail: {n_fail}, error: {n_error}"
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {report_path}")
    return 0 if n_fail == 0 and n_error == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=ROOT / "configs",
        help="Directory of YAML configs to audit (default: configs/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "reproducibility_audit",
        help="Where to write the duplicate run directories.",
    )
    parser.add_argument(
        "--glob",
        default="*.yaml",
        help="Glob pattern used inside --configs-dir.",
    )
    args = parser.parse_args()
    return audit(args.configs_dir, args.output_dir, args.glob)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
