"""Study 4 - downstream ML utility: train, evaluate, plot.

Trains the three detectors per network on normal-only data (semi-supervised),
evaluates them on held-out normal scenarios plus the difficulty-swept leak and
sensor-fault scenarios, and writes:

- ``data/metrics.json`` - every number (mean +/- std over seeds), software
  versions and wall-clock, for the report.
- ``figures/difficulty_vs_leak_area.png`` and ``figures/difficulty_vs_gain_factor.png``
  - the mandatory difficulty curves.

Data access: training windows come from the existing :class:`WDNWindowDataset`
PyTorch loader. Evaluation needs each window's scenario identity, ground-truth
label and end-timestamp (for the range-aware metrics and detection delay) which
the loader does not surface, so eval scenarios are read through the public
:class:`DatasetQuery` and windowed by a small local helper whose semantics match
the loader (time-sorted, fixed-length positional windows, window label = OR over
the window's timestep labels). Only the compact per-scenario matrices are cached,
so memory stays bounded (the overlapping windows are generated transiently).

Determinism: dataset, split, numpy and torch seeds are all fixed and recorded.
CPU LSTM training has minor run-to-run nondeterminism from threaded BLAS
reductions, which is exactly why each detector is run over several seeds and the
report carries mean +/- std rather than a single number.

Usage::

    python experiments/ml_utility/run_experiments.py
    python experiments/ml_utility/run_experiments.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import duckdb
import matplotlib.pyplot as plt
import numpy as np

from experiments.ml_utility import metrics as M
from experiments.ml_utility.detectors import DETECTORS, HYPERPARAMS, STRIDE, WINDOW_LENGTH
from wdn_pipeline.ml import WDNWindowDataset, scenario_train_test_split
from wdn_pipeline.query import DatasetQuery

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DUCKDB_PATH = DATA_DIR / "experiment.duckdb"
MANIFEST_PATH = DATA_DIR / "manifest.json"
METRICS_PATH = DATA_DIR / "metrics.json"

SPLIT_SEED = 1234
TEST_FRACTION = 0.3  # fraction of normal scenarios held out as eval negatives
THRESHOLD_Q = 0.99  # train-score quantile used as the alarm threshold
# WINDOW_LENGTH and STRIDE are imported from detectors (the single source of truth).

# Sensor subtypes treated as distinct fault categories in the breakdown.
SENSOR_SUBTYPES = ("gain", "bias", "drift")


# -- data assembly ---------------------------------------------------------


def _make_windows(
    matrix: np.ndarray, labels: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window a single scenario's matrix (mirrors WDNWindowDataset semantics)."""

    n = matrix.shape[0]
    if n < WINDOW_LENGTH:
        empty = np.empty((0,), dtype=np.float64)
        return np.empty((0, WINDOW_LENGTH, matrix.shape[1]), dtype=np.float32), empty, empty
    starts = list(range(0, n - WINDOW_LENGTH + 1, STRIDE))
    windows = np.stack([matrix[s : s + WINDOW_LENGTH] for s in starts]).astype(np.float32)
    wlabels = np.array([1.0 if labels[s : s + WINDOW_LENGTH].max() > 0 else 0.0 for s in starts])
    end_times = np.array([times[s + WINDOW_LENGTH - 1] for s in starts], dtype=np.float64)
    return windows, wlabels, end_times


def _read_scenario(
    q: DatasetQuery, basename: str, channels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one scenario's pressure matrix + window labels + window end times."""

    df = q.pressure(scenario=basename, nodes=channels, wide=True).sort_index()
    times = df.index.to_numpy()
    matrix = df.to_numpy(dtype=np.float32)
    lab = (
        q.labels(scenario=basename)
        .set_index("time_seconds")["label"]
        .reindex(df.index)
        .fillna(0)
        .to_numpy()
    )
    return _make_windows(matrix, lab, times)


def _train_windows(channels: list[str], train_scenarios: list[str]) -> np.ndarray:
    """Stack all training windows using the existing WDNWindowDataset loader."""

    ds = WDNWindowDataset(
        DUCKDB_PATH,
        train_scenarios,
        quantity="pressure",
        channels=channels,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
    )
    if len(ds) == 0:
        raise RuntimeError("no training windows produced; check the dataset")
    return np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)


# -- metric assembly -------------------------------------------------------


def _concat(subset: list[str], per_scenario: dict[str, np.ndarray]) -> np.ndarray:
    if not subset:
        return np.empty((0,))
    return np.concatenate([per_scenario[b] for b in subset])


def _auc_block(
    subset: list[str],
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict[str, float]:
    s = _concat(subset, scores)
    y = _concat(subset, labels)
    return {"auc_pr": M.auc_pr(y, s), "auc_roc": M.auc_roc(y, s)}


def _range_block(
    subset: list[str],
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, float]:
    preds = [(scores[b] > threshold).astype(int) for b in subset]
    trues = [labels[b].astype(int) for b in subset]
    rng = M.range_precision_recall(preds, trues)
    pa = M.point_adjusted_f1(preds, trues)
    return {
        "range_precision": rng["precision"],
        "range_recall": rng["recall"],
        "range_f1": rng["f1"],
        "point_f1": pa["point_f1"],
        "pa_f1": pa["pa_f1"],
    }


def _delay_block(
    leak_scenarios: list[str],
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    end_times: dict[str, np.ndarray],
    onsets: dict[str, float],
    threshold: float,
    timestep: int,
) -> dict[str, float]:
    delays: list[float] = []
    detected = 0
    for b in leak_scenarios:
        d = M.detection_delay(end_times[b], labels[b], scores[b], threshold, onsets[b])
        if d is not None:
            detected += 1
            delays.append(d)
    rate = detected / len(leak_scenarios) if leak_scenarios else float("nan")
    mean_delay_s = float(np.mean(delays)) if delays else float("nan")
    return {
        "detection_rate": rate,
        "delay_seconds": mean_delay_s,
        "delay_hours": mean_delay_s / 3600.0 if delays else float("nan"),
    }


def evaluate_seed(
    eval_normal: list[str],
    by_category: dict[str, list[str]],
    scores: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    end_times: dict[str, np.ndarray],
    onsets: dict[str, float],
    leak_by_area: dict[str, list[str]],
    gain_by_factor: dict[str, list[str]],
    threshold: float,
    timestep: int,
) -> dict:
    """All metrics for one trained detector instance (one seed)."""

    all_anom = [b for cat in by_category.values() for b in cat]
    out: dict = {}
    out["overall"] = {
        **_auc_block(eval_normal + all_anom, scores, labels),
        **_range_block(eval_normal + all_anom, scores, labels, threshold),
    }
    out["by_fault"] = {}
    for cat, members in by_category.items():
        block = _auc_block(eval_normal + members, scores, labels)
        block.update(_range_block(eval_normal + members, scores, labels, threshold))
        out["by_fault"][cat] = block
    out["leak_delay"] = _delay_block(
        by_category.get("leak", []), scores, labels, end_times, onsets, threshold, timestep
    )
    # Difficulty curves.
    out["difficulty_leak_area"] = {}
    for area, members in leak_by_area.items():
        auc = _auc_block(eval_normal + members, scores, labels)["auc_pr"]
        delay = _delay_block(members, scores, labels, end_times, onsets, threshold, timestep)
        out["difficulty_leak_area"][area] = {
            "auc_pr": auc,
            "detection_rate": delay["detection_rate"],
            "delay_hours": delay["delay_hours"],
        }
    out["difficulty_gain"] = {}
    for gain, members in gain_by_factor.items():
        auc = _auc_block(eval_normal + members, scores, labels)["auc_pr"]
        # Window-level detection rate: fraction of truly-anomalous windows flagged.
        flagged = 0
        total = 0
        for b in members:
            mask = labels[b].astype(bool)
            total += int(mask.sum())
            flagged += int(((scores[b] > threshold) & mask).sum())
        out["difficulty_gain"][gain] = {
            "auc_pr": auc,
            "window_detection_rate": flagged / total if total else float("nan"),
        }
    return out


def _aggregate(seed_dicts: list[dict]) -> dict:
    """Recursively turn parallel per-seed dicts into mean/std leaves."""

    first = seed_dicts[0]
    if isinstance(first, dict):
        return {k: _aggregate([d[k] for d in seed_dicts]) for k in first}
    arr = np.array(seed_dicts, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {"mean": float("nan"), "std": float("nan")}
    with np.errstate(all="ignore"):
        return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}


# -- per-network driver ----------------------------------------------------


def run_network(
    net: str,
    manifest_scn: list[dict],
    seeds: list[int],
) -> dict:
    entries = [e for e in manifest_scn if e["network"] == net]
    normals = [e["basename"] for e in entries if e["scenario_type"] == "normal"]
    leaks = [e["basename"] for e in entries if e["scenario_type"] == "leak"]
    train_normal, eval_normal = scenario_train_test_split(
        normals, test_fraction=TEST_FRACTION, seed=SPLIT_SEED
    )

    by_category: dict[str, list[str]] = {"leak": leaks}
    for sub in SENSOR_SUBTYPES:
        by_category[sub] = [
            e["basename"]
            for e in entries
            if e["scenario_type"] == "sensor_fault" and e["fault_subtype"] == sub
        ]
    leak_by_area: dict[str, list[str]] = {}
    for e in entries:
        if e["scenario_type"] == "leak":
            key = f"{e['leak_area_m2']:.2e}"
            leak_by_area.setdefault(key, []).append(e["basename"])
    gain_by_factor: dict[str, list[str]] = {}
    for e in entries:
        if e["scenario_type"] == "sensor_fault" and e["fault_subtype"] == "gain":
            key = f"{e['gain_factor']:g}"
            gain_by_factor.setdefault(key, []).append(e["basename"])
    onsets = {
        e["basename"]: float(e["onset_seconds"])
        for e in entries
        if e.get("onset_seconds") is not None
    }

    eval_scenarios = eval_normal + [b for cat in by_category.values() for b in cat]

    # Channels = the normal-scenario node set (leak split nodes excluded so the
    # input dimensionality is constant across all scenarios of this network).
    with DatasetQuery(DUCKDB_PATH) as q:
        sample = q.pressure(scenario=train_normal[0], wide=True)
        channels = sorted(str(c) for c in sample.columns)
        labels: dict[str, np.ndarray] = {}
        end_times: dict[str, np.ndarray] = {}
        matrices: dict[str, np.ndarray] = {}
        for b in eval_scenarios:
            w, y, et = _read_scenario(q, b, channels)
            matrices[b] = w
            labels[b] = y
            end_times[b] = et

    timestep = int(end_times[eval_scenarios[0]][1] - end_times[eval_scenarios[0]][0])
    train_windows = _train_windows(channels, train_normal)

    print(
        f"  [{net}] channels={len(channels)} train_normal={len(train_normal)} "
        f"eval_normal={len(eval_normal)} leak={len(leaks)} "
        f"train_windows={train_windows.shape[0]}"
    )

    detector_results: dict = {}
    train_times: dict[str, list[float]] = {}
    for det_name, det_cls in DETECTORS.items():
        seed_metrics: list[dict] = []
        train_times[det_name] = []
        for seed in seeds:
            det = det_cls(n_channels=len(channels), seed=seed)
            t0 = time.perf_counter()
            det.fit(train_windows)
            train_times[det_name].append(time.perf_counter() - t0)
            train_scores = det.score(train_windows)
            threshold = float(np.quantile(train_scores, THRESHOLD_Q))
            scores = {b: det.score(matrices[b]) for b in eval_scenarios}
            seed_metrics.append(
                evaluate_seed(
                    eval_normal,
                    by_category,
                    scores,
                    labels,
                    end_times,
                    onsets,
                    leak_by_area,
                    gain_by_factor,
                    threshold,
                    timestep,
                )
            )
        detector_results[det_name] = _aggregate(seed_metrics)
        detector_results[det_name]["train_seconds_mean"] = float(np.mean(train_times[det_name]))
        print(
            f"    {det_name:18s} overall AUC-PR="
            f"{detector_results[det_name]['overall']['auc_pr']['mean']:.3f}"
        )

    return {
        "channels": len(channels),
        "timestep_seconds": timestep,
        "counts": {
            "train_normal": len(train_normal),
            "eval_normal": len(eval_normal),
            **{k: len(v) for k, v in by_category.items()},
        },
        "leak_areas": sorted(leak_by_area.keys(), key=float),
        "gain_factors": sorted(gain_by_factor.keys(), key=float),
        "detectors": detector_results,
    }


# -- figures ---------------------------------------------------------------


def _plot_difficulty_leak_area(results: dict) -> None:
    networks = list(results.keys())
    fig, axes = plt.subplots(2, len(networks), figsize=(6 * len(networks), 8), squeeze=False)
    for col, net in enumerate(networks):
        areas = results[net]["leak_areas"]
        x = [float(a) for a in areas]
        for det_name, det in results[net]["detectors"].items():
            curve = det["difficulty_leak_area"]
            auc = [curve[a]["auc_pr"]["mean"] for a in areas]
            rate = [curve[a]["detection_rate"]["mean"] for a in areas]
            axes[0][col].plot(x, auc, marker="o", label=det_name)
            axes[1][col].plot(x, rate, marker="o", label=det_name)
        axes[0][col].set_xscale("log")
        axes[1][col].set_xscale("log")
        axes[0][col].set_title(f"{net}: AUC-PR vs leak area")
        axes[1][col].set_title(f"{net}: detection rate vs leak area")
        axes[1][col].set_xlabel("leak hole area (m^2)")
        axes[0][col].set_ylabel("AUC-PR")
        axes[1][col].set_ylabel("detection rate")
        axes[0][col].set_ylim(-0.02, 1.02)
        axes[1][col].set_ylim(-0.02, 1.02)
        axes[0][col].grid(True, alpha=0.3)
        axes[1][col].grid(True, alpha=0.3)
        axes[0][col].legend(fontsize=8)
    fig.suptitle("Difficulty curve: detection performance vs leak hole area")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "difficulty_vs_leak_area.png", dpi=130)
    plt.close(fig)


def _plot_difficulty_gain(results: dict) -> None:
    networks = list(results.keys())
    fig, axes = plt.subplots(2, len(networks), figsize=(6 * len(networks), 8), squeeze=False)
    for col, net in enumerate(networks):
        gains = results[net]["gain_factors"]
        x = [float(g) for g in gains]
        for det_name, det in results[net]["detectors"].items():
            curve = det["difficulty_gain"]
            auc = [curve[g]["auc_pr"]["mean"] for g in gains]
            rate = [curve[g]["window_detection_rate"]["mean"] for g in gains]
            axes[0][col].plot(x, auc, marker="o", label=det_name)
            axes[1][col].plot(x, rate, marker="o", label=det_name)
        axes[0][col].set_title(f"{net}: AUC-PR vs gain factor")
        axes[1][col].set_title(f"{net}: window detection rate vs gain factor")
        axes[1][col].set_xlabel("gain factor (1.0 = no fault)")
        axes[0][col].set_ylabel("AUC-PR")
        axes[1][col].set_ylabel("window detection rate")
        axes[0][col].set_ylim(-0.02, 1.02)
        axes[1][col].set_ylim(-0.02, 1.02)
        axes[0][col].grid(True, alpha=0.3)
        axes[1][col].grid(True, alpha=0.3)
        axes[0][col].axvline(1.0, color="grey", ls="--", alpha=0.5)
        axes[0][col].legend(fontsize=8)
    fig.suptitle("Difficulty curve: detection performance vs sensor gain factor")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "difficulty_vs_gain_factor.png", dpi=130)
    plt.close(fig)


# -- main ------------------------------------------------------------------


def _versions() -> dict:
    import matplotlib as mpl
    import pandas as pd
    import sklearn
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "duckdb": duckdb.__version__,
        "matplotlib": mpl.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if not DUCKDB_PATH.exists():
        raise SystemExit(f"{DUCKDB_PATH} not found. Run generate_experiment_dataset.py first.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    networks = manifest["networks"]

    started = time.perf_counter()
    results: dict = {}
    for net in networks:
        print(f"Network: {net}")
        results[net] = run_network(net, manifest["scenarios"], args.seeds)
    wall = time.perf_counter() - started

    _plot_difficulty_leak_area(results)
    _plot_difficulty_gain(results)

    payload = {
        "config": {
            "seeds": args.seeds,
            "split_seed": SPLIT_SEED,
            "test_fraction": TEST_FRACTION,
            "threshold_quantile": THRESHOLD_Q,
            "hyperparameters": HYPERPARAMS,
            "dataset_master_seed": manifest["master_seed"],
            "timestep_seconds": manifest["timestep_seconds"],
        },
        "versions": _versions(),
        "wall_clock_seconds": wall,
        "networks": results,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {METRICS_PATH} and difficulty figures to {FIG_DIR} ({wall:.1f}s)")


if __name__ == "__main__":
    main()
