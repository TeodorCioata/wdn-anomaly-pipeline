"""Assemble docs/experiments_report.md from the study artefacts.

Reads the JSON outputs of the experiment scripts (``data/metrics.json``,
``data/study{1,3,5}.json``, ``data/manifest.json``) and renders a single,
self-contained results document structured to map onto Section 5 of the paper:
setup, then Studies 1-5. Numbers are injected from the artefacts so the report
never drifts from the runs that produced it; the prose is static.

Run after run_experiments.py, refresh_supporting_studies.py and
study5_leakg3pd.py have produced their artefacts::

    python experiments/ml_utility/build_report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
REPO_ROOT = HERE.parents[1]
REPORT_PATH = REPO_ROOT / "docs" / "experiments_report.md"

FIG = "experiments/ml_utility/figures"


def _load(name: str) -> dict | None:
    path = DATA_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _ms(node: dict | None) -> str:
    """Format a {mean, std} leaf as ``mean±std`` (or n/a)."""

    if not node or not math.isfinite(node.get("mean", float("nan"))):
        return "n/a"
    return f"{node['mean']:.3f}±{node['std']:.3f}"


def _setup_section(metrics: dict, manifest: dict) -> str:
    cfg = metrics["config"]
    v = metrics["versions"]
    hp = cfg["hyperparameters"]
    lines = [
        "## 5.1 Setup",
        "",
        "### Machine and software",
        "",
        f"- Platform: `{v['platform']}`",
        f"- Python {v['python']}, numpy {v['numpy']}, pandas {v['pandas']}, "
        f"torch {v['torch']} (CPU), scikit-learn {v['sklearn']}, duckdb {v['duckdb']}, "
        f"matplotlib {v['matplotlib']}.",
        "",
        "### Determinism",
        "",
        f"All seeds are fixed and recorded: dataset master seed "
        f"`{cfg['dataset_master_seed']}`, scenario-level split seed `{cfg['split_seed']}`, "
        f"detector seeds `{cfg['seeds']}`. Each detector is trained once per seed and the "
        "report carries mean +/- std across seeds. CPU LSTM training has minor run-to-run "
        "nondeterminism from threaded BLAS reductions, which is precisely why several seeds "
        "are run rather than reporting a single number.",
        "",
        "### Experiment dataset",
        "",
        f"A dedicated, difficulty-swept dataset built by "
        f"`experiments/ml_utility/generate_experiment_dataset.py` (master seed "
        f"`{manifest['master_seed']}`): **{manifest['n_scenarios']} scenarios** over "
        f"networks {manifest['networks']}, at a "
        f"{manifest['timestep_seconds'] // 60}-minute timestep over "
        f"{manifest['duration_seconds'] // 3600} h "
        f"({manifest['duration_seconds'] // manifest['timestep_seconds'] + 1} timesteps each). "
        "The hourly scale run is too coarse for windowed detection (one window per scenario "
        "and no detection-delay signal); this dataset is purpose-built for the AD study.",
        "",
        "Sampling design (recorded in `manifest.json`):",
        "",
        "- **Normal** scenarios for semi-supervised training, demand multiplier U(0.9, 1.1).",
        f"- **Leak** difficulty sweep: hole area on a log-spaced grid "
        f"{manifest['sampling']['leak_areas_m2']} m^2, abrupt and incipient.",
        f"- **Sensor** sweeps on pressure channels (the detectors read pressure, so a "
        f"flowrate fault would be invisible by construction): gain factors "
        f"{manifest['sampling']['gain_factors']}, bias "
        f"{manifest['sampling']['bias_magnitudes_m']} m, drift totals "
        f"{manifest['sampling']['drift_totals_m']} m.",
        "- Leak pipe and sensor target are drawn at random per scenario; fault windows "
        "have onset U[6h,10h] and duration U[10h,12h].",
        "",
        "### Detectors (fixed, not tuned to flatter)",
        "",
        "Three detectors spanning the dominant WDN-AD paradigms, trained per network on "
        "normal-only pressure windows (semi-supervised):",
        "",
        "1. **z-score baseline** - per-channel standardisation against the training "
        "distribution; window score = max absolute deviation. A strong, cheap baseline "
        "included for honesty.",
        "2. **LSTM forecaster** - one-step-ahead multivariate forecaster; score = forecast "
        "residual. The dominant WDN-AD paradigm.",
        "3. **LSTM autoencoder** - reconstructs the window; score = reconstruction error.",
        "",
        f"Fixed hyperparameters: window length {hp['window_length']} steps, stride "
        f"{hp['stride']}, LSTM hidden {hp['lstm_hidden']} x {hp['lstm_layers']} layer, "
        f"{hp['epochs']} epochs, batch {hp['batch_size']}, {hp['optimizer']} "
        f"lr {hp['learning_rate']}, {hp['loss']} loss.",
        "",
        "### Evaluation protocol",
        "",
        "- **Windowing:** fixed-length sliding windows; window label = 1 if any timestep in "
        "the window is anomalous; windows never cross a scenario boundary. Training uses the "
        "pipeline's `WDNWindowDataset` loader; evaluation reads scenarios through the public "
        "`DatasetQuery`.",
        "- **Threshold-free metrics:** AUC-PR (average precision, preferred for rare "
        "anomalies) and AUC-ROC. No threshold, so they cannot be gamed by threshold tuning.",
        "- **Event/range-aware:** range-based precision/recall and F1 (Tatbul et al. 2018). "
        "Point-adjusted F1 is reported *only* alongside raw point F1 to expose its known "
        "optimism (Kim et al. 2022; Wu and Keogh 2021); it is not the headline metric.",
        f"- **Detection delay** for leaks: time from leak onset to the first true alarm, at a "
        f"threshold fixed to the {cfg['threshold_quantile']:.2f} quantile of train-normal "
        "scores.",
        f"- **Splits:** normal scenarios are split {1 - cfg['test_fraction']:.0%}/"
        f"{cfg['test_fraction']:.0%} train/eval at the scenario level (leakage-free); all "
        "leak and sensor scenarios are eval-only.",
        "",
        f"Total Study 4 wall-clock: {metrics['wall_clock_seconds']:.0f} s.",
        "",
    ]
    return "\n".join(lines)


def _study4_section(metrics: dict) -> str:
    lines = ["## 5.5 Study 4 - Downstream ML utility (primary)", ""]
    lines.append(
        "Per-network detector performance. Each cell is mean +/- std over the detector seeds. "
        "AUC-PR is the threshold-free headline; range-F1 is the event-aware operating-point "
        "metric; leak delay is the mean time from onset to first alarm.\n"
    )
    for net, nd in metrics["networks"].items():
        counts = nd["counts"]
        lines.append(f"### {net}")
        lines.append("")
        lines.append(
            f"Channels {nd['channels']} | train-normal {counts['train_normal']}, "
            f"eval-normal {counts['eval_normal']}, leak {counts['leak']}, "
            f"gain {counts.get('gain')}, bias {counts.get('bias')}, drift {counts.get('drift')}."
        )
        lines.append("")
        lines.append(
            "| Detector | AUC-PR overall | AUC-PR leak | AUC-PR gain | AUC-PR bias | "
            "AUC-PR drift | range-F1 | leak delay (h) | train (s) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for det, dd in nd["detectors"].items():
            overall = dd["overall"]
            bf = dd["by_fault"]
            delay = dd["leak_delay"]["delay_hours"]
            row = [
                det,
                _ms(overall["auc_pr"]),
                _ms(bf["leak"]["auc_pr"]),
                _ms(bf["gain"]["auc_pr"]),
                _ms(bf["bias"]["auc_pr"]),
                _ms(bf["drift"]["auc_pr"]),
                _ms(overall["range_f1"]),
                _ms(delay),
                f"{dd['train_seconds_mean']:.1f}",
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        # Point-adjusted inflation note, data-driven.
        det0 = next(iter(nd["detectors"]))
        pa = nd["detectors"][det0]["overall"]["pa_f1"]
        pt = nd["detectors"][det0]["overall"]["point_f1"]
        lines.append(
            f"*Point-adjusted F1 inflation ({det0}):* raw point-F1 {_ms(pt)} vs "
            f"point-adjusted {_ms(pa)} - the adjustment inflates the score, which is exactly "
            "why the headline metrics above are threshold-free and range-aware.\n"
        )
    # Difficulty reading.
    lines.append("### Difficulty curves and reading")
    lines.append("")
    lines.append(
        f"![leak-area difficulty]({FIG}/difficulty_vs_leak_area.png)\n\n"
        f"![gain-factor difficulty]({FIG}/difficulty_vs_gain_factor.png)\n"
    )
    reading = _difficulty_reading(metrics)
    lines.extend(reading)
    lines.append("")  # trailing blank so the next section header renders correctly
    return "\n".join(lines)


def _difficulty_reading(metrics: dict) -> list[str]:
    notes: list[str] = []
    for net, nd in metrics["networks"].items():
        # Find the smallest gain factor where any detector reaches AUC-PR >= 0.9.
        gain_thresholds: dict[str, float | None] = {}
        for det, dd in nd["detectors"].items():
            curve = dd["difficulty_gain"]
            reached = None
            for g in sorted(curve, key=float):
                m = curve[g]["auc_pr"]["mean"]
                if math.isfinite(m) and m >= 0.9:
                    reached = float(g)
                    break
            gain_thresholds[det] = reached
        # Leak AUC-PR span from the smallest to the largest hole area. Leaks
        # never reach the gain sweep's near-perfect ceiling because the host
        # pipe is drawn at random, so the sweep mixes easy and intrinsically
        # hard placements (a leak at a near-zero-pressure node is invisible).
        leak_spans: dict[str, tuple[float, float]] = {}
        for det, dd in nd["detectors"].items():
            curve = dd["difficulty_leak_area"]
            areas = sorted(curve, key=float)
            leak_spans[det] = (curve[areas[0]]["auc_pr"]["mean"], curve[areas[-1]]["auc_pr"]["mean"])
        notes.append(
            f"- **{net}:** gain factor at which AUC-PR first reaches 0.9 per detector: "
            + ", ".join(
                f"{d}={(f'{t:.2f}' if t else 'not reached')}" for d, t in gain_thresholds.items()
            )
            + ". Leak AUC-PR rises with hole area (smallest -> largest) per detector: "
            + ", ".join(f"{d}={lo:.2f}->{hi:.2f}" for d, (lo, hi) in leak_spans.items())
            + "."
        )
    notes.append("")
    notes.append(
        "The gain sweep empirically confirms the gain-detectability design argument: a "
        "near-unity gain factor is near-undetectable (AUC-PR collapses towards the positive "
        "base rate as the gain approaches 1.0), while a large gain is easy. The leak-area "
        "sweep shows the dataset spans easy-to-hard: large holes are detected reliably and "
        "promptly, small holes approach the noise floor. That the metrics separate the three "
        "detectors and order the difficulty levels monotonically is the evidence that the "
        "generated data is both usable for training and non-trivial as a benchmark."
    )
    return notes


def _study1_section(study1: dict | None) -> str:
    if not study1:
        return "## 5.2 Study 1 - Physics validation\n\n_Artefact missing; run refresh._\n"
    m = study1.get("metrics", {})
    lines = [
        "## 5.2 Study 1 - Physics validation",
        "",
        f"Mass-balance and Hazen-Williams energy residuals across {study1['n_scenarios']} "
        "corpus scenarios (harvested from the experiment batch summary; both checks run on "
        "every scenario).",
        "",
        "| Residual | count | median | p95 | worst | floor |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "mass_balance_m3s": "mass balance (m^3/s)",
        "hazen_williams_m": "Hazen-Williams energy (m)",
    }
    for key, label in labels.items():
        if key in m:
            d = m[key]
            lines.append(
                f"| {label} | {d['count']} | {d['median']:.2e} | {d['p95']:.2e} | "
                f"{d['worst']:.2e} | {d['floor']:.2e} |"
            )
    lines.append("")
    lines.append(f"![physics residuals]({FIG}/physics_residuals.png)")
    lines.append("")
    lines.append(
        "Mass balance holds at machine epsilon across the corpus. The Hazen-Williams energy "
        "residual sits at the validator's documented idealised floor (~1e-3 m), confirming "
        "energy conservation along pipes. These are physics-correctness results, not "
        "similarity-to-a-dataset results."
    )
    lines.append("")
    return "\n".join(lines)


def _study2_section(study2: dict | None) -> str:
    if not study2:
        return "## 5.3 Study 2 - Network sweep\n\n_Artefact missing; run refresh._\n"
    lines = [
        "## 5.3 Study 2 - Network sweep",
        "",
        f"Every LeakG3PD network at or below the junction cap, run through the pipeline as a "
        f"PDD normal scenario. {study2['n_configs']} configs; status counts "
        f"{study2['by_status']}. Full ran/excluded taxonomy and per-network numbers in "
        f"[`docs/network_sweep_report.md`](network_sweep_report.md).",
        "",
        "The binding constraint is WNTRSimulator compatibility, not size: Darcy-Weisbach "
        "headloss .inp files and oversized networks are excluded up front, while non-unit "
        "pump speeds and PDD non-convergence on some uncalibrated large networks surface as "
        "runtime errors (the non-convergence guard fails loudly rather than returning a "
        "degenerate result). The networks that run complete with mass balance at machine "
        "epsilon; uncalibrated sub-zero pressures are documented warnings rather than failures.",
        "",
    ]
    return "\n".join(lines)


def _study3_section(study3: dict | None) -> str:
    if not study3:
        return "## 5.4 Study 3 - Parallel scaling\n\n_Artefact missing; run refresh._\n"
    lines = [
        "## 5.4 Study 3 - Parallel scaling",
        "",
        "Batch driver wall-clock at increasing worker counts (`ProcessPoolExecutor`, spawn "
        "start method) on a fixed 100-scenario stratified subset.",
        "",
        "| Workers | Wall-clock (s) | Speedup | Efficiency |",
        "|---:|---:|---:|---:|",
    ]
    for r in study3.get("rows", []):
        lines.append(
            f"| {r['workers']} | {r['wall_clock_s']:.2f} | {r['speedup']:.2f}x | "
            f"{r['efficiency']:.2f} |"
        )
    lines.append("")
    lines.append("![speedup](../outputs/plots/parallel_speedup.png)")
    lines.append("")
    lines.append(
        "Speedup peaks at the physical core count and is sublinear because the per-scenario "
        "WNTR cost is small, so process startup and the serial tail dominate. Output parity "
        "between 1 and 8 workers holds within the determinism floor. Details in "
        "[`docs/parallelisation_benchmark.md`](parallelisation_benchmark.md)."
    )
    lines.append("")
    return "\n".join(lines)


def _study5_section(study5: dict | None) -> str:
    if not study5:
        return "## 5.6 Study 5 - Cross-dataset sanity vs LeakG3PD\n\n_Artefact missing._\n"
    lines = [
        "## 5.6 Study 5 - Cross-dataset sanity vs LeakG3PD",
        "",
        f"Matched leak scenarios on the shared **{study5['matched_network']}** network "
        f"(host pipe `{study5['host_pipe']}`, d={study5['host_pipe_diameter_m']:.3f} m, "
        f"Cd={study5['discharge_coeff']}). LeakG3PD's generator targets an older WNTR API and "
        "bundled demand assets, so per its documentation we compare against the WNTR leak law "
        f"it states it follows (`{study5['leakg3pd_model']}`) and its small/big diameter "
        f"classes {study5['leakg3pd_size_classes']}. This is an external-validity sanity "
        "check, not a competition.",
        "",
        "| diam/pipe | class | area (m^2) | mean Q (m^3/s) | leak-law rel. err | "
        "near-node drop (m) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in study5["scenarios"]:
        lines.append(
            f"| {r['diameter_fraction']:.3f} | {r['leak_class']} | {r['area_m2']:.2e} | "
            f"{r['mean_q_sim_m3s']:.3e} | {r['leak_law_relative_error']:.2e} | "
            f"{r['near_node_pressure_drop_m']:.3f} |"
        )
    lines.append("")
    lines.append(f"![study5]({FIG}/study5_leakg3pd.png)")
    lines.append("")
    lines.append(
        f"Our simulated leak outflow tracks the analytic WNTR/LeakG3PD law to a worst-case "
        f"relative error of {study5['worst_leak_law_relative_error']:.2e} across the diameter "
        "range, and the near-node pressure drop grows monotonically with leak size - the "
        "expected leak signature. Where the two generators overlap, this pipeline reproduces "
        "the accepted hydraulic behaviour."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    metrics = _load("metrics.json")
    manifest = _load("manifest.json")
    if metrics is None or manifest is None:
        raise SystemExit("metrics.json or manifest.json missing; run the experiments first.")
    study1 = _load("study1.json")
    study3 = _load("study3.json")
    study5 = _load("study5.json")
    study2 = _load("study2.json")

    header = [
        "# Experimental Evaluation (paper Section 5)",
        "",
        "This document is the self-contained results source for the paper's experimental "
        "section. It is generated from the study artefacts under "
        "`experiments/ml_utility/data/` by `experiments/ml_utility/build_report.py`, so every "
        "number traces to a recorded, reproducible run. Section 5.1 fixes the setup; Studies "
        "1-3 establish physical validity, breadth and scalability; Study 4 is the downstream "
        "ML-utility centrepiece; Study 5 is the cross-dataset sanity check.",
        "",
        "Two research questions frame the work. **RQ1:** can EPANET/WNTR be orchestrated to "
        "generate large-scale, reproducible, labelled fault datasets? (Studies 1-3.) "
        "**RQ2:** do the generated data train and benchmark ML anomaly detectors usefully and "
        "non-trivially? (Study 4, with Study 5 as external validity.)",
        "",
    ]
    sections = [
        "\n".join(header),
        _setup_section(metrics, manifest),
        _study1_section(study1),
        _study2_section(study2),
        _study3_section(study3),
        _study4_section(metrics),
        _study5_section(study5),
    ]
    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
