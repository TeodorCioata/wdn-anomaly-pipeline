"""Unit tests for the inline anomaly-detection metrics.

Run standalone (``python experiments/ml_utility/test_metrics.py``) or under
pytest. These pin the range-aware and point-adjusted behaviour to hand-computed
values so the metric implementations can be defended in the paper.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.ml_utility.metrics import (
    auc_pr,
    auc_roc,
    binary_to_ranges,
    detection_delay,
    point_adjusted_f1,
    range_precision_recall,
)


def test_binary_to_ranges() -> None:
    assert binary_to_ranges([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    assert binary_to_ranges([1, 1, 1]) == [(0, 3)]
    assert binary_to_ranges([0, 0]) == []
    assert binary_to_ranges([]) == []


def test_range_perfect() -> None:
    mask = np.ones(10)
    out = range_precision_recall([mask], [mask])
    assert math.isclose(out["precision"], 1.0)
    assert math.isclose(out["recall"], 1.0)
    assert math.isclose(out["f1"], 1.0)


def test_range_partial_overlap() -> None:
    true = np.ones(10)
    pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    out = range_precision_recall([pred], [true])
    # Prediction fully inside the true range -> precision 1; covers half -> recall 0.5.
    assert math.isclose(out["precision"], 1.0)
    assert math.isclose(out["recall"], 0.5)
    assert math.isclose(out["f1"], 2 * 1.0 * 0.5 / 1.5)


def test_range_cardinality_penalty() -> None:
    true = np.zeros(12)
    true[0:10] = 1
    pred = np.zeros(12)
    pred[0:2] = 1
    pred[4:6] = 1
    out = range_precision_recall([pred], [true])
    # Two predicted fragments hit one true range -> reciprocal cardinality 0.5,
    # covered 4 of 10 -> recall 0.5 * 4 / 10 = 0.2. Each fragment sits inside the
    # true range -> precision 1.0.
    assert math.isclose(out["recall"], 0.2)
    assert math.isclose(out["precision"], 1.0)


def test_range_no_prediction() -> None:
    true = np.ones(5)
    pred = np.zeros(5)
    out = range_precision_recall([pred], [true])
    assert out["recall"] == 0.0
    assert math.isnan(out["precision"])
    assert out["f1"] == 0.0


def test_point_adjusted_inflation() -> None:
    true = np.array([1, 1, 1, 1, 1])
    pred = np.array([1, 0, 0, 0, 0])
    out = point_adjusted_f1([pred], [true])
    # Raw: 1 of 5 true points -> recall 0.2. PA: one hit floods the range -> 1.0.
    assert math.isclose(out["point_recall"], 0.2)
    assert math.isclose(out["point_f1"], 2 * 1.0 * 0.2 / 1.2)
    assert math.isclose(out["pa_recall"], 1.0)
    assert math.isclose(out["pa_f1"], 1.0)


def test_auc_pr_perfect_and_roc_single_class() -> None:
    assert math.isclose(auc_pr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
    assert math.isnan(auc_roc([1, 1, 1], [0.1, 0.5, 0.9]))
    assert math.isclose(auc_roc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)


def test_detection_delay() -> None:
    ends = np.array([10, 20, 30, 40])
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.1, 0.9, 0.9])
    assert detection_delay(ends, labels, scores, threshold=0.5, onset_time=25) == 5.0
    # Never crosses the threshold on an anomalous window -> miss.
    assert detection_delay(ends, labels, np.array([0.1, 0.1, 0.2, 0.2]), 0.5, 25) is None
    # Fires before the onset timestamp -> floored at zero.
    assert detection_delay(ends, labels, scores, threshold=0.5, onset_time=35) == 0.0


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} metric tests passed")


if __name__ == "__main__":
    _run_all()
