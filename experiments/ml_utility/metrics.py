"""Anomaly-detection evaluation metrics for the ML-utility study.

The anomaly-detection literature has repeatedly shown that the *naive
point-adjusted F1* score overstates performance: a single correctly flagged
point inside a long anomaly segment is rewarded as if the whole segment were
detected (Kim et al. 2022; Wu and Keogh 2021). This module therefore reports a
panel of metrics and treats point-adjusted F1 as a deliberately-flawed baseline
shown only alongside honest alternatives:

- **Threshold-free:** ``auc_pr`` (average precision, preferred over ROC for rare
  anomalies) and ``auc_roc``. These need no threshold, so they cannot be gamed
  by threshold tuning.
- **Event / range-aware:** ``range_precision_recall`` implements the
  range-based precision and recall of Tatbul et al. (NeurIPS 2018, "Precision
  and Recall for Time Series"), which rewards overlap with whole anomaly
  *ranges* and penalises fragmented predictions. This is the honest event-level
  metric.
- **Point-adjusted F1 (reported with a caveat):** ``point_adjusted_f1`` returns
  both the raw point-wise F1 and the point-adjusted variant, so the report can
  show the inflation explicitly.
- **Detection delay:** ``detection_delay`` measures timesteps from anomaly onset
  to first alarm, an operationally meaningful quantity for water networks.

All functions are pure NumPy plus scikit-learn for the two AUC metrics, are
deterministic, and are unit-tested against hand-computed cases in
``test_metrics.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score

# -- range helpers ---------------------------------------------------------


def binary_to_ranges(mask: npt.ArrayLike) -> list[tuple[int, int]]:
    """Convert a 0/1 sequence to a list of half-open ``[start, end)`` ranges.

    Args:
        mask: 1-D array of 0/1 (or bool) values.

    Returns:
        Contiguous runs of truthy values as ``(start, end)`` index pairs,
        ``end`` exclusive. ``[0, 1, 1, 0, 1]`` becomes ``[(1, 3), (4, 5)]``.
    """

    m = np.asarray(mask).astype(bool).astype(int)
    if m.size == 0:
        return []
    padded = np.concatenate([[0], m, [0]])
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _overlap_size(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Size of the intersection of two half-open ranges (0 if disjoint)."""

    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _cardinality_factor(n_overlaps: int, mode: str) -> float:
    """Penalise a single range that overlaps several ranges on the other side.

    ``reciprocal`` (Tatbul's default) returns ``1 / n_overlaps`` so a prediction
    fragmented across many true ranges (or vice versa) is discounted. ``one``
    disables the penalty.
    """

    if n_overlaps <= 1:
        return 1.0
    if mode == "reciprocal":
        return 1.0 / n_overlaps
    return 1.0


def _range_term(
    target: tuple[int, int], others: list[tuple[int, int]], cardinality: str
) -> tuple[float, float]:
    """One range's (existence, overlap-reward) against the other range set.

    The overlap reward uses a *flat* positional bias (every position in the
    range counts equally), which is the most common and least opinionated
    choice. The reward is the fraction of ``target`` covered by ``others``,
    scaled by the cardinality factor and capped at 1.
    """

    overlapping = [o for o in others if _overlap_size(target, o) > 0]
    existence = 1.0 if overlapping else 0.0
    if not overlapping:
        return existence, 0.0
    target_len = target[1] - target[0]
    covered = sum(_overlap_size(target, o) for o in overlapping)
    gamma = _cardinality_factor(len(overlapping), cardinality)
    overlap_reward = min(1.0, gamma * covered / target_len)
    return existence, overlap_reward


def range_precision_recall(
    pred_masks: list[np.ndarray],
    true_masks: list[np.ndarray],
    alpha: float = 0.0,
    cardinality: str = "reciprocal",
) -> dict[str, float]:
    """Range-based precision, recall and F1 (Tatbul et al. 2018).

    Computed per scenario (a range never spans a scenario boundary) and pooled
    so every true range contributes equally to recall and every predicted range
    contributes equally to precision.

    Args:
        pred_masks: Per-scenario predicted-anomaly 0/1 sequences.
        true_masks: Per-scenario ground-truth 0/1 sequences (same shapes).
        alpha: Existence-reward weight for recall in ``[0, 1]``. ``0.0``
            (default) makes recall purely overlap-based; raising it rewards
            merely *touching* a true range. Precision has no existence term.
        cardinality: ``reciprocal`` (default) or ``one`` (see
            :func:`_cardinality_factor`).

    Returns:
        ``{"precision", "recall", "f1"}``. Recall is ``nan`` if there are no
        true ranges; precision is ``nan`` if the detector predicted nothing; F1
        is ``0.0`` whenever either side is ``nan`` (a detector that never fires,
        or one evaluated with no anomalies, scores no usable F1).
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    recall_terms: list[float] = []
    precision_terms: list[float] = []
    for pred, true in zip(pred_masks, true_masks, strict=True):
        pred_ranges = binary_to_ranges(pred)
        true_ranges = binary_to_ranges(true)
        for r in true_ranges:
            existence, overlap = _range_term(r, pred_ranges, cardinality)
            recall_terms.append(alpha * existence + (1.0 - alpha) * overlap)
        for p in pred_ranges:
            # Precision has no existence reward (alpha fixed to 0 for precision
            # in Tatbul's formulation).
            _, overlap = _range_term(p, true_ranges, cardinality)
            precision_terms.append(overlap)

    recall = float(np.mean(recall_terms)) if recall_terms else float("nan")
    precision = float(np.mean(precision_terms)) if precision_terms else float("nan")
    if np.isnan(recall) or np.isnan(precision) or (precision + recall) == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# -- point metrics ---------------------------------------------------------


def _point_prf(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    """Element-wise precision, recall, F1 over a pooled 0/1 sequence."""

    pred = np.asarray(pred).astype(bool)
    true = np.asarray(true).astype(bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def point_adjusted_f1(
    pred_masks: list[np.ndarray], true_masks: list[np.ndarray]
) -> dict[str, float]:
    """Raw and point-adjusted point-wise precision/recall/F1.

    Point adjustment (Xu et al. 2018): for every ground-truth anomaly range, if
    *any* point inside it is predicted positive, the entire range is counted as
    correctly detected. This is the protocol known to overstate performance; it
    is returned here only so the report can show the gap against the raw point
    F1. **Do not read the point-adjusted numbers as the headline result.**

    Returns:
        ``{"point_f1", "point_precision", "point_recall", "pa_f1",
        "pa_precision", "pa_recall"}``.
    """

    pred_pool: list[np.ndarray] = []
    adj_pool: list[np.ndarray] = []
    true_pool: list[np.ndarray] = []
    for pred, true in zip(pred_masks, true_masks, strict=True):
        pred = np.asarray(pred).astype(bool)
        true = np.asarray(true).astype(bool)
        adjusted = pred.copy()
        for start, end in binary_to_ranges(true):
            if pred[start:end].any():
                adjusted[start:end] = True
        pred_pool.append(pred)
        adj_pool.append(adjusted)
        true_pool.append(true)

    raw_p, raw_r, raw_f1 = _point_prf(np.concatenate(pred_pool), np.concatenate(true_pool))
    pa_p, pa_r, pa_f1 = _point_prf(np.concatenate(adj_pool), np.concatenate(true_pool))
    return {
        "point_precision": raw_p,
        "point_recall": raw_r,
        "point_f1": raw_f1,
        "pa_precision": pa_p,
        "pa_recall": pa_r,
        "pa_f1": pa_f1,
    }


# -- threshold-free --------------------------------------------------------


def auc_pr(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the precision-recall curve (average precision).

    Returns ``nan`` if ``labels`` is single-class (AP is undefined without both
    positives and negatives).
    """

    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return float("nan")
    return float(average_precision_score(labels, np.asarray(scores, dtype=float)))


def auc_roc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the ROC curve. ``nan`` if ``labels`` is single-class."""

    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=float)))


# -- detection delay -------------------------------------------------------


def detection_delay(
    window_end_times: npt.ArrayLike,
    window_labels: npt.ArrayLike,
    window_scores: npt.ArrayLike,
    threshold: float,
    onset_time: float,
) -> float | None:
    """Delay (in time units) from anomaly onset to the first true alarm.

    A streaming detector observes a window's data up to its end timestamp, so a
    window is treated as raising its alarm at ``window_end_times``. The delay is
    the gap between the onset and the end time of the first window that is (a)
    genuinely anomalous (``window_labels == 1``) and (b) scored above
    ``threshold``. Windows are considered in time order.

    Args:
        window_end_times: End timestamp of each window (same unit as
            ``onset_time``, e.g. seconds).
        window_labels: 0/1 window-level ground-truth labels.
        window_scores: Anomaly score per window.
        threshold: Alarm threshold; a window fires when its score exceeds it.
        onset_time: Timestamp of the anomaly onset.

    Returns:
        Non-negative delay in the same unit as the timestamps, or ``None`` if
        the anomaly is never detected (a miss). ``0.0`` means the first
        anomalous window already fired at or before the onset timestamp.
    """

    end_times = np.asarray(window_end_times, dtype=float)
    labels = np.asarray(window_labels).astype(bool)
    scores = np.asarray(window_scores, dtype=float)
    order = np.argsort(end_times, kind="stable")
    for i in order:
        if labels[i] and scores[i] > threshold:
            return max(0.0, float(end_times[i]) - float(onset_time))
    return None
