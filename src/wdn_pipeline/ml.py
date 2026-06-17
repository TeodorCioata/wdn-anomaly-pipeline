"""Machine-learning helpers over the pipeline DuckDB datasets.

This module gives an ML consumer a ready windowed dataset and a leakage
safe train/test split without re-implementing the slicing logic. It sits
on top of :class:`wdn_pipeline.query.DatasetQuery`, so it reads the same
consolidated long tables the rest of the pipeline writes.

Two pieces:

- :func:`scenario_train_test_split` splits a list of scenarios into train
  and test sets **at the scenario level**. Anomaly detection models must
  never see timesteps from the same scenario in both splits (the windows
  overlap in time and share the same hydraulic realisation), so the split
  is by whole scenario, deterministically, via a stable hash of the
  scenario name. The same scenarios and seed always produce the same
  split, independent of input order.
- :class:`WDNWindowDataset` is a PyTorch ``Dataset`` of fixed-length
  sliding windows over one measured quantity (pressure by default), with
  a window-level anomaly label (1 if any timestep in the window is
  anomalous). PyTorch is an optional dependency (the ``ml`` extra); the
  module still imports without it so :func:`scenario_train_test_split`
  is always available.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from wdn_pipeline.query import DatasetQuery

try:  # torch is optional (the `ml` extra).
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    _TorchDataset = object  # type: ignore[assignment, misc]
    _HAS_TORCH = False


def _unit_interval_hash(name: str, seed: int) -> float:
    """Map a scenario name to a stable float in ``[0, 1)``.

    Uses SHA-256 (not Python's salted ``hash``) so the value is identical
    across processes and runs, which is what makes the split reproducible.
    """

    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def scenario_train_test_split(
    scenarios: list[str], test_fraction: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split scenarios into train and test sets at the scenario level.

    Args:
        scenarios: Scenario basenames (e.g. from
            :meth:`DatasetQuery.scenarios`).
        test_fraction: Target fraction assigned to the test set. Each
            scenario is independently assigned, so the realised fraction
            approaches ``test_fraction`` as the scenario count grows.
        seed: Seed mixed into the hash so different seeds give different
            (but each reproducible) splits.

    Returns:
        A ``(train, test)`` pair of disjoint scenario lists whose union is
        ``scenarios`` (order preserved within each list).

    Raises:
        ValueError: If ``test_fraction`` is outside ``[0, 1]``.
    """

    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be in [0, 1]")
    train: list[str] = []
    test: list[str] = []
    for name in scenarios:
        if _unit_interval_hash(name, seed) < test_fraction:
            test.append(name)
        else:
            train.append(name)
    return train, test


class WDNWindowDataset(_TorchDataset):
    """Sliding-window PyTorch dataset over one measured quantity.

    Each item is a ``(window, label)`` pair:

    - ``window`` is a ``float32`` tensor of shape
      ``(window_length, n_channels)`` holding consecutive timesteps of the
      chosen quantity for one scenario.
    - ``label`` is a scalar ``float32`` tensor, ``1.0`` if any timestep in
      the window is anomalous (per the pipeline ``label`` column) and
      ``0.0`` otherwise.

    Windows never cross a scenario boundary. Construct the dataset from a
    train or test scenario list produced by
    :func:`scenario_train_test_split` to keep the splits leakage-free.

    Requires PyTorch (the ``ml`` extra).
    """

    def __init__(
        self,
        db_path: str | Path,
        scenarios: list[str],
        quantity: str = "pressure",
        channels: list[str] | None = None,
        window_length: int = 24,
        stride: int = 1,
        max_scenarios: int | None = None,
    ) -> None:
        """Build the window index by reading each scenario once.

        Every scenario's feature matrix is read eagerly into memory in the
        constructor. The matrices are small (one scenario is roughly
        ``timesteps x channels x 4`` bytes, e.g. ~9 KB for a 24-step
        90-channel pressure run), so even thousands of scenarios fit
        comfortably. For pathological cases (very long horizons or very wide
        networks, or simply to bound memory on a shared machine) set
        ``max_scenarios``; past it the constructor raises rather than
        silently allocating. Because the train/test split is at the
        scenario level, building several datasets over scenario subsets
        stays leakage-free.

        Args:
            db_path: Path to a pipeline DuckDB file.
            scenarios: Scenario basenames to include.
            quantity: One of ``pressure`` / ``flowrate`` / ``demand`` /
                ``quality``. ``flowrate`` is link-addressed; the others are
                node-addressed.
            channels: Node or link names to keep, or ``None`` for all
                channels present in the scenario.
            window_length: Number of consecutive timesteps per window.
            stride: Step between consecutive window starts.
            max_scenarios: Optional cap on the number of scenarios loaded
                eagerly. ``None`` (default) means no cap, preserving the
                original behaviour. When set and exceeded the constructor
                raises with an actionable message.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: On a non-positive ``window_length`` or ``stride``,
                or when ``len(scenarios)`` exceeds ``max_scenarios``.
        """

        if not _HAS_TORCH:
            raise ImportError(
                "WDNWindowDataset requires PyTorch. Install it with "
                "'pip install wdn-anomaly-pipeline[ml]'."
            )
        if window_length <= 0 or stride <= 0:
            raise ValueError("window_length and stride must be positive")
        if max_scenarios is not None and len(scenarios) > max_scenarios:
            raise ValueError(
                f"WDNWindowDataset would eagerly load {len(scenarios)} scenarios into "
                f"memory, above the configured max_scenarios={max_scenarios}. Raise the "
                "cap if the machine has the memory, or build several datasets over "
                "scenario subsets (the split is at the scenario level, so subsets stay "
                "leakage-free)."
            )

        self.window_length = window_length
        self.stride = stride
        self.quantity = quantity
        self._features: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []

        with DatasetQuery(db_path) as q:
            for scenario in scenarios:
                if quantity == "flowrate":
                    matrix = q.flowrate(scenario=scenario, links=channels, wide=True)
                else:
                    getter = getattr(q, quantity)
                    matrix = getter(scenario=scenario, nodes=channels, wide=True)
                if matrix.empty:
                    continue
                matrix = matrix.sort_index()
                labels = q.labels(scenario=scenario)
                label_series = (
                    labels.set_index("time_seconds")["label"].reindex(matrix.index).fillna(0)
                )

                features = matrix.to_numpy(dtype=np.float32)
                label_values = label_series.to_numpy(dtype=np.float32)
                n_rows = features.shape[0]
                if n_rows < window_length:
                    continue
                scenario_idx = len(self._features)
                self._features.append(features)
                self._labels.append(label_values)
                for start in range(0, n_rows - window_length + 1, stride):
                    self._index.append((scenario_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        scenario_idx, start = self._index[idx]
        end = start + self.window_length
        window = self._features[scenario_idx][start:end]
        window_labels = self._labels[scenario_idx][start:end]
        label = 1.0 if float(window_labels.max()) > 0.0 else 0.0
        return torch.from_numpy(window.copy()), torch.tensor(label, dtype=torch.float32)
