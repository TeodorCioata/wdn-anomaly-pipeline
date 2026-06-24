"""Anomaly detectors for the ML-utility study.

Three detectors spanning the dominant water-distribution anomaly-detection
paradigms. They are deliberately small and their hyperparameters are fixed up
front (recorded in :data:`HYPERPARAMS`): the point of the study is to
characterise the *dataset*, not to win an AD competition, so nothing is tuned to
flatter the results.

- :class:`ZScoreDetector` - a cheap, interpretable statistical baseline.
  Per-channel standardisation against the training distribution; the window
  anomaly score is the largest absolute deviation (in standard deviations) over
  the window. Strong simple baselines often rival deep models in this field, so
  it is included for honesty.
- :class:`LSTMForecaster` - one-step-ahead multivariate forecaster; the anomaly
  score is the forecast residual. This is the dominant WDN-AD paradigm.
- :class:`LSTMAutoencoder` - reconstructs the input window; the anomaly score is
  the reconstruction error. This mirrors published SCADA leak-detection
  autoencoder work.

Every detector exposes the same interface so the runner treats them uniformly::

    det = LSTMForecaster(n_channels=97, seed=0)
    det.fit(train_windows)              # (N, L, C) float32, normal data only
    scores = det.score(eval_windows)    # (M,) higher = more anomalous

All standardisation statistics are fit on the (normal-only) training windows, so
the detectors are semi-supervised, which is the standard WDN-AD setting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# Fixed hyperparameters, recorded verbatim in the report. Not tuned per run.
# Typed constants are the single source of truth for the hyperparameters; the
# HYPERPARAMS dict below is built from them purely for recording in the report.
WINDOW_LENGTH = 24
STRIDE = 1
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
STD_FLOOR = 1e-8

HYPERPARAMS: dict[str, object] = {
    "window_length": WINDOW_LENGTH,
    "stride": STRIDE,
    "lstm_hidden": LSTM_HIDDEN,
    "lstm_layers": LSTM_LAYERS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "loss": "MSE",
    "std_floor": STD_FLOOR,
}


@dataclass
class _Standardizer:
    """Per-channel mean/std fit on training windows.

    A floor on the std keeps constant channels (e.g. reservoir pressure) from
    producing infinities; such channels simply contribute zero deviation.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, windows: np.ndarray) -> _Standardizer:
        flat = windows.reshape(-1, windows.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < STD_FLOOR, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return ((windows - self.mean) / self.std).astype(np.float32)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class ZScoreDetector:
    """Statistical baseline: max absolute standardised deviation per window."""

    name = "zscore"

    def __init__(self, n_channels: int, seed: int = 0) -> None:
        self.n_channels = n_channels
        self.seed = seed
        self._scaler: _Standardizer | None = None

    def fit(self, train_windows: np.ndarray) -> None:
        self._scaler = _Standardizer.fit(train_windows)

    def score(self, windows: np.ndarray) -> np.ndarray:
        assert self._scaler is not None, "fit() must be called before score()"
        z = np.abs(self._scaler.transform(windows))
        # Largest deviation anywhere in the window -> the window's anomaly score.
        return z.reshape(windows.shape[0], -1).max(axis=1).astype(np.float64)


class _LSTMForecastNet(nn.Module):
    def __init__(self, n_channels: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out)


class _LSTMAutoencoderNet(nn.Module):
    def __init__(self, n_channels: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_channels, hidden, num_layers=layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        latent = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        dec, _ = self.decoder(latent)
        return self.head(dec)


class _TorchDetector:
    """Shared train/score machinery for the two neural detectors."""

    name = "torch"

    def __init__(self, n_channels: int, seed: int = 0) -> None:
        self.n_channels = n_channels
        self.seed = seed
        self._scaler: _Standardizer | None = None
        self._net: nn.Module | None = None

    # Subclasses implement these two.
    def _build_net(self) -> nn.Module:  # pragma: no cover - overridden
        raise NotImplementedError

    def _residual(self, net: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        """Per-window mean squared error (shape ``(batch,)``)."""

        raise NotImplementedError  # pragma: no cover - overridden

    def fit(self, train_windows: np.ndarray) -> None:
        _set_seed(self.seed)
        self._scaler = _Standardizer.fit(train_windows)
        x = torch.from_numpy(self._scaler.transform(train_windows))
        net = self._build_net()
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        batch_size = BATCH_SIZE
        epochs = EPOCHS
        n = x.shape[0]
        net.train()
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch = x[idx]
                opt.zero_grad()
                loss = self._residual(net, batch).mean()
                loss.backward()
                opt.step()
        net.eval()
        self._net = net

    def score(self, windows: np.ndarray) -> np.ndarray:
        assert self._scaler is not None and self._net is not None, "fit() before score()"
        x = torch.from_numpy(self._scaler.transform(windows))
        scores: list[np.ndarray] = []
        batch_size = BATCH_SIZE
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch = x[start : start + batch_size]
                scores.append(self._residual(self._net, batch).cpu().numpy())
        return np.concatenate(scores).astype(np.float64)


class LSTMForecaster(_TorchDetector):
    """One-step-ahead forecaster; score = mean one-step forecast MSE."""

    name = "lstm_forecaster"

    def _build_net(self) -> nn.Module:
        return _LSTMForecastNet(
            self.n_channels,
            LSTM_HIDDEN,
            LSTM_LAYERS,
        )

    def _residual(self, net: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        # Predict step t+1 from steps up to t.
        pred = net(batch[:, :-1, :])
        target = batch[:, 1:, :]
        return ((pred - target) ** 2).mean(dim=(1, 2))


class LSTMAutoencoder(_TorchDetector):
    """Sequence autoencoder; score = mean reconstruction MSE."""

    name = "lstm_autoencoder"

    def _build_net(self) -> nn.Module:
        return _LSTMAutoencoderNet(
            self.n_channels,
            LSTM_HIDDEN,
            LSTM_LAYERS,
        )

    def _residual(self, net: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        recon = net(batch)
        return ((recon - batch) ** 2).mean(dim=(1, 2))


# Detector registry consumed by run_experiments.
DETECTORS: dict[str, type] = {
    ZScoreDetector.name: ZScoreDetector,
    LSTMForecaster.name: LSTMForecaster,
    LSTMAutoencoder.name: LSTMAutoencoder,
}
