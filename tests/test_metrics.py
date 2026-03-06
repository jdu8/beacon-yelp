import numpy as np
import pytest
from beacon.training.metrics import compute_qwk, compute_mse, compute_per_star_metrics


def test_qwk_perfect():
    labels = [1, 2, 3, 4, 5]
    assert compute_qwk(labels, labels) == 1.0


def test_qwk_worst():
    preds  = [1, 1, 1, 1, 1]
    labels = [5, 5, 5, 5, 5]
    assert compute_qwk(preds, labels) <= 0


def test_qwk_clips_predictions():
    # Predictions outside [1,5] should be clipped not crash
    preds  = [0.2, 5.8, 3.0, 1.1, 4.9]
    labels = [1,   5,   3,   1,   5  ]
    qwk = compute_qwk(preds, labels)
    assert -1 <= qwk <= 1


def test_mse_perfect():
    labels = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert compute_mse(labels, labels) == 0.0


def test_mse_known_value():
    preds  = [2.0, 2.0]
    labels = [1.0, 3.0]
    assert compute_mse(preds, labels) == 1.0


def test_per_star_metrics_keys():
    losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_per_star_metrics(losses, labels)
    assert set(result.keys()) == {1, 2, 3, 4, 5}


def test_per_star_metrics_values():
    losses = np.array([1.0, 1.0, 2.0, 2.0])
    labels = np.array([1.0, 1.0, 2.0, 2.0])
    result = compute_per_star_metrics(losses, labels)
    assert result[1]["mean_loss"] == pytest.approx(1.0)
    assert result[2]["mean_loss"] == pytest.approx(2.0)
    assert result[1]["n"] == 2
    assert result[2]["n"] == 2