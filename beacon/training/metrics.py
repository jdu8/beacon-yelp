import numpy as np
from sklearn.metrics import cohen_kappa_score


def compute_qwk(preds: list, labels: list) -> float:
    preds_rounded = np.clip(np.round(preds), 1, 5).astype(int)
    labels_int    = np.array(labels).astype(int)
    return cohen_kappa_score(labels_int, preds_rounded, weights="quadratic")


def compute_mse(preds: list, labels: list) -> float:
    return float(np.mean((np.array(preds) - np.array(labels)) ** 2))


def compute_per_star_metrics(
    losses: np.ndarray,
    labels: np.ndarray,
) -> dict:
    return {
        int(star): {
            "mean_loss": float(losses[labels == star].mean()),
            "n":         int((labels == star).sum()),
        }
        for star in [1, 2, 3, 4, 5]
        if (labels == star).sum() > 0
    }