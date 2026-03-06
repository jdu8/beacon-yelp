import numpy as np
from typing import Union


def build_similarity_matrix(
    train_embs: np.ndarray,
    guide_embs: np.ndarray,
) -> np.ndarray:
    train_norm = train_embs / np.linalg.norm(train_embs, axis=1, keepdims=True)
    guide_norm = guide_embs / np.linalg.norm(guide_embs, axis=1, keepdims=True)
    return train_norm @ guide_norm.T   # [n_train, n_guide]


def apply_topk_mask(
    sim_matrix: np.ndarray,
    k: Union[int, str],
) -> np.ndarray:
    n_train, n_guide = sim_matrix.shape
    k_actual = n_guide if k == 'all' else min(int(k), n_guide)

    masked = np.zeros_like(sim_matrix)

    if k == 'all':
        masked = np.clip(sim_matrix.copy(), 0, None)
    else:
        topk_idx = np.argpartition(sim_matrix, -k_actual, axis=1)[:, -k_actual:]
        rows     = np.arange(n_train)[:, None]
        masked[rows, topk_idx] = sim_matrix[rows, topk_idx]
        masked   = np.clip(masked, 0, None)

    row_sums = masked.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return masked / row_sums


def compute_sample_weights(
    topk_matrix: np.ndarray,
    guide_losses: np.ndarray,
    scheme: str,
    **kwargs,
) -> np.ndarray:
    n_train = topk_matrix.shape[0]

    if scheme == 'uniform':
        return np.ones(n_train)

    if scheme == 'rank':
        ranks       = guide_losses.argsort().argsort().astype(float)
        ranks       = ranks / ranks.max()
        raw_weights = topk_matrix @ ranks

    elif scheme == 'exponential':
        temperature = float(kwargs.get('temperature', 0.5))
        signal      = np.exp(guide_losses / temperature)
        raw_weights = topk_matrix @ signal

    else:  # minmax — both soft and hard
        raw_weights = topk_matrix @ guide_losses

    w_min = float(kwargs.get('w_min', 0.5))
    w_max = float(kwargs.get('w_max', 2.0))

    r_min = raw_weights.min()
    r_max = raw_weights.max()

    if r_max > r_min:
        raw_weights = w_min + (w_max - w_min) * (raw_weights - r_min) / (r_max - r_min)
    else:
        raw_weights = np.ones(n_train)

    return raw_weights / raw_weights.sum() * n_train


def get_weight_stats(sample_weights: np.ndarray) -> dict:
    return {
        "w_mean": float(sample_weights.mean()),
        "w_std":  float(sample_weights.std()),
        "w_min":  float(sample_weights.min()),
        "w_max":  float(sample_weights.max()),
        "w_ratio": float(sample_weights.max() / max(sample_weights.min(), 1e-8)),
    }