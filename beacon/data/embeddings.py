import numpy as np
from omegaconf import DictConfig
from beacon.training.reweighting import build_similarity_matrix, apply_topk_mask


def build_reverse_topk_matrix(
    train_embs: np.ndarray,
    guide_embs: np.ndarray,
    cfg: DictConfig,
) -> np.ndarray:
    """
    Reverse direction: for each guide point, find K nearest train points
    and push guide loss to them.

    Returns:
        matrix — np.ndarray [n_train, n_guide] (transposed for downstream compat)
    """
    k = cfg.reweighting.get('k', 50)
    reverse_mode = cfg.reweighting.get('reverse_mode', 'uniform')

    # (n_guide, n_train) — each row is one guide point's sim to all train points
    sim_matrix = build_similarity_matrix(guide_embs, train_embs)
    n_guide, n_train = sim_matrix.shape
    k_actual = min(int(k), n_train)

    topk_idx = np.argpartition(sim_matrix, -k_actual, axis=1)[:, -k_actual:]
    rows = np.arange(n_guide)[:, None]

    masked = np.zeros_like(sim_matrix)

    if reverse_mode == 'inverse':
        # push MORE to less-similar neighbors among top-K
        inv_sim = 1.0 - sim_matrix
        inv_sim = np.clip(inv_sim, 1e-8, None)
        masked[rows, topk_idx] = inv_sim[rows, topk_idx]
        row_sums = masked.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        masked = masked / row_sums
    else:  # uniform
        # push equally to all K neighbors
        masked[rows, topk_idx] = 1.0 / k_actual

    # transpose: (n_train, n_guide) so topk_matrix @ guide_losses -> (n_train,)
    result = masked.T

    row_sums = result.sum(axis=1)
    print(f"  reverse_topk: {result.shape} | k={k} | mode={reverse_mode} | "
          f"row_sum min={row_sums.min():.4f} max={row_sums.max():.4f}")

    return result


def build_topk_matrix(
    train_embs: np.ndarray,
    guide_embs: np.ndarray,
    cfg: DictConfig,
) -> np.ndarray:
    """
    Build row-normalized top-K similarity matrix.
    Computed once before training, reused every epoch.

    Returns:
        topk_matrix — np.ndarray [n_train, n_guide], row-normalized
    """
    k = cfg.reweighting.get('k', 50)

    sim_matrix  = build_similarity_matrix(train_embs, guide_embs)
    topk_matrix = apply_topk_mask(sim_matrix, k)

    # Sanity check — uniform guide losses should give all weights = 1.0
    dummy     = np.ones(guide_embs.shape[0])
    test_w    = topk_matrix @ dummy
    assert np.allclose(test_w, 1.0, atol=1e-5), \
        f"Row normalization failed: min={test_w.min():.4f} max={test_w.max():.4f}"

    print(f"  topk_matrix: {topk_matrix.shape} | k={k} | "
          f"sanity min={test_w.min():.4f} max={test_w.max():.4f}")

    return topk_matrix