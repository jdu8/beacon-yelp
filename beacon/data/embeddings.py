import numpy as np
from omegaconf import DictConfig
from beacon.training.reweighting import build_similarity_matrix, apply_topk_mask


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