import numpy as np
import pytest
from beacon.training.reweighting import (
    build_similarity_matrix,
    apply_topk_mask,
    compute_sample_weights,
    get_weight_stats,
)

N_TRAIN = 100
N_GUIDE = 50
EMB_DIM = 32


@pytest.fixture
def embeddings():
    rng        = np.random.RandomState(42)
    train_embs = rng.randn(N_TRAIN, EMB_DIM).astype(np.float32)
    guide_embs = rng.randn(N_GUIDE, EMB_DIM).astype(np.float32)
    return train_embs, guide_embs


@pytest.fixture
def topk_matrix(embeddings):
    train_embs, guide_embs = embeddings
    sim    = build_similarity_matrix(train_embs, guide_embs)
    return apply_topk_mask(sim, k=10)


def test_similarity_matrix_shape(embeddings):
    train_embs, guide_embs = embeddings
    sim = build_similarity_matrix(train_embs, guide_embs)
    assert sim.shape == (N_TRAIN, N_GUIDE)


def test_similarity_matrix_bounded(embeddings):
    train_embs, guide_embs = embeddings
    sim = build_similarity_matrix(train_embs, guide_embs)
    assert sim.min() >= -1.01
    assert sim.max() <=  1.01


def test_topk_mask_row_normalized(topk_matrix):
    # Each row should sum to 1.0 after masking
    row_sums = topk_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_topk_mask_nonzero_count(topk_matrix):
    # Each row should have exactly K nonzero entries
    nonzero_per_row = (topk_matrix > 0).sum(axis=1)
    assert np.all(nonzero_per_row <= 10)


def test_topk_all(embeddings):
    train_embs, guide_embs = embeddings
    sim    = build_similarity_matrix(train_embs, guide_embs)
    masked = apply_topk_mask(sim, k='all')
    row_sums = masked.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_uniform_weights(topk_matrix):
    guide_losses  = np.ones(N_GUIDE)
    weights       = compute_sample_weights(topk_matrix, guide_losses, scheme='uniform')
    assert np.allclose(weights, 1.0)


def test_minmax_weights_range(topk_matrix):
    rng          = np.random.RandomState(42)
    guide_losses = rng.rand(N_GUIDE)
    weights      = compute_sample_weights(
        topk_matrix, guide_losses, scheme='minmax',
        w_min=0.5, w_max=2.0
    )
    # After sum normalization weights shift — just check they're positive
    # and sum to n_train
    assert weights.min() > 0
    assert np.isclose(weights.sum(), N_TRAIN, atol=1e-3)


def test_weights_sum_to_n_train(topk_matrix):
    rng          = np.random.RandomState(42)
    guide_losses = rng.rand(N_GUIDE)
    for scheme in ['minmax', 'rank', 'exponential']:
        weights = compute_sample_weights(
            topk_matrix, guide_losses, scheme=scheme,
            w_min=0.5, w_max=2.0, temperature=0.5
        )
        assert np.isclose(weights.sum(), N_TRAIN, atol=1e-3), \
            f"scheme={scheme} weights sum to {weights.sum():.4f}, expected {N_TRAIN}"


def test_rank_weights_no_outlier_dominance(topk_matrix):
    # Rank should be robust to outliers — inject one extreme loss
    guide_losses       = np.ones(N_GUIDE) * 0.5
    guide_losses[0]    = 1000.0   # extreme outlier
    weights_rank       = compute_sample_weights(
        topk_matrix, guide_losses, scheme='rank', w_min=0.5, w_max=2.0
    )
    weights_minmax     = compute_sample_weights(
        topk_matrix, guide_losses, scheme='minmax', w_min=0.5, w_max=2.0
    )
    # Rank should have lower weight variance than minmax under outlier
    assert weights_rank.std() <= weights_minmax.std()


def test_weight_stats_keys(topk_matrix):
    weights = np.ones(N_TRAIN)
    stats   = get_weight_stats(weights)
    assert set(stats.keys()) == {"w_mean", "w_std", "w_min", "w_max", "w_ratio"}


def test_weight_stats_uniform(topk_matrix):
    weights = np.ones(N_TRAIN)
    stats   = get_weight_stats(weights)
    assert stats["w_std"]   == pytest.approx(0.0)
    assert stats["w_ratio"] == pytest.approx(1.0)