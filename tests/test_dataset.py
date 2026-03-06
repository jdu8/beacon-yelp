import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_mock_cfg(guide_size=300, synth_size=500):
    cfg = MagicMock()
    cfg.paths.data_dir              = "/tmp/mock_data"
    cfg.data.guide_size             = guide_size
    cfg.data.synth_size             = synth_size
    cfg.data.synth_file             = "synthetic_500k.csv"
    cfg.data.synth_emb_file         = "embeddings_synthetic_500k.npy"
    cfg.data.real_train_file        = "restaurant_train_70k.json"
    cfg.data.real_test_file         = "restaurant_test_30k.json"
    cfg.data.real_emb_file          = "embeddings_base_70k.npy"
    return cfg


def test_stratified_sampling_equal_classes():
    """Each star class should have equal representation after sampling."""
    import pandas as pd
    from beacon.data.dataset import load_and_sample

    n_per_star  = 100
    n_stars     = 5
    total_synth = n_per_star * n_stars * 2   # 2x so sampling has room

    synth_df = pd.DataFrame({
        "full_text": [f"review {i}" for i in range(total_synth)],
        "stars":     [float((i % n_stars) + 1) for i in range(total_synth)],
    })
    synth_embs  = np.random.randn(total_synth, 32).astype(np.float32)

    guide_df = pd.DataFrame({
        "text":  [f"real review {i}" for i in range(n_per_star * n_stars)],
        "stars": [float((i % n_stars) + 1) for i in range(n_per_star * n_stars)],
    })
    guide_embs  = np.random.randn(n_per_star * n_stars, 32).astype(np.float32)

    eval_df = pd.DataFrame({
        "text":  [f"eval review {i}" for i in range(50)],
        "stars": [float((i % n_stars) + 1) for i in range(50)],
    })

    cfg = make_mock_cfg(guide_size=n_per_star * n_stars, synth_size=n_per_star * n_stars)

    with patch("pandas.read_csv", return_value=synth_df), \
         patch("numpy.load", side_effect=[synth_embs, guide_embs]), \
         patch("pandas.read_json", side_effect=[guide_df, eval_df]):

        ds, train_embs, ret_guide_embs = load_and_sample(cfg, seed=42)

    # Check each star has equal count in guide
    import collections
    guide_labels = ds["guide"]["label"]
    counts = collections.Counter(guide_labels)
    assert len(set(counts.values())) == 1, \
        f"Unequal star distribution in guide: {dict(counts)}"


def test_embedding_shape_matches_dataset():
    """train_embs rows must match ds train split size."""
    import pandas as pd
    from beacon.data.dataset import load_and_sample

    n          = 50
    synth_df   = pd.DataFrame({
        "full_text": [f"r {i}" for i in range(n * 5)],
        "stars":     [float((i % 5) + 1) for i in range(n * 5)],
    })
    synth_embs  = np.random.randn(n * 5, 32).astype(np.float32)
    guide_df    = pd.DataFrame({
        "text":  [f"g {i}" for i in range(n * 5)],
        "stars": [float((i % 5) + 1) for i in range(n * 5)],
    })
    guide_embs  = np.random.randn(n * 5, 32).astype(np.float32)
    eval_df     = pd.DataFrame({
        "text":  [f"e {i}" for i in range(20)],
        "stars": [float((i % 5) + 1) for i in range(20)],
    })

    cfg = make_mock_cfg(guide_size=n * 5, synth_size=n * 5)

    with patch("pandas.read_csv", return_value=synth_df), \
         patch("numpy.load", side_effect=[synth_embs, guide_embs]), \
         patch("pandas.read_json", side_effect=[guide_df, eval_df]):

        ds, train_embs, ret_guide_embs = load_and_sample(cfg, seed=42)

    assert len(ds["train"]) == train_embs.shape[0], \
        f"Dataset size {len(ds['train'])} != embedding rows {train_embs.shape[0]}"