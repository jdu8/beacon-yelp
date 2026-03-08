import os
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig


def stratified_sample(df: pd.DataFrame, n_per_star: int, seed: int) -> pd.DataFrame:
    return pd.concat([
        group.sample(n=min(n_per_star, len(group)), random_state=seed)
        for _, group in df.groupby('label')
    ]).reset_index(drop=True)


def sample_by_overlap(
    synth_df:     pd.DataFrame,
    synth_embs:   np.ndarray,
    guide_embs:   np.ndarray,
    guide_labels: np.ndarray,
    n_per_star:   int,
    percentile_low:  float,
    percentile_high: float,
    seed:         int = 42,
) -> tuple:
    """
    Sample synthetic data from a specific similarity percentile band.
    percentile_low=0, percentile_high=100 → random (full distribution)
    percentile_low=90, percentile_high=100 → top 10% most similar to real
    percentile_low=0, percentile_high=10  → bottom 10% least similar to real
    """
    selected = []
    selected_emb_idx = []

    for star in [1.0, 2.0, 3.0, 4.0, 5.0]:
        mask_s = (synth_df['label'] == star).values
        s_df   = synth_df[mask_s].reset_index(drop=True)
        s_embs = synth_embs[mask_s]

        mask_g     = guide_labels == star
        g_embs     = guide_embs[mask_g]
        g_centroid = g_embs.mean(axis=0, keepdims=True)

        s_norm = s_embs / np.linalg.norm(s_embs, axis=1, keepdims=True)
        g_norm = g_centroid / np.linalg.norm(g_centroid)
        sims   = (s_norm @ g_norm.T).squeeze()

        lo = np.percentile(sims, percentile_low)
        hi = np.percentile(sims, percentile_high)
        band_mask = (sims >= lo) & (sims <= hi)
        band_idx  = np.where(band_mask)[0]

        n = min(n_per_star, len(band_idx))
        rng = np.random.RandomState(seed)
        idx = rng.choice(band_idx, n, replace=False)

        selected.append(s_df.iloc[idx])
        orig_indices = np.where(mask_s)[0][idx]
        selected_emb_idx.extend(orig_indices.tolist())

    result_df   = pd.concat(selected).reset_index(drop=True)
    result_embs = synth_embs[selected_emb_idx]
    return result_df, result_embs


def load_and_sample(cfg: DictConfig, seed: int) -> tuple:
    data_dir = cfg.paths.data_dir

    # ── Synthetic (ts) ────────────────────────────────────────────────
    synth_df   = pd.read_csv(os.path.join(data_dir, cfg.data.synth_file))
    synth_df   = synth_df.rename(columns={'full_text': 'text', 'stars': 'label'})
    synth_df['label'] = synth_df['label'].astype(float)
    synth_embs = np.load(os.path.join(data_dir, cfg.data.synth_emb_file))
    assert len(synth_embs) == len(synth_df), \
        f"Synthetic embedding mismatch: {len(synth_embs)} vs {len(synth_df)}"

    # ── Real train → tg_guide ─────────────────────────────────────────
    guide_full_df = pd.read_json(
        os.path.join(data_dir, cfg.data.real_train_file), lines=True
    )
    guide_full_df = guide_full_df.rename(columns={'stars': 'label'})
    guide_full_df['label'] = guide_full_df['label'].astype(float)
    guide_full_embs = np.load(os.path.join(data_dir, cfg.data.real_emb_file))
    assert len(guide_full_embs) == len(guide_full_df), \
        f"Guide embedding mismatch: {len(guide_full_embs)} vs {len(guide_full_df)}"
    guide_full_df['emb_idx'] = np.arange(len(guide_full_df))

    guide_df   = stratified_sample(guide_full_df, cfg.data.guide_size // 5, seed)
    guide_embs = guide_full_embs[guide_df['emb_idx'].values]
    guide_df   = guide_df.drop(columns=['emb_idx'])

    # ── Sample synthetic with overlap control ─────────────────────────
    overlap_low  = float(cfg.data.get('overlap_pct_low',  0))
    overlap_high = float(cfg.data.get('overlap_pct_high', 100))
    n_per_star   = cfg.data.synth_size // 5

    if overlap_low == 0 and overlap_high == 100:
        # random — use fast stratified sample
        synth_df['emb_idx'] = np.arange(len(synth_df))
        synth_sampled      = stratified_sample(synth_df, n_per_star, seed)
        synth_embs_sampled = synth_embs[synth_sampled['emb_idx'].values]
        synth_sampled      = synth_sampled.drop(columns=['emb_idx'])
    else:
        synth_sampled, synth_embs_sampled = sample_by_overlap(
            synth_df, synth_embs,
            guide_embs, guide_df['label'].values.astype(float),
            n_per_star, overlap_low, overlap_high, seed
        )

    # ── Real test → tg_eval (locked) ──────────────────────────────────
    eval_df = pd.read_json(
        os.path.join(data_dir, cfg.data.real_test_file), lines=True
    )
    eval_df = eval_df.rename(columns={'stars': 'label'})
    eval_df['label'] = eval_df['label'].astype(float)

    # ── DatasetDict ───────────────────────────────────────────────────
    ds = DatasetDict({
        'train': Dataset.from_pandas(synth_sampled[['text', 'label']].copy()),
        'guide': Dataset.from_pandas(guide_df[['text', 'label']].copy()),
        'test':  Dataset.from_pandas(
            eval_df[['text', 'label']].reset_index(drop=True).copy()
        ),
    })

    return ds, synth_embs_sampled, guide_embs


def sample_test_subset(ds: DatasetDict, n: int = 5000, seed: int = 42) -> Dataset:
    test_df = ds['test'].to_pandas()
    sampled = stratified_sample(test_df, n // 5, seed)
    return Dataset.from_pandas(sampled.copy())


def print_split_summary(ds: DatasetDict) -> None:
    print(f"\n{'Split':<10} {'Size':<10}")
    print("-" * 20)
    for split in ds:
        print(f"{split:<10} {len(ds[split]):<10}")