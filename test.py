import numpy as np
import pandas as pd
import os
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel


def sample_by_percentile(synth_df, synth_embs, guide_embs, guide_labels, n_per_star, percentile_low, percentile_high, seed=42):
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


def frechet_distance(embs_real, embs_synth):
    mu_r, mu_s = embs_real.mean(0), embs_synth.mean(0)
    sigma_r = np.cov(embs_real.T)
    sigma_s = np.cov(embs_synth.T)
    diff = mu_r - mu_s
    covmean = sqrtm(sigma_r @ sigma_s)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_s - 2 * covmean))


def mmd(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


if __name__ == "__main__":
    DATA_DIR = os.getenv("DATA_DIR", "./data")

    print("Loading embeddings...")
    synth_embs  = np.load(f"{DATA_DIR}/embeddings_synthetic_500k.npy").astype(np.float32)
    guide_embs  = np.load(f"{DATA_DIR}/embeddings_base_70k.npy").astype(np.float32)
    synth_df    = pd.read_csv(f"{DATA_DIR}/synthetic_500k.csv")
    synth_df    = synth_df.rename(columns={'full_text': 'text', 'stars': 'label'})
    synth_df['label'] = synth_df['label'].astype(float)
    guide_df    = pd.read_json(f"{DATA_DIR}/restaurant_train_70k.json", lines=True)
    guide_df    = guide_df.rename(columns={'stars': 'label'})

    from beacon.data.dataset import stratified_sample
    guide_sampled      = stratified_sample(guide_df, 600, seed=42)
    guide_embs_sampled = guide_embs[guide_sampled.index.values]
    guide_labels       = guide_sampled['label'].values.astype(float)

    n_per_star = 6000

    bands = [
        ("top10",    90, 100),
        ("top25",    75, 100),
        ("top50",    50, 100),
        ("top75",    25, 100),
        ("top100",    0, 100),
        ("bot25",     0,  25),
        ("bot10",     0,  10),
    ]

    print(f"\n{'Band':<12} {'Frechet':>10} {'MMD':>12} {'MeanSim':>10}")
    print("-" * 48)

    for name, plo, phi in bands:
        _, s_embs = sample_by_percentile(
            synth_df, synth_embs,
            guide_embs_sampled, guide_labels,
            n_per_star, plo, phi, seed=42
        )

        idx_s = np.random.choice(len(s_embs), min(2000, len(s_embs)), replace=False)
        idx_g = np.random.choice(len(guide_embs_sampled), min(2000, len(guide_embs_sampled)), replace=False)
        s_sub = s_embs[idx_s]
        g_sub = guide_embs_sampled[idx_g]

        fd  = frechet_distance(g_sub, s_sub)
        m   = mmd(g_sub, s_sub)

        s_norm   = s_sub / np.linalg.norm(s_sub, axis=1, keepdims=True)
        g_norm   = g_sub / np.linalg.norm(g_sub, axis=1, keepdims=True)
        mean_sim = float((s_norm @ g_norm.T).mean())

        print(f"{name:<12} {fd:>10.3f} {m:>12.6f} {mean_sim:>10.6f}")