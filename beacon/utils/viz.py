import numpy as np
import matplotlib.pyplot as plt
import umap
import wandb


def plot_weight_umap(
    train_embs:     np.ndarray,
    sample_weights: np.ndarray,
    guide_embs:     np.ndarray,
    guide_losses:   np.ndarray,
    epoch:          int,
    run_name:       str,
) -> None:
    """
    Project train and guide embeddings to 2D via UMAP.
    Color synthetic samples by sampling weight.
    Size guide points by loss magnitude.
    Log directly to W&B — nothing saved locally.
    """

    # ── Fit UMAP on combined embeddings so both share same projection ──
    combined    = np.vstack([train_embs, guide_embs])
    reducer     = umap.UMAP(n_components=2, random_state=42, verbose=False)
    projected   = reducer.fit_transform(combined)

    train_proj  = projected[:len(train_embs)]
    guide_proj  = projected[len(train_embs):]

    # ── Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # Synthetic samples — colored by weight
    sc = ax.scatter(
        train_proj[:, 0],
        train_proj[:, 1],
        c=sample_weights,
        cmap='RdYlGn',
        s=3,
        alpha=0.6,
        vmin=sample_weights.min(),
        vmax=sample_weights.max(),
    )

    # Guide points — sized by loss, white outline
    guide_sizes = 20 + 200 * (guide_losses / guide_losses.max())
    ax.scatter(
        guide_proj[:, 0],
        guide_proj[:, 1],
        c=guide_losses,
        cmap='hot',
        s=guide_sizes,
        alpha=0.9,
        edgecolors='white',
        linewidths=0.4,
        marker='D',
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Sample Weight', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_title(
        f'{run_name} — Epoch {epoch} | Weight Distribution in Embedding Space',
        color='white', fontsize=11, pad=12
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # ── Log to W&B ────────────────────────────────────────────────────
    wandb.log({
        f"umap/epoch_{epoch:02d}": wandb.Image(fig),
    }, step=epoch)

    plt.close(fig)


def plot_weight_histogram(
    sample_weights: np.ndarray,
    epoch:          int,
) -> None:
    """
    Log weight distribution histogram to W&B each epoch.
    Quick sanity check that weights aren't collapsing.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    ax.hist(sample_weights, bins=50, color='#e05555', edgecolor='none', alpha=0.85)
    ax.axvline(sample_weights.mean(), color='white', linewidth=1, linestyle='--', alpha=0.7)

    ax.set_xlabel('Sample Weight', color='white', fontsize=9)
    ax.set_ylabel('Count', color='white', fontsize=9)
    ax.set_title(f'Weight Distribution — Epoch {epoch}', color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')

    plt.tight_layout()

    wandb.log({
        f"weights/histogram_epoch_{epoch:02d}": wandb.Image(fig),
    }, step=epoch)

    plt.close(fig)