import hydra
from omegaconf import DictConfig

from beacon.utils.seed import set_seed
from beacon.data.dataset import load_and_sample, sample_test_subset, print_split_summary
from beacon.models.scorer import load_model, load_tokenizer, get_device_and_dtype, tokenize_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    from beacon.training.trainer import train

    # ── Seed ─────────────────────────────────────────────────────────────────
    set_seed(cfg.experiment.seed)

    # ── Device + dtype ────────────────────────────────────────────────────────
    device, dtype = get_device_and_dtype()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    ds, train_embs, guide_embs = load_and_sample(cfg, seed=cfg.experiment.seed)
    print_split_summary(ds)

    # ── Quick test subset ─────────────────────────────────────────────────────
    test_quick = sample_test_subset(ds, n=5000, seed=cfg.experiment.seed)
    ds["test_quick"] = test_quick

    # ── Tokenize ──────────────────────────────────────────────────────────────
    print("\nTokenizing...")
    tokenizer    = load_tokenizer(cfg)
    ds_tokenized = tokenize_dataset(ds, tokenizer, cfg)
    print(ds_tokenized)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = load_model(cfg, device, dtype)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    history = train(
        cfg        = cfg,
        model      = model,
        tokenizer  = tokenizer,
        ds_tokenized = ds_tokenized,
        train_embs = train_embs,
        guide_embs = guide_embs,
        device     = device,
        dtype      = dtype,
    )


if __name__ == "__main__":
    main()