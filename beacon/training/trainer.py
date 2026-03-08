import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_scheduler
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb

from beacon.data.embeddings import build_topk_matrix
from beacon.training.reweighting import compute_sample_weights, get_weight_stats
from beacon.training.metrics import compute_qwk, compute_mse, compute_per_star_metrics
from beacon.utils.viz import plot_weight_umap, plot_weight_histogram


# ── UMAP subsample size — full 21k is too slow mid-training ──────────────────
UMAP_SUBSAMPLE = 5000


def _get_epoch1_ckpt_name(cfg: DictConfig) -> str:
    model_tag = cfg.model.name.replace("/", "_")
    data_tag  = cfg.data.get("_name_", "default")
    seed      = cfg.experiment.seed
    return f"epoch1_{model_tag}_{data_tag}_seed{seed}.pt"


def _build_dataloader(dataset, batch_size, sampler=None, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
    )


def _collect_guide_losses(
    model, guide_loader, device, dtype
) -> np.ndarray:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in guide_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device).to(dtype)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = outputs.logits.squeeze(-1).to(dtype)
            per_sample     = (preds - labels) ** 2
            losses.extend(per_sample.float().cpu().numpy())
    return np.array(losses)


def _evaluate(model, loader, device, split_name="eval") -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].float().to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = outputs.logits.squeeze(-1)
            all_preds.extend(preds.float().cpu().numpy())
            all_labels.extend(labels.float().cpu().numpy())

    qwk = compute_qwk(all_preds, all_labels)
    mse = compute_mse(all_preds, all_labels)
    print(f"  [{split_name}] QWK: {qwk:.4f} | MSE: {mse:.4f}")
    return {"qwk": qwk, "mse": mse}


def train(
    cfg:           DictConfig,
    model,
    tokenizer,
    ds_tokenized,
    train_embs:    np.ndarray,
    guide_embs:    np.ndarray,
    device:        torch.device,
    dtype:         torch.dtype,
) -> dict:
    """
    Main training loop. Returns full history dict.
    """

    run_name   = cfg.experiment.name
    seed       = cfg.experiment.seed
    output_dir = os.path.join(cfg.paths.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── W&B init ─────────────────────────────────────────────────────────────
    wandb.init(
        project = cfg.wandb.project,
        entity  = cfg.wandb.get("entity", None),
        name    = run_name,
        config  = OmegaConf.to_container(cfg, resolve=True),
        mode    = os.environ.get("WANDB_MODE", cfg.wandb.get("mode", "online")),
    )

    # ── Similarity matrix — computed once ────────────────────────────────────
    print("Building topk similarity matrix...")
    topk_matrix = build_topk_matrix(train_embs, guide_embs, cfg)

    # ── Dataloaders ───────────────────────────────────────────────────────────
    sample_weights = np.ones(len(ds_tokenized["train"]))
    train_sampler  = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = _build_dataloader(
        ds_tokenized["train"], cfg.training.batch_size, sampler=train_sampler
    )
    guide_loader = _build_dataloader(
        ds_tokenized["guide"], cfg.training.batch_size
    )
    test_quick_loader = _build_dataloader(
        ds_tokenized["test_quick"], cfg.training.batch_size
    )
    test_full_loader = _build_dataloader(
        ds_tokenized["test"], cfg.training.batch_size
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = cfg.training.num_epochs * len(train_loader)
    scheduler   = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(cfg.training.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    loss_fn = nn.MSELoss()

    # ── Epoch 1 checkpoint ────────────────────────────────────────────────────
    ckpt_name = _get_epoch1_ckpt_name(cfg)
    ckpt_path = os.path.join(cfg.paths.output_dir, "epoch1_checkpoints", ckpt_name)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    epoch1_exists = os.path.exists(ckpt_path)

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        "run_name":       run_name,
        "train_loss":     [],
        "guide_qwk":      [],
        "guide_mse":      [],
        "test_quick_qwk": [],
        "test_quick_mse": [],
        "weight_stats":   [],
        "per_star":       [],
    }

    best_qwk          = -1
    best_epoch        = -1
    epochs_no_improve = 0
    start_epoch       = 0

    # ── Load epoch 1 checkpoint if exists ────────────────────────────────────
    if epoch1_exists:
        print(f"Loading epoch 1 checkpoint: {ckpt_name}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        start_epoch = 1
        print("Skipping epoch 1 — loaded from checkpoint.")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.num_epochs):

        # ── Reweighting ───────────────────────────────────────────────────────
        if epoch > 0:
            print(f"\n  Computing guide losses...")
            guide_losses = _collect_guide_losses(model, guide_loader, device, dtype)

            # Per-star breakdown
            guide_label_arr = np.array(ds_tokenized["guide"]["label"])
            per_star        = compute_per_star_metrics(guide_losses, guide_label_arr)
            per_star_str = {k: f"{v['mean_loss']:.3f}" for k, v in per_star.items()}
            print(f"  Per-star guide loss: {per_star_str}")

            # Compute weights
            scheme = cfg.reweighting.scheme
	    	kwargs = OmegaConf.to_container(cfg.reweighting, resolve=True)
	    	kwargs.pop('scheme')  # already passed as positional arg
	    	sample_weights = compute_sample_weights(topk_matrix, guide_losses, scheme, **kwargs)

            w_stats = get_weight_stats(sample_weights)
            print(f"  Weights — std:{w_stats['w_std']:.4f} "
                  f"max:{w_stats['w_max']:.4f} ratio:{w_stats['w_ratio']:.2f}x")

            history["weight_stats"].append({"epoch": epoch, **w_stats})
            history["per_star"].append({"epoch": epoch, **per_star})

            # W&B logging
            wandb.log({
                "weights/std":   w_stats["w_std"],
                "weights/max":   w_stats["w_max"],
                "weights/ratio": w_stats["w_ratio"],
                **{f"per_star/{k}": v["mean_loss"] for k, v in per_star.items()},
            }, step=epoch)

            # Visualizations
            plot_weight_histogram(sample_weights, epoch)

            # UMAP — subsample for speed
            umap_idx = np.random.choice(
                len(train_embs), min(UMAP_SUBSAMPLE, len(train_embs)), replace=False
            )
            plot_weight_umap(
                train_embs[umap_idx],
                sample_weights[umap_idx],
                guide_embs,
                guide_losses,
                epoch,
                run_name,
            )

            # Rebuild sampler
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = _build_dataloader(
                ds_tokenized["train"], cfg.training.batch_size, sampler=train_sampler
            )

        # ── Train one epoch ───────────────────────────────────────────────────
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}", leave=False)

        for batch in loop:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device).to(dtype)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.squeeze(-1).to(dtype)
            loss    = loss_fn(preds, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_clip
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | avg_loss: {avg_loss:.4f}")

        # ── Save epoch 1 checkpoint ───────────────────────────────────────────
        if epoch == 0 and not epoch1_exists:
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Epoch 1 checkpoint saved: {ckpt_name}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        guide_metrics      = _evaluate(model, guide_loader,      device, "guide     ")
        test_quick_metrics = _evaluate(model, test_quick_loader, device, "test_quick")

        history["train_loss"].append(avg_loss)
        history["guide_qwk"].append(guide_metrics["qwk"])
        history["guide_mse"].append(guide_metrics["mse"])
        history["test_quick_qwk"].append(test_quick_metrics["qwk"])
        history["test_quick_mse"].append(test_quick_metrics["mse"])

        wandb.log({
            "train/loss":          avg_loss,
            "eval/guide_qwk":      guide_metrics["qwk"],
            "eval/guide_mse":      guide_metrics["mse"],
            "eval/test_quick_qwk": test_quick_metrics["qwk"],
            "eval/test_quick_mse": test_quick_metrics["mse"],
        }, step=epoch)

        # ── Early stopping + best model ───────────────────────────────────────
        if test_quick_metrics["qwk"] > best_qwk:
            best_qwk          = test_quick_metrics["qwk"]
            best_epoch        = epoch + 1
            epochs_no_improve = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  ✓ Best saved — test_quick QWK: {best_qwk:.4f} (epoch {best_epoch})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{cfg.training.patience})")
            if epochs_no_improve >= cfg.training.patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    # ── Final evaluation on full 30k test set ────────────────────────────────
    print("\nLoading best checkpoint for final evaluation...")
    from transformers import AutoModelForSequenceClassification
    best_model = AutoModelForSequenceClassification.from_pretrained(
        output_dir, torch_dtype=dtype
    ).to(device)

    test_full_metrics = _evaluate(best_model, test_full_loader, device, "test_full ")
    history["test_full_qwk"] = test_full_metrics["qwk"]
    history["test_full_mse"] = test_full_metrics["mse"]
    history["best_epoch"]    = best_epoch

    wandb.log({
        "eval/test_full_qwk": test_full_metrics["qwk"],
        "eval/test_full_mse": test_full_metrics["mse"],
        "best_epoch":         best_epoch,
    })

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {run_name} COMPLETE")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  test_quick QWK:   {best_qwk:.4f}")
    print(f"  test_full  QWK:   {test_full_metrics['qwk']:.4f}")
    print(f"{'='*55}")

    wandb.finish()
    return history
