import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from omegaconf import DictConfig


def load_model(cfg: DictConfig, device: torch.device, dtype: torch.dtype):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        dtype=dtype,
    )
    model = model.to(device)
    return model


def load_tokenizer(cfg: DictConfig):
    return AutoTokenizer.from_pretrained(cfg.model.name)


def get_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cap    = torch.cuda.get_device_capability()
        dtype  = torch.bfloat16 if cap[0] >= 8 else torch.float32
    else:
        device = torch.device("cpu")
        dtype  = torch.float32

    print(f"Device: {device} | "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'} | "
          f"Dtype: {dtype}")

    return device, dtype


def tokenize_dataset(ds, tokenizer, cfg: DictConfig):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.model.max_len,
        )

    # Add sample indices to train split for loss-weighted mode
    if "train" in ds:
        ds["train"] = ds["train"].add_column("sample_idx", list(range(len(ds["train"]))))

    ds_tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    torch_cols = ["input_ids", "attention_mask", "label"]
    if "sample_idx" in ds_tokenized["train"].column_names:
        torch_cols_train = torch_cols + ["sample_idx"]
    else:
        torch_cols_train = torch_cols

    for split in ds_tokenized:
        cols = torch_cols_train if split == "train" else torch_cols
        ds_tokenized[split].set_format("torch", columns=cols)

    return ds_tokenized
