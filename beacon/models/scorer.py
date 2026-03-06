import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from omegaconf import DictConfig


def load_model(cfg: DictConfig, device: torch.device, dtype: torch.dtype):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )
    model = model.to(device)
    return model


def load_tokenizer(cfg: DictConfig):
    return AutoTokenizer.from_pretrained(cfg.model.name)


def get_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cap    = torch.cuda.get_device_capability()
        dtype  = torch.bfloat16 if cap[0] >= 8 else torch.float16
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

    ds_tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds_tokenized