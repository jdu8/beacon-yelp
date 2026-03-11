"""
Extract embeddings from a fine-tuned model for all synthetic and real train data.
Saves new .npy files to DATA_DIR that can be used by the ablation runs.

Usage:
    python scripts/extract_embeddings.py \
        --model_dir outputs/pretrain_deberta_base \
        --batch_size 128 \
        --output_suffix finetuned

This produces:
    data/embeddings_synthetic_500k_finetuned.npy
    data/embeddings_base_70k_finetuned.npy
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def extract(model, dataloader, device):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pool over non-padding tokens
            hidden = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            summed = (hidden * mask_expanded).sum(dim=1)  # [B, hidden_dim]
            counts = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
            embs = summed / counts  # [B, hidden_dim]
            all_embs.append(embs.float().cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "./data"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--output_suffix", default="finetuned", help="Suffix for output .npy files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            dtype = torch.bfloat16
    print(f"Device: {device} | Dtype: {dtype}")

    # Load base encoder (without classification head) from the fine-tuned checkpoint
    print(f"Loading model from {args.model_dir}...")
    model = AutoModel.from_pretrained(
        args.model_dir, torch_dtype=dtype, ignore_mismatched_sizes=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # ── Synthetic 500k ──────────────────────────────────────────────
    print("\nLoading synthetic data...")
    synth_df = pd.read_csv(os.path.join(args.data_dir, "synthetic_500k.csv"))
    texts_col = "full_text" if "full_text" in synth_df.columns else "text"
    synth_texts = synth_df[texts_col].tolist()
    print(f"  {len(synth_texts)} synthetic samples")

    synth_dataset = TextDataset(synth_texts, tokenizer, args.max_len)
    synth_loader = DataLoader(synth_dataset, batch_size=args.batch_size, shuffle=False)

    synth_embs = extract(model, synth_loader, device)
    synth_out = os.path.join(args.data_dir, f"embeddings_synthetic_500k_{args.output_suffix}.npy")
    np.save(synth_out, synth_embs)
    print(f"  Saved: {synth_out} — shape {synth_embs.shape}")

    # ── Real train 70k ──────────────────────────────────────────────
    print("\nLoading real train data...")
    real_df = pd.read_json(os.path.join(args.data_dir, "restaurant_train_70k.json"), lines=True)
    real_texts = real_df["text"].tolist()
    print(f"  {len(real_texts)} real samples")

    real_dataset = TextDataset(real_texts, tokenizer, args.max_len)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

    real_embs = extract(model, real_loader, device)
    real_out = os.path.join(args.data_dir, f"embeddings_base_70k_{args.output_suffix}.npy")
    np.save(real_out, real_embs)
    print(f"  Saved: {real_out} — shape {real_embs.shape}")

    print(f"\nDone! Embedding dim: {synth_embs.shape[1]}")
    print(f"Use these in configs with:")
    print(f"  synth_emb_file: embeddings_synthetic_500k_{args.output_suffix}.npy")
    print(f"  real_emb_file: embeddings_base_70k_{args.output_suffix}.npy")


if __name__ == "__main__":
    main()
