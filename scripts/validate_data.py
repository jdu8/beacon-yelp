import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")

REQUIRED_FILES = [
    "synthetic_500k.csv",
    "embeddings_synthetic_500k.npy",
    "restaurant_train_70k.json",
    "restaurant_test_30k.json",
    "embeddings_base_70k.npy",
]

STAR_RATINGS  = [1.0, 2.0, 3.0, 4.0, 5.0]
MAX_GUIDE     = 10000   # largest guide config
MAX_SYNTH     = 150000  # largest ratio config (50x with guide=3k)
SAMPLES_PER_STAR_GUIDE = MAX_GUIDE  // 5
SAMPLES_PER_STAR_SYNTH = MAX_SYNTH // 5

PASS = "  ✓"
FAIL = "  ✗"


def check(condition: bool, msg: str) -> bool:
    if condition:
        print(f"{PASS} {msg}")
    else:
        print(f"{FAIL} {msg}")
    return condition


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def main() -> None:
    errors = 0

    # ── 1. Files exist ────────────────────────────────────────────────
    section("File existence")
    for fname in REQUIRED_FILES:
        path = os.path.join(DATA_DIR, fname)
        ok   = check(os.path.exists(path), fname)
        if not ok:
            errors += 1

    if errors > 0:
        print(f"\n{errors} file(s) missing. Fix before continuing.")
        sys.exit(1)

    # ── 2. Load files ─────────────────────────────────────────────────
    section("Loading files")
    synth_df    = pd.read_csv(os.path.join(DATA_DIR, "synthetic_500k.csv"))
    synth_df    = synth_df.rename(columns={"full_text": "text", "stars": "label"})
    synth_df["label"] = synth_df["label"].astype(float)
    print(f"{PASS} synthetic_500k.csv loaded — {len(synth_df):,} rows")

    synth_embs  = np.load(os.path.join(DATA_DIR, "embeddings_synthetic_500k.npy"))
    print(f"{PASS} embeddings_synthetic_500k.npy loaded — {synth_embs.shape}")

    guide_df    = pd.read_json(os.path.join(DATA_DIR, "restaurant_train_70k.json"), lines=True)
    guide_df    = guide_df.rename(columns={"stars": "label"})
    guide_df["label"] = guide_df["label"].astype(float)
    print(f"{PASS} restaurant_train_70k.json loaded — {len(guide_df):,} rows")

    guide_embs  = np.load(os.path.join(DATA_DIR, "embeddings_base_70k.npy"))
    print(f"{PASS} embeddings_base_70k.npy loaded — {guide_embs.shape}")

    eval_df     = pd.read_json(os.path.join(DATA_DIR, "restaurant_test_30k.json"), lines=True)
    eval_df     = eval_df.rename(columns={"stars": "label"})
    print(f"{PASS} restaurant_test_30k.json loaded — {len(eval_df):,} rows")

    # ── 3. Embedding alignment ────────────────────────────────────────
    section("Embedding alignment")
    ok = check(
        len(synth_embs) == len(synth_df),
        f"Synthetic embeddings aligned: {len(synth_embs):,} == {len(synth_df):,}"
    )
    if not ok: errors += 1

    ok = check(
        len(guide_embs) == len(guide_df),
        f"Guide embeddings aligned: {len(guide_embs):,} == {len(guide_df):,}"
    )
    if not ok: errors += 1

    ok = check(
        synth_embs.shape[1] == guide_embs.shape[1],
        f"Embedding dims match: {synth_embs.shape[1]} == {guide_embs.shape[1]}"
    )
    if not ok: errors += 1

    # ── 4. NaN / inf check ────────────────────────────────────────────
    section("Embedding integrity")
    ok = check(
        not np.isnan(synth_embs).any(),
        "No NaN in synthetic embeddings"
    )
    if not ok: errors += 1

    ok = check(
        not np.isinf(synth_embs).any(),
        "No inf in synthetic embeddings"
    )
    if not ok: errors += 1

    ok = check(
        not np.isnan(guide_embs).any(),
        "No NaN in guide embeddings"
    )
    if not ok: errors += 1

    ok = check(
        not np.isinf(guide_embs).any(),
        "No inf in guide embeddings"
    )
    if not ok: errors += 1

    # Check roughly unit normalized
    synth_norms = np.linalg.norm(synth_embs[:1000], axis=1)
    guide_norms = np.linalg.norm(guide_embs[:1000], axis=1)
    ok = check(
        np.allclose(synth_norms, 1.0, atol=0.1),
        f"Synthetic embeddings approx unit norm "
        f"(mean={synth_norms.mean():.3f} std={synth_norms.std():.3f})"
    )
    if not ok: errors += 1

    ok = check(
        np.allclose(guide_norms, 1.0, atol=0.1),
        f"Guide embeddings approx unit norm "
        f"(mean={guide_norms.mean():.3f} std={guide_norms.std():.3f})"
    )
    if not ok: errors += 1

    # ── 5. Label distribution ─────────────────────────────────────────
    section("Label distribution")
    print("\n  Synthetic:")
    synth_counts = synth_df["label"].value_counts().sort_index()
    for star, count in synth_counts.items():
        bar = "█" * (count // 5000)
        print(f"    {int(star)}★  {count:>7,}  {bar}")

    print("\n  Real train (guide pool):")
    guide_counts = guide_df["label"].value_counts().sort_index()
    for star, count in guide_counts.items():
        bar = "█" * (count // 1000)
        print(f"    {int(star)}★  {count:>7,}  {bar}")

    print("\n  Real test (eval):")
    eval_counts = eval_df["label"].value_counts().sort_index()
    for star, count in eval_counts.items():
        bar = "█" * (count // 500)
        print(f"    {int(star)}★  {count:>7,}  {bar}")

    # All stars present
    for star in STAR_RATINGS:
        ok = check(star in synth_counts.index, f"{int(star)}★ present in synthetic")
        if not ok: errors += 1
        ok = check(star in guide_counts.index, f"{int(star)}★ present in real train")
        if not ok: errors += 1

    # ── 6. Stratified sampling feasibility ───────────────────────────
    section("Stratified sampling feasibility")
    print(f"  Checking largest configs:")
    print(f"  Max guide per star: {SAMPLES_PER_STAR_GUIDE:,} "
          f"(guide_size={MAX_GUIDE:,})")
    print(f"  Max synth per star: {SAMPLES_PER_STAR_SYNTH:,} "
          f"(synth_size={MAX_SYNTH:,})")

    for star in STAR_RATINGS:
        n_guide = int((guide_df["label"] == star).sum())
        ok = check(
            n_guide >= SAMPLES_PER_STAR_GUIDE,
            f"  {int(star)}★ guide pool sufficient: "
            f"{n_guide:,} >= {SAMPLES_PER_STAR_GUIDE:,}"
        )
        if not ok: errors += 1

    for star in STAR_RATINGS:
        n_synth = int((synth_df["label"] == star).sum())
        ok = check(
            n_synth >= SAMPLES_PER_STAR_SYNTH,
            f"  {int(star)}★ synthetic pool sufficient: "
            f"{n_synth:,} >= {SAMPLES_PER_STAR_SYNTH:,}"
        )
        if not ok: errors += 1

    # ── 7. Quick cosine similarity sanity ────────────────────────────
    section("Cosine similarity sanity check")
    sample_train = synth_embs[:500]
    sample_guide = guide_embs[:500]
    train_norm   = sample_train / np.linalg.norm(sample_train, axis=1, keepdims=True)
    guide_norm   = sample_guide / np.linalg.norm(sample_guide, axis=1, keepdims=True)
    sim_sample   = train_norm @ guide_norm.T

    check(True, f"Sim range: [{sim_sample.min():.3f}, {sim_sample.max():.3f}]")
    check(True, f"Sim mean:  {sim_sample.mean():.3f} | std: {sim_sample.std():.3f}")

    ok = check(
        sim_sample.max() <= 1.01,
        "Cosine similarities bounded correctly"
    )
    if not ok: errors += 1

    # ── Final result ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    if errors == 0:
        print("  ALL CHECKS PASSED — data is ready for training")
    else:
        print(f"  {errors} CHECK(S) FAILED — fix before launching sweep")
    print(f"{'='*55}\n")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()