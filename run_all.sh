#!/bin/bash
cd /home/ubuntu/beacon-yelp
source venv/bin/activate
mkdir -p logs

echo "=== BEACON FULL SWEEP ==="
date

# ── OVERLAP SWEEP ─────────────────────────────────────────────────────
echo "--- Overlap sweep seed 42 ---"
python scripts/train.py --multirun \
  data=default,overlap_top10,overlap_top25,overlap_top50,overlap_bot25,overlap_bot10 \
  reweighting=uniform,minmax_soft \
  experiment.seed=42 2>&1 | tee logs/overlap_seed42.log

echo "--- Overlap sweep seed 123 ---"
python scripts/train.py --multirun \
  data=default,overlap_top10,overlap_top25,overlap_top50,overlap_bot25,overlap_bot10 \
  reweighting=uniform,minmax_soft \
  experiment.seed=123 2>&1 | tee logs/overlap_seed123.log

echo "--- Overlap sweep seed 67 ---"
python scripts/train.py --multirun \
  data=default,overlap_top10,overlap_top25,overlap_top50,overlap_bot25,overlap_bot10 \
  reweighting=uniform,minmax_soft \
  experiment.seed=67 2>&1 | tee logs/overlap_seed67.log

# ── K SWEEP at default + bot10 ────────────────────────────────────────
echo "--- K sweep default seed 42 ---"
python scripts/train.py --multirun \
  data=default \
  reweighting=uniform,minmax_soft \
  reweighting.k=10,50,100,all \
  experiment.seed=42 2>&1 | tee logs/k_default_seed42.log

echo "--- K sweep default seed 123 ---"
python scripts/train.py --multirun \
  data=default \
  reweighting=uniform,minmax_soft \
  reweighting.k=10,50,100,all \
  experiment.seed=123 2>&1 | tee logs/k_default_seed123.log

echo "--- K sweep default seed 67 ---"
python scripts/train.py --multirun \
  data=default \
  reweighting=uniform,minmax_soft \
  reweighting.k=10,50,100,all \
  experiment.seed=67 2>&1 | tee logs/k_default_seed67.log

echo "--- K sweep bot10 seed 42 ---"
python scripts/train.py --multirun \
  data=overlap_bot10 \
  reweighting=uniform,minmax_soft \
  reweighting.k=10,50,100,all \
  experiment.seed=42 2>&1 | tee logs/k_bot10_seed42.log

echo "--- K sweep bot10 seed 123 ---"
python scripts/train.py --multirun \
  data=overlap_bot10 \
  reweighting=uniform,minmax_soft \
  reweighting.k=10,50,100,all \
  experiment.seed=123 2>&1 | tee logs/k_bot10_seed123.log

echo "--- K sweep bot10 seed 67 ---"
python scripts/train.py --multirun \
  data=overlap_bot10 \
  reweighting=uniform,minmax_soft \
  reweighting.k=10,50,100,all \
  experiment.seed=67 2>&1 | tee logs/k_bot10_seed67.log

# ── RATIO SWEEP ───────────────────────────────────────────────────────
echo "--- Ratio sweep all seeds ---"
python scripts/train.py --multirun \
  data=default,ratio_5x,ratio_20x,ratio_50x \
  reweighting=uniform,minmax_soft \
  experiment.seed=42,123,67 2>&1 | tee logs/ratio_sweep.log

# ── GUIDE SWEEP ───────────────────────────────────────────────────────
echo "--- Guide sweep all seeds ---"
python scripts/train.py --multirun \
  data.guide_size=1000,3000,5000,10000 \
  reweighting=uniform,minmax_soft \
  experiment.seed=42,123,67 2>&1 | tee logs/guide_sweep.log

echo "=== ALL SWEEPS COMPLETE ==="
date

echo "=== Uploading to GCP ==="
gsutil -m rsync -r outputs/ gs://beacon-research-iy2159/outputs/
gsutil -m rsync -r logs/ gs://beacon-research-iy2159/logs/

echo "=== Done — terminate instance from Lambda Labs dashboard ==="