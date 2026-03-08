#!/bin/bash
cd /home/ubuntu/beacon-yelp
source venv/bin/activate
mkdir -p logs

echo "=== Starting all sweeps ==="

python scripts/train.py --config-path=../configs/sweep --config-name=scheme_sweep --multirun >> logs/scheme_sweep.log 2>&1
python scripts/train.py --config-path=../configs/sweep --config-name=k_sweep --multirun >> logs/k_sweep.log 2>&1
python scripts/train.py --config-path=../configs/sweep --config-name=ratio_sweep --multirun >> logs/ratio_sweep.log 2>&1
python scripts/train.py --config-path=../configs/sweep --config-name=guide_sweep --multirun >> logs/guide_sweep.log 2>&1

echo "=== Uploading outputs ==="
gsutil -m rsync -r outputs/ gs://beacon-research-iy2159/outputs/
gsutil -m rsync -r logs/ gs://beacon-research-iy2159/logs/

echo "=== Shutting down ==="
sudo shutdown -h now
