#!/bin/bash
# Runs on Lambda Labs VM at job start
# Pulls data from GCP bucket, trains, pushes results back

set -e

# Auth GCP from service account key
gcloud auth activate-service-account --key-file=/workspace/gcp-key.json

# Pull data from bucket to local NVMe
mkdir -p /data
gsutil -m cp -r gs://${GCS_BUCKET}/data/ /data/

# Pull latest code
git pull origin main

# Run training — args passed from launch script
python scripts/train.py "$@"

# Push results back to bucket
gsutil -m cp -r /outputs/ gs://${GCS_BUCKET}/outputs/