#!/bin/bash
set -e

echo "=== Beacon startup ==="

# Auth GCP
gcloud auth activate-service-account --key-file=/workspace/gcp-key.json

# Pull data from bucket to local NVMe
echo "Pulling data from GCP bucket..."
mkdir -p /data
gsutil -m cp -r gs://${GCS_BUCKET}/data/ /data/

# Pull latest code
echo "Pulling latest code..."
git pull origin main

# Install package
pip install -e .

echo "=== Startup complete ==="