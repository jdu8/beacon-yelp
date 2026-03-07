#!/bin/bash
set -e

PROJECT_ID=$1
BUCKET=$2

if [ -z "$PROJECT_ID" ] || [ -z "$BUCKET" ]; then
    echo "Usage: ./cloud/setup_bucket.sh <project_id> <bucket_name>"
    exit 1
fi

echo "=== Setting up GCP bucket ==="

gcloud config set project $PROJECT_ID

# Create bucket
gsutil mb -l us-central1 gs://$BUCKET

# GCP buckets don't need explicit folder creation
# Folders are created implicitly when files are uploaded
# Just create placeholder files to establish structure
echo "real" | gsutil cp - gs://$BUCKET/data/real/.keep
echo "synthetic" | gsutil cp - gs://$BUCKET/data/synthetic/.keep
echo "outputs" | gsutil cp - gs://$BUCKET/outputs/.keep
echo "checkpoints" | gsutil cp - gs://$BUCKET/epoch1_checkpoints/.keep

echo "=== Bucket setup complete ==="
gsutil ls gs://$BUCKET/