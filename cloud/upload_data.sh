#!/bin/bash
set -e

BUCKET=$1

if [ -z "$BUCKET" ]; then
    echo "Usage: ./cloud/upload_data.sh <bucket_name>"
    exit 1
fi

DATA_DIR=${DATA_DIR:-./data}

echo "=== Uploading data to gs://$BUCKET ==="

# # Real data
# echo "Uploading real data..."
# gsutil -m cp $DATA_DIR/restaurant_train_70k.json gs://$BUCKET/data/real/
# gsutil -m cp $DATA_DIR/restaurant_test_30k.json  gs://$BUCKET/data/real/
# gsutil -m cp $DATA_DIR/embeddings_base_70k.npy   gs://$BUCKET/data/real/

# # Synthetic data
# echo "Uploading synthetic data..."
# gsutil -m cp $DATA_DIR/synthetic_500k.csv              gs://$BUCKET/data/synthetic/
gsutil -m cp $DATA_DIR/embeddings_synthetic_500k.npy   gs://$BUCKET/data/synthetic/

echo "=== Upload complete ==="
echo "Run next: python scripts/validate_data.py"