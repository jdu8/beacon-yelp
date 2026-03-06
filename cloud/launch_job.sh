#!/bin/bash
# Usage: ./cloud/launch_job.sh scheme_sweep

SWEEP=${1:-scheme_sweep}
echo "Launching sweep: $SWEEP"
python scripts/sweep.py $SWEEP