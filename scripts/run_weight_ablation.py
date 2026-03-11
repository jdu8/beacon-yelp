"""
Weight ablation: baseline vs sampler-weighted vs loss-weighted (3 ranges).
5 runs total, all using data=test_50k (50k synth, 5k guide).

Usage:
    python scripts/run_weight_ablation.py
"""

import subprocess
import sys

RUNS = [
    {
        "name": "baseline_uniform",
        "overrides": [
            "data=test_50k",
            "reweighting=uniform",
            "experiment.name=ablation_baseline",
            "experiment.seed=42",
        ],
    },
    {
        "name": "sampler_minmax_05_20",
        "overrides": [
            "data=test_50k",
            "reweighting=minmax_soft",
            "experiment.name=ablation_sampler_05_20",
            "experiment.seed=42",
        ],
    },
    {
        "name": "loss_minmax_05_20",
        "overrides": [
            "data=test_50k",
            "reweighting=minmax_loss_05_20",
            "experiment.name=ablation_loss_05_20",
            "experiment.seed=42",
        ],
    },
    {
        "name": "loss_minmax_00_20",
        "overrides": [
            "data=test_50k",
            "reweighting=minmax_loss_00_20",
            "experiment.name=ablation_loss_00_20",
            "experiment.seed=42",
        ],
    },
    {
        "name": "loss_minmax_neg05_20",
        "overrides": [
            "data=test_50k",
            "reweighting=minmax_loss_neg05_20",
            "experiment.name=ablation_loss_neg05_20",
            "experiment.seed=42",
        ],
    },
]


def main():
    results = {}
    for i, run in enumerate(RUNS):
        print(f"\n{'='*60}")
        print(f"  RUN {i+1}/{len(RUNS)}: {run['name']}")
        print(f"{'='*60}\n")

        cmd = [sys.executable, "scripts/train.py"] + run["overrides"]
        proc = subprocess.run(cmd, cwd=".")
        if proc.returncode != 0:
            print(f"  FAILED: {run['name']} (exit code {proc.returncode})")
            results[run["name"]] = "FAILED"
        else:
            results[run["name"]] = "OK"

    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name:<30} {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
