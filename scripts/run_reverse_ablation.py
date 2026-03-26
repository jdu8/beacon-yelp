"""
Reverse-direction reweighting ablation.
4 runs: baseline, forward minmax, reverse uniform, reverse inverse.
Sequential with live output — use large batch_size to saturate A100.

Usage (Colab cell):
    !python scripts/run_reverse_ablation.py training.batch_size=256
"""

import subprocess
import sys

RUNS = [
    {
        "name": "baseline_uniform",
        "overrides": [
            "reweighting=uniform",
            "experiment.name=reverse_abl_baseline",
            "experiment.seed=42",
        ],
    },
    {
        "name": "forward_minmax",
        "overrides": [
            "reweighting=minmax_soft",
            "experiment.name=reverse_abl_forward_minmax",
            "experiment.seed=42",
        ],
    },
    {
        "name": "reverse_uniform",
        "overrides": [
            "reweighting=reverse_uniform",
            "experiment.name=reverse_abl_reverse_uniform",
            "experiment.seed=42",
        ],
    },
    {
        "name": "reverse_inverse",
        "overrides": [
            "reweighting=reverse_inverse",
            "experiment.name=reverse_abl_reverse_inverse",
            "experiment.seed=42",
        ],
    },
]


def main():
    extra = sys.argv[1:]
    # drop any experiment.name from extra — each run has its own
    extra = [o for o in extra if not o.startswith("experiment.name=")]

    print(f"\n{'='*60}")
    print(f"  REVERSE REWEIGHTING ABLATION — {len(RUNS)} runs, sequential")
    if extra:
        print(f"  Extra overrides: {extra}")
    print(f"{'='*60}\n")

    results = {}
    for i, run in enumerate(RUNS):
        print(f"\n{'='*60}")
        print(f"  RUN {i+1}/{len(RUNS)}: {run['name']}")
        print(f"{'='*60}\n")

        cmd = [sys.executable, "scripts/train.py"] + run["overrides"] + extra
        proc = subprocess.run(cmd)

        status = "OK" if proc.returncode == 0 else "FAILED"
        results[run["name"]] = status
        if proc.returncode != 0:
            print(f"\n  !! {run['name']} FAILED (exit {proc.returncode})")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for run in RUNS:
        status = results.get(run["name"], "UNKNOWN")
        print(f"  {run['name']:<25} {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
