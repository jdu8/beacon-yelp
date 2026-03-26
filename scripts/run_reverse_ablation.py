"""
Reverse-direction reweighting ablation.
4 runs: baseline, forward minmax, reverse uniform, reverse inverse.
Runs 2 at a time to better utilize A100.

Usage (Colab cell):
    !python scripts/run_reverse_ablation.py

    # with overrides
    !python scripts/run_reverse_ablation.py training.batch_size=256
"""

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def run_one(name, cmd):
    print(f"  START: {name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    status = "OK" if proc.returncode == 0 else "FAILED"
    if proc.returncode != 0:
        print(f"  FAILED: {name}\n{proc.stderr[-500:]}")
    else:
        # print last few lines (final metrics)
        lines = proc.stdout.strip().split("\n")
        for line in lines[-6:]:
            print(f"  [{name}] {line}")
    return name, status


def main():
    extra = sys.argv[1:]
    max_parallel = 2

    print(f"{'='*60}")
    print(f"  REVERSE REWEIGHTING ABLATION — {len(RUNS)} runs, {max_parallel} parallel")
    if extra:
        print(f"  Extra overrides: {extra}")
    print(f"{'='*60}\n")

    results = {}
    with ProcessPoolExecutor(max_workers=max_parallel) as ex:
        futures = {}
        for run in RUNS:
            cmd = [sys.executable, "scripts/train.py"] + run["overrides"] + extra
            f = ex.submit(run_one, run["name"], cmd)
            futures[f] = run["name"]

        for f in as_completed(futures):
            name, status = f.result()
            results[name] = status

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for run in RUNS:
        status = results.get(run["name"], "UNKNOWN")
        print(f"  {run['name']:<25} {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
