import subprocess
import sys
from pathlib import Path


SWEEPS = {
    "ratio":   "configs/sweep/ratio_sweep.yaml",
    "guide":   "configs/sweep/guide_sweep.yaml",
    "scheme":  "configs/sweep/scheme_sweep.yaml",
    "k":       "configs/sweep/k_sweep.yaml",
    "model":   "configs/sweep/model_sweep.yaml",
    "full":    "configs/sweep/full_ablation.yaml",
}


def run_sweep(sweep_name: str) -> None:
    if sweep_name not in SWEEPS:
        print(f"Unknown sweep: {sweep_name}")
        print(f"Available: {list(SWEEPS.keys())}")
        sys.exit(1)

    sweep_cfg = SWEEPS[sweep_name]

    if not Path(sweep_cfg).exists():
        print(f"Sweep config not found: {sweep_cfg}")
        sys.exit(1)

    print(f"\nLaunching sweep: {sweep_name}")
    print(f"Config: {sweep_cfg}")
    print("=" * 55)

    cmd = [
        "python", "scripts/train.py",
        f"--config-path=../configs/sweep",
        f"--config-name={Path(sweep_cfg).stem}",
        "--multirun",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/sweep.py <sweep_name>")
        print(f"Available sweeps: {list(SWEEPS.keys())}")
        sys.exit(1)

    run_sweep(sys.argv[1])