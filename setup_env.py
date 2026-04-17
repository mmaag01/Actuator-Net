"""Actuator Net — environment setup script.

Run once from the Miniforge Prompt in the project root:
    python setup_env.py

The script will:
  1. Detect whether an NVIDIA GPU is present.
  2. Create (or update) the 'actuator-net' conda environment from the
     appropriate .yml file.
  3. Register the environment as a Jupyter kernel so VS Code can find it.
  4. Verify the installation by importing every project dependency.
"""

import subprocess
import sys
from pathlib import Path

ENV_NAME    = "actuator-net"
ROOT        = Path(__file__).parent
CPU_YML     = ROOT / "environment.yml"
GPU_YML     = ROOT / "environment-gpu.yml"

VERIFY_SCRIPT = (
    "import torch, numpy, pandas, sklearn, matplotlib, joblib, xgboost; "
    "cuda = torch.cuda.is_available(); "
    "print(f'PyTorch {torch.__version__} | CUDA available: {cuda}'); "
    "print(f'GPU: {torch.cuda.get_device_name(0)}') if cuda else None"
)


def run(cmd: list[str], check: bool = True) -> bool:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"  [ERROR] Command failed (exit {result.returncode})")
        return False
    return result.returncode == 0


def has_nvidia() -> bool:
    return subprocess.run(
        ["nvidia-smi"], capture_output=True
    ).returncode == 0


def env_exists() -> bool:
    result = subprocess.run(
        ["conda", "env", "list"], capture_output=True, text=True
    )
    return ENV_NAME in result.stdout


def main() -> int:
    print("Actuator Net — Environment Setup")
    print("=" * 42)

    gpu = has_nvidia()
    yml = GPU_YML if gpu else CPU_YML
    tag = "GPU (CUDA 12.8, RTX 5070 Ti)" if gpu else "CPU (integrated graphics)"
    print(f"Hardware : {tag}")
    print(f"Env file : {yml.name}\n")

    assert yml.exists(), f"Environment file not found: {yml}"

    # ── Create or update the conda environment ────────────────────────────
    if env_exists():
        print(f"Environment '{ENV_NAME}' already exists — updating …")
        ok = run(["conda", "env", "update", "-f", str(yml),
                  "--name", ENV_NAME, "--prune"])
    else:
        print(f"Creating environment '{ENV_NAME}' …")
        ok = run(["conda", "env", "create", "-f", str(yml), "--name", ENV_NAME])

    if not ok:
        print("\n[ERROR] Environment creation failed. Check the output above.")
        return 1

    # ── Register as a Jupyter kernel ──────────────────────────────────────
    display = f"Actuator Net ({'GPU' if gpu else 'CPU'})"
    print(f"\nRegistering Jupyter kernel as '{display}' …")
    run([
        "conda", "run", "-n", ENV_NAME,
        "python", "-m", "ipykernel", "install",
        "--user", "--name", ENV_NAME, "--display-name", display,
    ])

    # ── Verify imports ────────────────────────────────────────────────────
    print("\nVerifying installation …")
    ok = run([
        "conda", "run", "-n", ENV_NAME,
        "python", "-c", VERIFY_SCRIPT,
    ])

    if not ok:
        print("\n[WARNING] Verification failed — check the output above.")
        return 1

    print("\nSetup complete!")
    print("-" * 42)
    print(f"  conda activate {ENV_NAME}")
    print(f"  python train.py --model gru")
    return 0


if __name__ == "__main__":
    sys.exit(main())
