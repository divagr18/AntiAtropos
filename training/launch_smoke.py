#!/usr/bin/env python3
"""
launch_smoke.py — Launch AntiAtropos smoke training on Hugging Face Jobs.

Uses Docker-style job (run_job) with a CUDA image so GPU is available.
The bootstrap script (run_smoke_uv.py) handles cloning + pip install + training.

Prerequisites:
  1. pip install "huggingface_hub>=0.25.0"
  2. huggingface-cli login   (or set HF_TOKEN env var)
  3. HF Pro/Team account (required for GPU jobs)

Usage:
  # Quick smoke test (a10g-small ≈ $0.17):
  python training/launch_smoke.py

  # Custom flavor / longer timeout:
  python training/launch_smoke.py --flavor a10g-large --timeout 2h

  # Dry-run:
  python training/launch_smoke.py --dry-run
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent

DOCKER_IMAGE = "pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel"


def build_job_command() -> str:
    """Build the shell script that runs INSIDE the HF Job container."""
    return (
        "set -e\n"
        "\n"
        "echo '[bootstrap] Installing git...'\n"
        "apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1\n"
        "\n"
        "echo '[bootstrap] Cloning '$REPO'...'\n"
        "mkdir -p /workspace\n"
        "git clone --depth 1 https://hf:${HF_TOKEN}@huggingface.co/$REPO /workspace/AntiAtropos\n"
        "cd /workspace/AntiAtropos\n"
        "\n"
        "echo '[bootstrap] Installing dependencies...'\n"
        "pip install --break-system-packages --no-deps torchvision -q\n"
        "pip install --break-system-packages flash-attn --no-build-isolation -q\n"
        "pip install --break-system-packages -r training/requirements.txt -q\n"
        "\n"
        "echo '[bootstrap] Launching training...'\n"
        "python training/train.py "
        "--smoke "
        "--run-id $RUN_ID "
        "--num-iterations $NUM_ITERATIONS "
        "--num-episodes $NUM_EPISODES "
        "--max-steps $MAX_STEPS "
        "--eval-interval $EVAL_INTERVAL "
        "--plot-interval $PLOT_INTERVAL\n"
        "\n"
        "echo '[bootstrap] Done.'"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Launch AntiAtropos smoke training on Hugging Face Jobs"
    )
    parser.add_argument("--flavor", default="a10g-small",
                        help="GPU flavor (default: a10g-small). "
                             "Run 'hf jobs hardware' for full list.")
    parser.add_argument("--timeout", default="45m",
                        help="Job timeout (default: 45m). "
                             "Examples: 30m, 2h, 7200")
    parser.add_argument("--repo", default="Keshav051/AntiAtropos",
                        help="HuggingFace repo to clone")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier (default: smoke_YYYYMMDD_HHMMSS)")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Training iterations (default: 10)")
    parser.add_argument("--num-episodes", type=int, default=2,
                        help="Episodes per iteration (default: 2)")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max steps per episode (default: 20)")
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Eval every N iterations (default: 5)")
    parser.add_argument("--plot-interval", type=int, default=5,
                        help="Plot every N iterations (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without launching")
    args = parser.parse_args()

    run_id = args.run_id or f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ---- Print summary ----
    print("=" * 50)
    print(" AntiAtropos Smoke Test — HF Jobs")
    print("=" * 50)
    print(f"  Image:         {DOCKER_IMAGE}")
    print(f"  Flavor:        {args.flavor}")
    print(f"  Timeout:       {args.timeout}")
    print(f"  Repo:          {args.repo}")
    print(f"  Run ID:        {run_id}")
    print(f"  Iterations:    {args.num_iterations}")
    print(f"  Episodes:      {args.num_episodes}")
    print(f"  Steps/ep:      {args.max_steps}")
    print("=" * 50)

    if args.dry_run:
        print("\n[DRY RUN] Command that would run inside container:")
        print(build_job_command())
        return

    # ---- Resolve HF_TOKEN ----
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(token_path):
            with open(token_path) as f:
                hf_token = f.read().strip()
    if not hf_token:
        print("\nWARNING: HF_TOKEN not found. The job may fail to push to Hub.")
        print("  Set HF_TOKEN env var or run: huggingface-cli login")
        secrets_dict = {}
    else:
        secrets_dict = {"HF_TOKEN": hf_token}

    # ---- Launch via run_job (Docker-style) ----
    try:
        from huggingface_hub import run_job
    except ImportError:
        print("\nERROR: huggingface_hub too old. Run:")
        print("  pip install 'huggingface_hub>=0.25.0'")
        sys.exit(1)

    job_command = build_job_command().replace("\r", "")

    print("\nLaunching job...")
    job = run_job(
        image=DOCKER_IMAGE,
        command=["bash", "-c", job_command],
        flavor=args.flavor,
        timeout=args.timeout,
        secrets=secrets_dict,
        env={
            "REPO": args.repo,
            "RUN_ID": run_id,
            "NUM_ITERATIONS": str(args.num_iterations),
            "NUM_EPISODES": str(args.num_episodes),
            "MAX_STEPS": str(args.max_steps),
            "EVAL_INTERVAL": str(args.eval_interval),
            "PLOT_INTERVAL": str(args.plot_interval),
        },
    )

    print(f"\nJob launched! ID: {job.id}")
    print(f"  Monitor: {job.url}")
    print(f"  Logs:    hf jobs logs {job.id}")
    print(f"  Cancel:  hf jobs cancel {job.id}")

    # ---- Stream logs ----
    print("\nStreaming logs (Ctrl+C to stop watching)...\n")
    try:
        from huggingface_hub import fetch_job_logs, inspect_job
        import time
        seen = 0
        while True:
            status = None
            try:
                info = inspect_job(job_id=job.id)
                status = info.status.stage
            except Exception:
                pass

            try:
                logs = list(fetch_job_logs(job_id=job.id))
                for line in logs[seen:]:
                    print(line, end="" if line.endswith("\n") else "\n")
                seen = len(logs)
            except Exception:
                pass

            if status in ("COMPLETED", "ERROR", "CANCELED"):
                print(f"\nJob finished with status: {status}")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nStopped watching logs. Job still running remotely.")
        print(f"  Check status: hf jobs inspect {job.id}")
        print(f"  Resume logs:  hf jobs logs {job.id}")


if __name__ == "__main__":
    main()
