#!/usr/bin/env python3
"""
launch_smoke.py — Launch AntiAtropos smoke training on Hugging Face Jobs.

Uses `hf jobs uv run` under the hood (via run_uv_job API).

Prerequisites:
  1. pip install "huggingface_hub[hf_x11]>=0.25.0"
  2. huggingface-cli login   (or set HF_TOKEN env var)
  3. HF Pro/Team account (required for GPU jobs)

Usage:
  # Quick smoke test (~10 min, t4-medium ≈ $0.40):
  python training/launch_smoke.py

  # Custom flavor / longer timeout:
  python training/launch_smoke.py --flavor a10g-small --timeout 2h

  # Dry-run:
  python training/launch_smoke.py --dry-run
"""

import argparse
import os
import sys
from datetime import datetime

from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Launch AntiAtropos smoke training via huggingface_hub.run_uv_job"
    )
    parser.add_argument("--flavor", default="t4-medium",
                        help="GPU flavor (default: t4-medium). "
                             "Run 'hf jobs hardware' for full list.")
    parser.add_argument("--timeout", default="45m",
                        help="Job timeout (default: 45m). "
                             "Examples: 30m, 2h, 7200")
    parser.add_argument("--repo", default="pranavkk/AntiAtropos",
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
    bootstrap_script = str(TRAINING_DIR / "run_smoke_uv.py")

    # ---- Print summary ----
    print("=" * 50)
    print(" AntiAtropos Smoke Test — HF Jobs (uv)")
    print("=" * 50)
    print(f"  Flavor:        {args.flavor}")
    print(f"  Timeout:       {args.timeout}")
    print(f"  Repo:          {args.repo}")
    print(f"  Run ID:        {run_id}")
    print(f"  Iterations:    {args.num_iterations}")
    print(f"  Episodes:      {args.num_episodes}")
    print(f"  Steps/ep:      {args.max_steps}")
    print(f"  Bootstrap:     {bootstrap_script}")
    print("=" * 50)

    if not os.path.isfile(bootstrap_script):
        print(f"\nERROR: {bootstrap_script} not found.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Would launch with run_uv_job()")
        print(f"  script_path: {bootstrap_script}")
        print(f"  script_args: --smoke --run-id {run_id}")
        print(f"  flavor:      {args.flavor}")
        print(f"  timeout:     {args.timeout}")
        print(f"  secrets:     HF_TOKEN")
        print(f"  env:         REPO={args.repo}, RUN_ID={run_id}")
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

    # ---- Launch via run_uv_job ----
    try:
        from huggingface_hub import run_uv_job
    except ImportError:
        print("\nERROR: huggingface_hub too old. Run:")
        print("  pip install 'huggingface_hub>=0.25.0'")
        sys.exit(1)

    print("\nLaunching job via hf jobs uv run...")
    job = run_uv_job(
        bootstrap_script,
        script_args=[
            "--smoke",
            "--run-id", run_id,
            "--num-iterations", str(args.num_iterations),
            "--num-episodes", str(args.num_episodes),
            "--max-steps", str(args.max_steps),
            "--eval-interval", str(args.eval_interval),
            "--plot-interval", str(args.plot_interval),
        ],
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": hf_token} if hf_token else {},
        env={
            "REPO": args.repo,
            "RUN_ID": run_id,
        },
    )

    print(f"\nJob launched! ID: {job.id}")
    print(f"  Monitor: {job.url}")
    print(f"  Logs:    hf jobs logs {job.id}")

    # ---- Stream logs ----
    print("\nStreaming logs (Ctrl+C to stop watching)...\n")
    try:
        from huggingface_hub import fetch_job_logs, inspect_job
        import time
        seen = 0
        while True:
            try:
                info = inspect_job(job_id=job.id)
                status = info.status.stage
            except Exception:
                status = None

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
