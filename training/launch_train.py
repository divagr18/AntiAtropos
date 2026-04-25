#!/usr/bin/env python3
"""
launch_train.py — Launch full AntiAtropos training on Hugging Face Jobs.

Pushes model checkpoints, metrics dataset, and plots to HF Hub.
The local server is co-located for zero-latency environment interaction.
Supports automatic resume from latest Hub checkpoint.

Prerequisites:
  1. pip install "huggingface_hub>=0.25.0"
  2. huggingface-cli login   (or set HF_TOKEN env var)
  3. HF Pro/Team account (required for GPU jobs)
  4. The Hub model and dataset repos are auto-created if they don't exist.
     Alternatively create them manually:
       hf repo create <hub-model-repo> --type model
       hf repo create <hub-metrics-dataset> --type dataset

Lifecycle:
  ┌─────────────────────────────────────────────┐
  │  HF Job (A10G, ~4h)                        │
  │  ┌──────────────────────────────────────┐   │
  │  │  uvicorn :8000  ←──→  train.py       │   │
  │  │  (simulator)          (GPU model)     │   │
  │  └──────────┬───────────────────────────┘   │
  │             │ push adapter + plots           │
  │             ▼                                │
  │      HF Hub Model Repo                       │
  │      (checkpoint-25, checkpoint-50, ...)     │
  │             │ push metrics.jsonl              │
  │             ▼                                │
  │      HF Hub Metrics Dataset                  │
  └─────────────────────────────────────────────┘

Usage:
  # Quick test (∼10 min):
  python training/launch_train.py \
    --hub-model-repo Keshav051/antiatropos-qlora \
    --hub-metrics-dataset Keshav051/antiatropos-training-metrics \
    --num-iterations 20 --num-episodes 4

  # Full training (a10g-large ≈ $0.34/hr, ∼2h):
  python training/launch_train.py \
    --hub-model-repo Keshav051/antiatropos-qlora \
    --hub-metrics-dataset Keshav051/antiatropos-training-metrics

  # Custom flavor / longer timeout:
  python training/launch_train.py \
    --hub-model-repo Keshav051/antiatropos-qlora \
    --hub-metrics-dataset Keshav051/antiatropos-training-metrics \
    --flavor a10g-xlarge --timeout 12h \
    --num-iterations 2000 --num-episodes 24

  # Resume from latest Hub checkpoint:
  python training/launch_train.py \
    --hub-model-repo Keshav051/antiatropos-qlora \
    --hub-metrics-dataset Keshav051/antiatropos-training-metrics \
    --run-id exp_002

  # Dry run (prints job command without launching):
  python training/launch_train.py \
    --hub-model-repo Keshav051/antiatropos-qlora \
    --hub-metrics-dataset Keshav051/antiatropos-training-metrics \
    --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

TRAINING_DIR = Path(__file__).resolve().parent

DOCKER_IMAGE = "pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel"

DEFAULT_NUM_ITERATIONS = 500
DEFAULT_NUM_EPISODES = 12
DEFAULT_MAX_STEPS = 40
DEFAULT_EVAL_INTERVAL = 50
DEFAULT_CHECKPOINT_INTERVAL = 25
DEFAULT_PLOT_INTERVAL = 25


def build_job_command() -> str:
    """Build the shell script that runs INSIDE the HF Job container.

    Starts the AntiAtropos FastAPI server locally (eliminating HTTP latency)
    then runs training against localhost:8000 with Hub persistence.
    """
    return (
        "set -e\n"
        "\n"
        "echo '[bootstrap] Installing git...'\n"
        "apt-get update -qq && apt-get install -y -qq git netcat-openbsd > /dev/null 2>&1\n"
        "\n"
        "echo '[bootstrap] Cloning $REPO...'\n"
        "mkdir -p /workspace\n"
        "git clone --depth 1 https://hf:${HF_TOKEN}@huggingface.co/$REPO /workspace/AntiAtropos\n"
        "cd /workspace/AntiAtropos\n"
        "\n"
        "echo '[bootstrap] Installing dependencies...'\n"
        "pip install --break-system-packages --no-deps torchvision -q\n"
        "pip install --break-system-packages -r training/requirements.txt -q\n"
        "\n"
        "echo '[bootstrap] Starting local AntiAtropos server (simulated mode)...'\n"
        "export ANTIATROPOS_ENV_MODE=simulated\n"
        "uvicorn server.app:app --host 127.0.0.1 --port 8000 &\n"
        "SERVER_PID=$!\n"
        "\n"
        "# Wait for server to be ready\n"
        "echo '[bootstrap] Waiting for server...'\n"
        "for i in $(seq 1 30); do\n"
        "  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then\n"
        "    echo '[bootstrap] Server ready.'\n"
        "    break\n"
        "  fi\n"
        "  sleep 1\n"
        "done\n"
        "\n"
        "echo '[bootstrap] Launching training (local server, Hub persistence)...'\n"
        "ANTIATROPOS_HUB_MODEL_REPO=$HUB_MODEL_REPO "
        "ANTIATROPOS_HUB_METRICS_DATASET=$HUB_METRICS_DATASET "
        "ANTIATROPOS_ENV_URL=http://localhost:8000 "
        "python training/train.py "
        "--run-id $RUN_ID "
        "--num-iterations $NUM_ITERATIONS "
        "--num-episodes $NUM_EPISODES "
        "--max-steps $MAX_STEPS "
        "--eval-interval $EVAL_INTERVAL "
        "--checkpoint-interval $CHECKPOINT_INTERVAL "
        "--plot-interval $PLOT_INTERVAL\n"
        "TRAIN_EXIT=$?\n"
        "\n"
        "echo '[bootstrap] Stopping server...'\n"
        "kill $SERVER_PID 2>/dev/null || true\n"
        "wait $SERVER_PID 2>/dev/null || true\n"
        "\n"
        "exit $TRAIN_EXIT"
    )


def ensure_hub_repos(
    hub_model_repo: str,
    hub_metrics_dataset: str,
    hf_token: Optional[str],
) -> None:
    """Check if Hub repos exist; create them automatically if not."""
    if not hf_token:
        print("  [hub] No HF_TOKEN available, skipping repo check")
        return

    try:
        from huggingface_hub import HfApi

        api = HfApi()

        for repo_id, repo_type in [
            (hub_model_repo, "model"),
            (hub_metrics_dataset, "dataset"),
        ]:
            base_url = (
                "https://huggingface.co"
                if repo_type == "model"
                else "https://huggingface.co/datasets"
            )
            try:
                info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
                print(f"  [hub] Repo OK: {base_url}/{repo_id}")
            except Exception:
                print(f"  [hub] Creating repo: {repo_id} ({repo_type})...")
                api.create_repo(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    private=True,
                    exist_ok=True,
                )
                print(f"  [hub] Created: {base_url}/{repo_id}")
    except Exception as e:
        print(f"\n  [hub] WARNING: Could not verify/create Hub repos: {e}")
        print("  [hub] Create them manually:")
        print(f"    hf repo create {hub_model_repo} --type model")
        print(f"    hf repo create {hub_metrics_dataset} --type dataset")
        print(f"    Then visit:")
        print(f"      https://huggingface.co/{hub_model_repo}")
        print(f"      https://huggingface.co/datasets/{hub_metrics_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AntiAtropos Full Training — HF Jobs with Hub persistence"
    )
    parser.add_argument(
        "--flavor",
        default="a10g-large",
        help="GPU flavor (default: a10g-large). Run 'hf jobs hardware' for full list.",
    )
    parser.add_argument(
        "--timeout",
        default="4h",
        help="Job timeout (default: 4h). Examples: 30m, 2h, 7200",
    )
    parser.add_argument(
        "--repo",
        default="Keshav051/AntiAtropos",
        help="HF repo to clone (project source code)",
    )
    parser.add_argument(
        "--hub-model-repo",
        required=True,
        help="HF Hub model repo for checkpoints, final adapter, and plots "
        "(e.g. Keshav051/antiatropos-qlora)",
    )
    parser.add_argument(
        "--hub-metrics-dataset",
        required=True,
        help="HF Hub dataset repo for training metrics.jsonl "
        "(e.g. Keshav051/antiatropos-training-metrics)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier (default: train_YYYYMMDD_HHMMSS). "
        "Use same ID to resume a previous run.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help=f"Training iterations (default: {DEFAULT_NUM_ITERATIONS})",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=f"Episodes per iteration (default: {DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=DEFAULT_EVAL_INTERVAL,
        help=f"Evaluate every N iterations (default: {DEFAULT_EVAL_INTERVAL})",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Checkpoint every N iterations (default: {DEFAULT_CHECKPOINT_INTERVAL})",
    )
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=DEFAULT_PLOT_INTERVAL,
        help=f"Plot every N iterations (default: {DEFAULT_PLOT_INTERVAL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without launching",
    )
    parser.add_argument(
        "--no-create-repos",
        action="store_true",
        help="Skip automatic Hub repo creation",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ---- Print summary ----
    print("=" * 60)
    print(" ANTIATROPOS FULL TRAINING — HF Jobs")
    print("=" * 60)
    print(f"  Image:               {DOCKER_IMAGE}")
    print(f"  Flavor:              {args.flavor}")
    print(f"  Timeout:             {args.timeout}")
    print(f"  Code repo:           {args.repo}")
    print(f"  Hub model repo:      {args.hub_model_repo}")
    print(f"  Hub metrics dataset: {args.hub_metrics_dataset}")
    print(f"  Run ID:              {run_id}")
    print(f"  Iterations:          {args.num_iterations}")
    print(f"  Episodes/iter:       {args.num_episodes}")
    print(f"  Steps/episode:       {args.max_steps}")
    print(f"  Eval interval:       {args.eval_interval}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print(f"  Plot interval:       {args.plot_interval}")
    print("=" * 60)

    # Estimated time
    est_hours = (
        args.num_iterations
        * args.num_episodes
        * args.max_steps
        * 0.04  # ~40ms per step with parallel episodes
        / 3600
    )
    print(f"  Est. runtime:  ~{est_hours:.1f}h (at 40ms/step)")
    print(f"  Est. cost:     ~${est_hours * 0.34:.2f} (a10g-large at $0.34/hr)")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Job command:")
        print(build_job_command())
        print("\n[DRY RUN] To launch manually inside the container:")
        print(
            "  python training/train.py \\\n"
            f"    --run-id {run_id} \\\n"
            f"    --num-iterations {args.num_iterations} \\\n"
            f"    --num-episodes {args.num_episodes} \\\n"
            f"    --max-steps {args.max_steps} \\\n"
            f"    --eval-interval {args.eval_interval} \\\n"
            f"    --checkpoint-interval {args.checkpoint_interval} \\\n"
            f"    --plot-interval {args.plot_interval}"
        )
        return

    # ---- Resolve HF_TOKEN ----
    hf_token: Optional[str] = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not hf_token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(token_path):
            with open(token_path) as f:
                hf_token = f.read().strip()

    secrets_dict: dict = {}
    if not hf_token:
        print("\nWARNING: HF_TOKEN not found. Job will FAIL to push to Hub.")
        print("  Set HF_TOKEN env var or run: huggingface-cli login")
    else:
        secrets_dict = {"HF_TOKEN": hf_token}

    # ---- Ensure Hub repos exist ----
    if not args.no_create_repos and hf_token:
        ensure_hub_repos(
            args.hub_model_repo, args.hub_metrics_dataset, hf_token
        )

    # ---- Launch via run_job ----
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
            "HUB_MODEL_REPO": args.hub_model_repo,
            "HUB_METRICS_DATASET": args.hub_metrics_dataset,
            "NUM_ITERATIONS": str(args.num_iterations),
            "NUM_EPISODES": str(args.num_episodes),
            "MAX_STEPS": str(args.max_steps),
            "EVAL_INTERVAL": str(args.eval_interval),
            "CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
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
            status: Optional[str] = None
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
