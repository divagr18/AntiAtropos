# /// script
# dependencies = [
#   "requests>=2.31.0",
#   "pyyaml>=6.0",
# ]
# ///
"""
AntiAtropos smoke training bootstrap for hf jobs uv run.

Usage (from repo root):
  hf jobs uv run --flavor t4-medium --timeout 45m --secrets HF_TOKEN training/run_smoke_uv.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("REPO", "Keshav051/AntiAtropos")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
WORKSPACE = Path("/workspace")

print(f"[bootstrap] Cloning {REPO}...")
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Build auth URL
if HF_TOKEN:
    clone_url = f"https://hf:{HF_TOKEN}@huggingface.co/{REPO}"
else:
    clone_url = f"https://huggingface.co/{REPO}"

subprocess.run(
    ["git", "clone", "--depth", "1", clone_url, str(WORKSPACE / "AntiAtropos")],
    check=True,
)

os.chdir(str(WORKSPACE / "AntiAtropos"))

print("[bootstrap] Upgrading torch to satisfy torchao dependency...")
subprocess.run(
    ["uv", "pip", "install", "torch>=2.5.0", "-q"],
    check=True,
)

print("[bootstrap] Installing full dependencies...")
subprocess.run(
    ["uv", "pip", "install", "-r", "training/requirements.txt", "-q"],
    check=True,
)

print("[bootstrap] Launching smoke training...")
cmd = [sys.executable, "training/train.py"] + sys.argv[1:]
subprocess.run(cmd, check=True)

print("[bootstrap] Smoke training complete.")
