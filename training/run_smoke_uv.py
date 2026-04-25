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

print("[bootstrap] Installing full dependencies...")
subprocess.run(
    ["uv", "pip", "install", "--no-deps", "torchvision", "-q"],
    check=True,
)
subprocess.run(
    ["uv", "pip", "install", "-r", "training/requirements.txt", "-q"],
    check=True,
)

print("[bootstrap] Starting local AntiAtropos server (simulated mode)...")
import os as _os
_os.environ["ANTIATROPOS_ENV_MODE"] = "simulated"
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server.app:app",
     "--host", "127.0.0.1", "--port", "8000"],
)

# Wait for server
import time as _time
for _ in range(30):
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
        print("[bootstrap] Server ready.")
        break
    except Exception:
        _time.sleep(1)

print("[bootstrap] Launching training (local server)...")
_os.environ["ANTIATROPOS_ENV_URL"] = "http://localhost:8000"
cmd = [sys.executable, "training/train.py"] + sys.argv[1:]
try:
    subprocess.run(cmd, check=True)
finally:
    print("[bootstrap] Stopping server...")
    server_proc.terminate()
    server_proc.wait()

print("[bootstrap] Smoke training complete.")
