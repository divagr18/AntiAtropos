import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


load_dotenv()

ROOT = Path(__file__).resolve().parent
OPENENV_CANDIDATES = [
    ROOT / "openenv.yaml",
    ROOT / "AntiAtropos" / "openenv.yaml",
]


def find_openenv_yaml() -> Path:
    for candidate in OPENENV_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find openenv.yaml in the repo root or AntiAtropos/openenv.yaml"
    )


def check_openenv_yaml() -> None:
    print("\n=== Checking openenv.yaml ===")
    yaml_path = find_openenv_yaml()
    with yaml_path.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    tasks = d.get("tasks", [])
    for t in tasks:
        task_id = t.get("id")
        grader = t.get("grader")
        status = "✅" if task_id and grader else "❌"
        print(f"{status} Task '{task_id}' grader={grader.get('type') if grader else None}")
    print(f"Total tasks with graders: {sum(1 for t in tasks if t.get('grader'))}")


def check_inference() -> None:
    print("\n=== Running inference.py ===")
    env = os.environ.copy()
    groq_key = env.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    env["GROQ_API_KEY"] = groq_key
    env["API_KEY"] = groq_key
    env.setdefault("API_BASE_URL", "https://api.groq.com/openai/v1")
    env.setdefault("MODEL_NAME", "llama-3.1-8b-instant")
    env.setdefault("ENV_URL", "https://your-space.hf.space")

    r = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    for line in r.stdout.splitlines():
        if line.startswith(("[START]", "[STEP]", "[END]")):
            if "[END]" in line and "score=" in line:
                score = float(line.split("score=")[1].split()[0])
                ok = 0.0 < score < 1.0
                print(f"{'✅' if ok else '❌'} {line}")
            else:
                print(f"✅ {line}")
    if r.returncode != 0:
        print(f"❌ Crashed: {r.stderr[:200]}")


if __name__ == "__main__":
    check_openenv_yaml()
    check_inference()
