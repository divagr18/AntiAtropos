# Phase 2 Validation Guide — What Actually Works

A practical guide based on debugging Phase 2 for the SQLab environment. Every fix here came from a real validation failure and a 2-hour wait for results.

---

## Quick Checklist

Before submitting, verify ALL of these:

- [ ] `openenv.yaml` has 3+ tasks with **inline** `grader:` blocks (not `grader_id` references)
- [ ] `inference.py` runs **all tasks** in a single `python inference.py` invocation
- [ ] Each task emits its own `[START]` and `[END]` line
- [ ] `task=` field in `[START]` and `[END]` **exactly matches** task IDs in `openenv.yaml`
- [ ] All scores are **strictly between 0 and 1** (not 0.0, not 1.0)
- [ ] LLM client uses `os.environ.get("API_KEY")` and `os.environ.get("API_BASE_URL")` — no other credentials
- [ ] No `from_docker_image()` — use HTTP requests to your HF Space instead
- [ ] `pre_validation.sh` passes 3/3

---

## The Errors and How to Fix Them

### 1. "No API requests through the LiteLLM proxy"

**What happens:** The validator injects `API_KEY` and `API_BASE_URL` env vars pointing to their LiteLLM proxy, then runs your `inference.py`. If no LLM calls go through that proxy, you fail.

**Common causes:**

- **Using `from_docker_image()`**: This tries to spin up a Docker container inside the validator's environment. If it fails (permissions, disk, timeout), your script crashes before making any LLM calls. The validator sees zero proxy traffic.

- **Wrong env var names**: The validator injects `API_KEY` (not `HF_TOKEN`). If you read `HF_TOKEN` first and it's empty, your script might exit or use wrong credentials.

- **Hardcoded API keys or URLs**: If you have a fallback URL like `https://router.huggingface.co/v1` and the validator's `API_BASE_URL` is empty or unset, you bypass their proxy.

**Fix:**

```python
# Read exactly what the validator injects
API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Single OpenAI client — all LLM calls go through this
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
```

Do NOT use `from_docker_image()`. Connect to your HF Space via HTTP instead:

```python
ENV_URL = os.environ.get("ENV_URL", "https://your-space.hf.space")

# Use requests to talk to your environment
resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
resp = requests.post(f"{ENV_URL}/step", json={"action": {"command": sql}})
```

### 2. "Not enough tasks with graders"

**What happens:** The validator checks two things:
1. Your `openenv.yaml` has 3+ tasks with grader definitions
2. Your `inference.py` output has 3+ `[END]` lines with valid scores

**Cause A — Wrong openenv.yaml format:**

```yaml
# WRONG — separate graders block with references
graders:
  - id: small_grader
    entrypoint: task_graders:grade_small

tasks:
  - id: small
    grader_id: small_grader  # validator doesn't understand this
```

```yaml
# CORRECT — inline grader inside each task
tasks:
  - id: task_1
    difficulty: easy
    grader:
      type: deterministic
      endpoint: /grader
    description: >
      Your task description here.
```

Verify with:
```bash
python -c "
import yaml
d = yaml.safe_load(open('openenv.yaml'))
tasks = d.get('tasks', [])
for t in tasks:
    print(f'{t[\"id\"]}: grader={t.get(\"grader\") is not None}')
print(f'Total with graders: {sum(1 for t in tasks if t.get(\"grader\"))}')
"
```

**Cause B — inference.py runs only one task:**

The validator runs `python inference.py` **once**. If your script only handles one task (e.g. via a `TASK_NAME` env var), the validator sees only one `[END]` line and fails the "3+ tasks" check.

**Fix:** Run all tasks in a loop in a single invocation:

```python
TASKS = [
    ("task_1", "easy"),
    ("task_6", "medium"),
    ("task_12", "hard"),
    # ... add all your tasks
]

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_URL)

    for task_id, difficulty in TASKS:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        # ... run episode ...
        log_end(task=task_id, success=success, steps=steps, score=score, rewards=rewards)
```

### 3. "Task scores out of range"

**What happens:** The validator checks that each task's score is **strictly** between 0 and 1. A score of exactly `0.0` or `1.0` fails.

**Fix:** Clamp scores:

```python
score = metadata.get("grader_score", 0.0) or 0.0
score = max(0.001, min(0.999, score))
```

Also set a floor score on exceptions so a crashed task doesn't emit `score=0.000`:

```python
except Exception as exc:
    print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
    score = 0.001  # never exactly 0
```

### 4. Slow HF Space restarts after Dockerfile changes

**What happens:** HF Spaces defaults to port 7860. If you change `ENV PORT=8000` in your Dockerfile, HF might have trouble detecting when your app is ready, causing very slow restarts.

**Fix:** Keep `PORT=7860` for HF Spaces and use `socat` to forward port 8000 for OpenEnv compatibility:

```dockerfile
# Dockerfile
RUN apt-get install -y socat
ENV PORT=7860
EXPOSE 7860 8000
```

```bash
# start.sh
APP_PORT=${PORT:-7860}
if [ "$APP_PORT" != "8000" ]; then
    socat TCP-LISTEN:8000,fork,reuseaddr TCP:localhost:${APP_PORT} &
fi
exec uvicorn app:app --host 0.0.0.0 --port ${APP_PORT}
```

---

## Output Format Reference

The validator parses stdout for these exact patterns:

```
[START] task=<task_id> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] task=<task_id> success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
```

Rules:
- One `[START]`/`[END]` pair **per task** (not per script invocation)
- `task=` value must **exactly match** the `id` field in `openenv.yaml`
- `score` must be strictly `0 < score < 1`
- `done` and `success` are lowercase (`true`/`false`)
- `error` is `null` when there's no error (not empty string, not `None`)
- `[END]` must **always** be emitted, even if the task crashes (use `finally:`)

---

## Minimal Working inference.py Structure

```python
import os, json, requests
from openai import OpenAI

API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.environ.get("ENV_URL", "https://your-space.hf.space")

TASKS = [
    ("task_easy", "easy"),
    ("task_medium", "medium"),
    ("task_hard", "hard"),
]

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id, difficulty in TASKS:
        rewards = []
        steps = 0
        score = 0.001
        success = False

        print(f"[START] task={task_id} env=myenv model={MODEL_NAME}", flush=True)

        try:
            resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}).json()
            obs = resp.get("observation", {})
            done = resp.get("done", False)

            for step in range(1, 16):
                if done:
                    break

                # LLM call through validator's proxy
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Your system prompt"},
                        {"role": "user", "content": str(obs)},
                    ],
                    max_tokens=500,
                    temperature=0.0,
                )
                action = completion.choices[0].message.content.strip()

                resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": {"command": action}}
                ).json()
                obs = resp.get("observation", {})
                reward = resp.get("reward", 0.0) or 0.0
                done = resp.get("done", False)
                error = obs.get("error")

                rewards.append(reward)
                steps = step

                error_str = error if error else "null"
                print(
                    f"[STEP] step={step} action={action[:200]} "
                    f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
                    flush=True,
                )

                if done:
                    break

            score = max(0.001, min(0.999, obs.get("metadata", {}).get("grader_score", 0.0) or 0.0))
            success = obs.get("metadata", {}).get("resolved", False)

        except Exception as e:
            print(f"[DEBUG] {task_id} error: {e}", flush=True)
            score = 0.001

        finally:
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] task={task_id} success={str(success).lower()} "
                f"steps={steps} score={score:.3f} rewards={rewards_str}",
                flush=True,
            )

if __name__ == "__main__":
    main()
```

---

## Local Validation Script

Run this before every submission:

```python
# local_validator.py
import yaml

def check():
    d = yaml.safe_load(open("openenv.yaml"))
    tasks = d.get("tasks", [])
    ok = 0
    for t in tasks:
        has_grader = t.get("grader") is not None
        print(f"  {'OK' if has_grader else 'FAIL'} {t['id']} grader={has_grader}")
        if has_grader:
            ok += 1
    print(f"\nTasks with graders: {ok}/{len(tasks)}")
    print(f"{'PASS' if ok >= 3 else 'FAIL'}: need at least 3")

check()
```

```bash
# Also run pre_validation.sh
bash pre_validation.sh https://your-space.hf.space your_env_dir/
```

---

## Timeline of Our Failures

| Attempt | Error | Root Cause | Fix |
|---------|-------|-----------|-----|
| 1 | No API calls through proxy | `from_docker_image()` crashed in validator | Switch to HTTP client |
| 2 | No API calls through proxy | `API_KEY` read as `HF_TOKEN` | Use `os.environ.get("API_KEY")` |
| 3 | Not enough tasks with graders | inference.py ran 1 task | Loop through all tasks |
| 4 | Task scores out of range | Score was exactly 0.0 | Clamp to (0.001, 0.999) |
| 5 | Not enough tasks with graders | openenv.yaml missing inline graders | Add `grader:` block inside each task |
| 6 | PASSED | - | - |

Each attempt cost ~2 hours of wait time. Save yourself the pain — check everything locally first.