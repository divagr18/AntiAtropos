import asyncio
import inspect
import json
import os
import random
import textwrap
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI

from AntiAtropos.client import AntiAtroposEnv
from AntiAtropos.grader import EpisodeGrader
from AntiAtropos.models import ActionType, SREAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    # Local fallback to keep developer runs convenient.
    API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

DEFAULT_ENV_URL = "https://pranavkk-antiatropos.hf.space"
ENV_URL = os.getenv("ENV_URL") or os.getenv("ANTIATROPOS_ENV_URL") or DEFAULT_ENV_URL
ENV_MODE = os.getenv("ANTIATROPOS_MODE", "simulated")
TASK_NAME = os.getenv("ANTIATROPOS_TASK", "task-1")
BENCHMARK = os.getenv("ANTIATROPOS_BENCHMARK", "antiatropos")

MAX_STEPS_PER_TASK = 60
MESSAGE_TIMEOUT_S = 300
MODEL_TIMEOUT_S = 25

TEMPERATURE = float(os.getenv("ANTIATROPOS_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("ANTIATROPOS_MAX_TOKENS", "180"))
SEED = int(os.getenv("ANTIATROPOS_SEED", "42"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("ANTIATROPOS_SUCCESS_THRESHOLD", "0.55"))

TASK_BRIEFS: Dict[str, str] = {
    "task-1": "Traffic increases linearly. Scale proactively to keep latency low and cost efficient.",
    "task-2": "A node fails randomly. Detect quickly and recover with reroute/scale actions.",
    "task-3": "Protect VIP node-0 under surges. Keep VIP healthy without invalid actions.",
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous SRE controller managing a five-node microservice cluster.

    Return exactly one JSON object:
    {
      "action_type": "SCALE_UP" | "SCALE_DOWN" | "REROUTE_TRAFFIC" | "SHED_LOAD" | "NO_OP",
      "target_node_id": "node-0" | "node-1" | "node-2" | "node-3" | "node-4",
      "parameter": 0.0
    }
    """
).strip()


class InferenceError(RuntimeError):
    """Raised when inference execution fails for a task."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _task_seed(base_seed: int, task_id: str) -> int:
    offsets = {"task-1": 0, "task-2": 1, "task-3": 2}
    return int(base_seed + offsets.get(task_id, 0))


def _strict_score(score: float, eps: float = 0.001) -> float:
    del eps
    return min(1.0, max(0.0, float(score)))


def _hf_web_fallback_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if host.endswith(".hf.space") and path == "":
        return base_url.rstrip("/") + "/web"
    return base_url


@asynccontextmanager
async def open_env(message_timeout_s: int):
    # Precedence rule: use ENV_URL when both ENV_URL and LOCAL_IMAGE_NAME are set.
    if ENV_URL:
        try:
            async with AntiAtroposEnv(ENV_URL, message_timeout_s=message_timeout_s) as env:
                yield env
                return
        except ConnectionError as e:
            fallback_url = _hf_web_fallback_url(ENV_URL)
            if fallback_url == ENV_URL or "404" not in str(e):
                raise
            async with AntiAtroposEnv(fallback_url, message_timeout_s=message_timeout_s) as env:
                yield env
        return

    if LOCAL_IMAGE_NAME:
        env = AntiAtroposEnv.from_docker_image(LOCAL_IMAGE_NAME)
        try:
            yield env
        finally:
            close_result = env.close()
            if inspect.isawaitable(close_result):
                await close_result
        return

    raise RuntimeError("Missing environment target. Set ENV_URL/ANTIATROPOS_ENV_URL or LOCAL_IMAGE_NAME.")


def build_user_prompt(task_id: str, step: int, obs: dict, history: List[str]) -> str:
    recent = "\n".join(history[-4:]) if history else "None"
    brief = TASK_BRIEFS.get(task_id, "Maintain SLA, stability, and efficient cost.")
    return textwrap.dedent(
        f"""
        Task: {task_id}
        Objective: {brief}
        Step: {step}

        Current state:
        {json.dumps(obs, separators=(",", ":"))}

        Recent decisions:
        {recent}

        Choose the next SRE action.
        """
    ).strip()


def observation_for_model(obs) -> dict:
    return {
        "task_id": obs.task_id,
        "mode": getattr(obs.mode, "value", str(obs.mode)),
        "step": obs.step,
        "max_steps": obs.max_steps,
        "lyapunov_energy": obs.lyapunov_energy,
        "average_latency_ms": obs.average_latency_ms,
        "error_rate": obs.error_rate,
        "total_queue_backlog": obs.total_queue_backlog,
        "sla_violations": obs.sla_violations,
        "invalid_action_count": obs.invalid_action_count,
        "nodes": [
            {
                "node_id": node.node_id,
                "status": getattr(node.status, "value", str(node.status)),
                "is_vip": node.is_vip,
                "queue_depth": node.queue_depth,
                "latency_ms": node.latency_ms,
                "incoming_request_rate": node.incoming_request_rate,
                "cpu_utilization": node.cpu_utilization,
            }
            for node in obs.nodes
        ],
    }


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(stripped[start : end + 1])


def _parse_action(payload: dict) -> SREAction:
    action_type = str(payload.get("action_type", "NO_OP")).upper()
    target_node_id = str(payload.get("target_node_id", "node-0"))
    parameter = float(payload.get("parameter", 0.0))
    return SREAction(
        action_type=ActionType(action_type),
        target_node_id=target_node_id,
        parameter=parameter,
    )


async def get_model_action(client: AsyncOpenAI, task_id: str, step: int, obs: dict, history: List[str]) -> SREAction:
    prompt = build_user_prompt(task_id=task_id, step=step, obs=obs, history=history)
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            timeout=MODEL_TIMEOUT_S,
            seed=SEED,
        )
        content = completion.choices[0].message.content or ""
        if not content.strip():
            raise InferenceError("Model returned empty content.")
        return _parse_action(_extract_json_object(content))
    except Exception as exc:
        raise InferenceError(f"Model inference failed at step {step} for {task_id}: {exc}") from exc


def _compact_action(action: SREAction) -> str:
    payload = {
        "action_type": action.action_type.value,
        "target_node_id": action.target_node_id,
        "parameter": round(float(action.parameter), 4),
    }
    return json.dumps(payload, separators=(",", ":"))


async def run_single_task(env: AntiAtroposEnv, client: AsyncOpenAI, task_id: str) -> dict:
    task_seed = _task_seed(SEED, task_id)
    result = await env.reset(task_id=task_id, mode=ENV_MODE, seed=task_seed)

    grader = EpisodeGrader(task_id=task_id)
    grader.record(result.observation)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    for step in range(1, MAX_STEPS_PER_TASK + 1):
        if result.done:
            break

        action = await get_model_action(
            client=client,
            task_id=task_id,
            step=step,
            obs=observation_for_model(result.observation),
            history=history,
        )
        result = await env.step(action)
        grader.record(result.observation)

        reward = float(result.reward or 0.0)
        rewards.append(reward)
        steps_taken = step
        action_str = _compact_action(action)
        history.append(f"step={step} action={action_str} reward={reward:.2f}")

        error = getattr(result.observation, "last_action_error", None)
        log_step(step=step, action=action_str, reward=reward, done=bool(result.done), error=error)

    grade = grader.score()
    score = _strict_score(float(grade.composite))
    success = score >= SUCCESS_SCORE_THRESHOLD
    return {
        "task_id": task_id,
        "success": success,
        "score": score,
        "steps": steps_taken,
        "rewards": rewards,
    }


async def run_all_tasks() -> None:
    _seed_everything(SEED)
    all_tasks = ["task-1", "task-2", "task-3"]
    run_single = os.getenv("ANTIATROPOS_RUN_SINGLE_TASK", "false").lower() == "true"
    task_id = TASK_NAME if TASK_NAME in set(all_tasks) else "task-1"
    tasks_to_run = [task_id] if run_single else all_tasks
    if not API_KEY:
        raise RuntimeError("Missing API key (API_KEY/HF_TOKEN/OPENAI_API_KEY).")

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        async with open_env(MESSAGE_TIMEOUT_S) as env:
            for task in tasks_to_run:
                success = False
                steps = 0
                score = 0.0
                rewards: List[float] = []
                task_error: Optional[Exception] = None
                log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
                try:
                    report = await run_single_task(env=env, client=client, task_id=task)
                    success = bool(report["success"])
                    steps = int(report["steps"])
                    score = _strict_score(float(report["score"]))
                    rewards = list(report["rewards"])
                except Exception as exc:
                    task_error = exc
                    score = 0.0
                finally:
                    log_end(success=success, steps=steps, score=score, rewards=rewards)
                if task_error is not None:
                    raise InferenceError(f"Task {task} failed.") from task_error
    finally:
        await client.close()


def main() -> None:
    asyncio.run(run_all_tasks())


if __name__ == "__main__":
    main()
