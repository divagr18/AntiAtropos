import asyncio
import json
import os
import random
import textwrap
import time
from contextlib import asynccontextmanager
from typing import Dict, List
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI

from AntiAtropos.client import AntiAtroposEnv
from AntiAtropos.grader import EpisodeGrader
from AntiAtropos.models import ActionType, SREAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY = (
    os.getenv("GROQ_API_KEY")      # prioritize Groq key since we default to groq API
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or os.getenv("HF_TOKEN")
)

ENV_URL = os.getenv("ANTIATROPOS_ENV_URL", "https://pranavkk-antiatropos.hf.space")
ENV_MODE = os.getenv("ANTIATROPOS_MODE", "simulated")
TASKS = ["task-1", "task-2", "task-3"]

TOTAL_BUDGET_SECONDS = 1080  # 18-minute limit
MIN_TASK_BUDGET_SECONDS = 60
MAX_STEPS_PER_TASK = 60       # 60 steps = ~5 minutes at this rate
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _task_seed(base_seed: int, task_id: str) -> int:
    offsets = {"task-1": 0, "task-2": 1, "task-3": 2}
    return int(base_seed + offsets.get(task_id, 0))


def _hf_web_fallback_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if host.endswith(".hf.space") and path == "":
        return base_url.rstrip("/") + "/web"
    return base_url


@asynccontextmanager
async def open_env_with_ws_fallback(base_url: str, message_timeout_s: int):
    try:
        async with AntiAtroposEnv(base_url, message_timeout_s=message_timeout_s) as env:
            yield env
            return
    except ConnectionError as e:
        fallback_url = _hf_web_fallback_url(base_url)
        if fallback_url == base_url or "404" not in str(e):
            raise
        print(f"[connect] ws 404 on {base_url}; retrying with {fallback_url}", flush=True)
        async with AntiAtroposEnv(fallback_url, message_timeout_s=message_timeout_s) as env:
            yield env


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
        return _parse_action(_extract_json_object(content))
    except Exception as e:
        print(f"[LLM_ERROR] task={task_id} step={step} error={type(e).__name__}: {e}", flush=True)
        return SREAction(action_type=ActionType.NO_OP, target_node_id="node-0", parameter=0.0)


def _compact_action(action: SREAction) -> str:
    payload = {
        "action_type": action.action_type.value,
        "target_node_id": action.target_node_id,
        "parameter": round(float(action.parameter), 4),
    }
    return json.dumps(payload, separators=(",", ":"))


async def run_single_task(env: AntiAtroposEnv, client: AsyncOpenAI, task_id: str, deadline: float) -> dict:
    start = time.monotonic()
    task_seed = _task_seed(SEED, task_id)
    result = await env.reset(task_id=task_id, mode=ENV_MODE, seed=task_seed)

    grader = EpisodeGrader(task_id=task_id)
    grader.record(result.observation)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    timed_out = False

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        if time.monotonic() >= deadline:
            timed_out = True
            break
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
        ack = getattr(result.observation, "action_ack_status", "")
        action_str = _compact_action(action)
        history.append(f"step={step} action={action_str} reward={reward:.4f} ack={ack or 'null'}")

        error = ack if ack.startswith(("Rejected:", "Error:")) else None
        print(
            f"[STEP] task={task_id} step={step} action={action_str} reward={reward:.4f} done={str(result.done).lower()} error={error or 'null'}",
            flush=True,
        )

    grade = grader.score()
    score = max(0.0, min(1.0, float(grade.composite)))
    elapsed = time.monotonic() - start
    success = score >= SUCCESS_SCORE_THRESHOLD and not timed_out
    print(
        f"[TASK_END] task={task_id} success={str(success).lower()} score={score:.4f} "
        f"steps={steps_taken} elapsed_s={elapsed:.1f} timed_out={str(timed_out).lower()} seed={task_seed}",
        flush=True,
    )
    return {
        "task_id": task_id,
        "success": success,
        "score": score,
        "steps": steps_taken,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "grade_summary": grade.summary(),
        "rewards": rewards,
    }


async def run_all_tasks() -> None:
    _seed_everything(SEED)
    tasks = [task for task in TASKS if task in {"task-1", "task-2", "task-3"}]
    if not tasks:
        raise RuntimeError("ANTIATROPOS_TASKS must include at least one of: task-1,task-2,task-3")
    if not API_KEY:
        raise RuntimeError("Missing API key (HF_TOKEN/OPENAI_API_KEY/API_KEY/GROQ_API_KEY).")

    print(
        f"[START] tasks={','.join(tasks)} env={ENV_URL} mode={ENV_MODE} model={MODEL_NAME} "
        f"budget_s={TOTAL_BUDGET_SECONDS} seed={SEED}",
        flush=True,
    )

    start = time.monotonic()
    deadline = start + TOTAL_BUDGET_SECONDS
    reports: List[dict] = []

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    try:
        async with open_env_with_ws_fallback(ENV_URL, MESSAGE_TIMEOUT_S) as env:
            for idx, task_id in enumerate(tasks):
                now = time.monotonic()
                if now >= deadline:
                    print(f"[BUDGET] stopping before {task_id}; time budget exhausted", flush=True)
                    break

                remaining_tasks = len(tasks) - idx
                remaining_seconds = max(0.0, deadline - now)
                allocated_seconds = max(
                    float(MIN_TASK_BUDGET_SECONDS),
                    remaining_seconds / float(remaining_tasks),
                )
                task_deadline = min(deadline, now + allocated_seconds)
                print(
                    f"[BUDGET] task={task_id} allocated_s={allocated_seconds:.1f} "
                    f"remaining_s={remaining_seconds:.1f} remaining_tasks={remaining_tasks}",
                    flush=True,
                )

                report = await run_single_task(
                    env=env,
                    client=client,
                    task_id=task_id,
                    deadline=task_deadline,
                )
                reports.append(report)
    finally:
        await client.close()

    total_elapsed = time.monotonic() - start
    completed_scores = [r["score"] for r in reports]
    aggregate_score = sum(completed_scores) / len(completed_scores) if completed_scores else 0.0
    aggregate_score = max(0.0, min(1.0, aggregate_score))
    all_success = len(reports) == len(tasks) and all(r["success"] for r in reports)

    for report in reports:
        print(f"[GRADE] {report['grade_summary']}", flush=True)

    print(
        f"[END] success={str(all_success).lower()} completed_tasks={len(reports)}/{len(tasks)} "
        f"aggregate_score={aggregate_score:.4f} elapsed_s={total_elapsed:.1f}",
        flush=True,
    )


def main() -> None:
    asyncio.run(run_all_tasks())


if __name__ == "__main__":
    main()
