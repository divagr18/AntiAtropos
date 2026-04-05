import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import AsyncOpenAI

from AntiAtropos.client import AntiAtroposEnv
from AntiAtropos.grader import EpisodeGrader
from AntiAtropos.models import ActionType, SREAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-70b-versatile")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or os.getenv("GROQ_API_KEY")
)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ANTIATROPOS_ENV_URL", "http://127.0.0.1:8000")
TASK_NAME = os.getenv("ANTIATROPOS_TASK", "task-3")
BENCHMARK = os.getenv("ANTIATROPOS_BENCHMARK", "antiatropos")
ENV_MODE = os.getenv("ANTIATROPOS_MODE", "simulated")
MAX_STEPS = int(os.getenv("ANTIATROPOS_MAX_STEPS", "35"))
TEMPERATURE = float(os.getenv("ANTIATROPOS_TEMPERATURE", "0.05"))
MAX_TOKENS = int(os.getenv("ANTIATROPOS_MAX_TOKENS", "180"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("ANTIATROPOS_SUCCESS_THRESHOLD", "0.55"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous SRE controller managing a five-node microservice cluster.

    Objectives:
    - minimize Lyapunov energy and queue growth
    - keep normalized average latency at or below 0.20
    - avoid invalid actions, especially SHED_LOAD on node-0, node-1, and node-2
    - scale proactively because SCALE_UP takes 5 ticks to take effect
    - protect the VIP gateway node-0

    Output exactly one JSON object:
    {
      "action_type": "SCALE_UP" | "SCALE_DOWN" | "REROUTE_TRAFFIC" | "SHED_LOAD" | "NO_OP",
      "target_node_id": "node-0" | "node-1" | "node-2" | "node-3" | "node-4",
      "parameter": 0.0
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current state:
        {json.dumps(obs, separators=(",", ":"))}

        Recent decisions:
        {history_block}

        Choose the next SRE action.
        """
    ).strip()


def compact_action(action: SREAction) -> str:
    payload = {
        "action_type": action.action_type.value,
        "target_node_id": action.target_node_id,
        "parameter": round(float(action.parameter), 4),
    }
    return json.dumps(payload, separators=(",", ":"))


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model response")

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(stripped[start : end + 1])


def parse_action(payload: dict) -> SREAction:
    action_type = str(payload.get("action_type", "NO_OP")).upper()
    target_node_id = str(payload.get("target_node_id", "node-0"))
    parameter = float(payload.get("parameter", 0.0))
    return SREAction(
        action_type=ActionType(action_type),
        target_node_id=target_node_id,
        parameter=parameter,
    )


async def get_model_action(
    client: AsyncOpenAI,
    step: int,
    obs: dict,
    history: List[str],
) -> SREAction:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or ""
        return parse_action(extract_json_object(content))
    except Exception:
        return SREAction(
            action_type=ActionType.NO_OP,
            target_node_id="node-0",
            parameter=0.0,
        )


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


async def run_episode() -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    env = None
    client = None

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        if not API_KEY:
            raise RuntimeError("missing API key")

        client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        if LOCAL_IMAGE_NAME:
            env = await AntiAtroposEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = AntiAtroposEnv(base_url=ENV_URL)
            await env.__aenter__()

        grader = EpisodeGrader(task_id=TASK_NAME)
        history: List[str] = []

        result = await env.reset(task_id=TASK_NAME, mode=ENV_MODE)
        grader.record(result.observation)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            action = await get_model_action(
                client=client,
                step=step,
                obs=observation_for_model(obs),
                history=history,
            )
            result = await env.step(action)
            grader.record(result.observation)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step

            ack_status = getattr(result.observation, "action_ack_status", "")
            error = ack_status if ack_status.startswith(("Rejected:", "Error:")) else None
            action_str = compact_action(action)
            log_step(step=step, action=action_str, reward=reward, done=result.done, error=error)

            history.append(
                f"step={step} action={action_str} reward={reward:.2f} ack={ack_status or 'null'}"
            )

            if result.done:
                break

        score = max(0.0, min(1.0, grader.score().composite))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        success = False
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    asyncio.run(run_episode())


if __name__ == "__main__":
    main()
