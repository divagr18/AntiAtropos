import argparse
import asyncio
import inspect
import json
import math
import os
import random
import textwrap
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from AntiAtropos.client import AntiAtroposEnv
    from AntiAtropos.grader import EpisodeGrader
    from AntiAtropos.models import ActionType, SREAction
    from AntiAtropos.replay import EpisodeReplayBuffer, compress_trajectory
except ModuleNotFoundError:
    from client import AntiAtroposEnv  # type: ignore
    from grader import EpisodeGrader  # type: ignore
    from models import ActionType, SREAction  # type: ignore
    from replay import EpisodeReplayBuffer, compress_trajectory  # type: ignore

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or (
    "https://api.groq.com/openai/v1" if GROQ_API_KEY else "https://router.huggingface.co/v1"
)
MODEL_NAME = os.getenv("MODEL_NAME") or (
    "llama-3.1-8b-instant" if GROQ_API_KEY else "Qwen/Qwen2.5-72B-Instruct"
)
API_KEY = os.getenv("API_KEY") or GROQ_API_KEY
if not API_KEY:
    # Local fallback to keep developer runs convenient.
    API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

DEFAULT_ENV_URL = "https://keshav051-antiatropos.hf.space"
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
EVAL_RUNS = int(os.getenv("ANTIATROPOS_EVAL_RUNS", "3"))  # Num eval runs per task
TEMPERATURE_SWEEP = [0.6, 0.3, 0.7]  # Fixed temperatures for multi-episode eval

# Leaderboard weight: task-1 gets more weight (fundamentals)
LEADERBOARD_WEIGHTS = {"task-1": 0.4, "task-2": 0.3, "task-3": 0.3}

TASK_BRIEFS: Dict[str, str] = {
    "task-1": "Traffic ramps linearly every tick. Scale up proactively — new capacity takes 5 ticks to boot. Keep latency under SLA (200ms) while minimizing cost. Scale down when queues are safe.",
    "task-2": "One node (node-1 through node-4) will fail permanently. Wait until you SEE a FAILED node — do NOT pre-scale. Once a node shows status=FAILED: reroute traffic FROM the failed node to healthy peers, and scale up any starved children. Do NOT scale node-0 unless node-4 failed independently. SCALE_DOWN cancels pending boots and reduces cost. If reward is falling, stop scaling.",
    "task-3": "A surge (~75 req/tick) will hit node-1 and node-2 via a side channel bypassing node-0. Do NOT scale node-0 — it is NOT affected. ONLY scale node-1 or node-2 when their queue_depth rises. Do NOT pre-scale. 3-4 SCALE_UPs on each is sufficient. SCALE_DOWN cancels pending boots and reduces cost — use it when queues are safe. If reward is falling, STOP scaling and SCALE_DOWN to recover.",
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous SRE controller managing a five-node microservice cluster.

    CRITICAL: NO-THINK mode (/no_think). DO NOT output ` response` or ` ➤` tags. DO NOT use reply or reasoning blocks. Output ONLY your action directly as plain text. NO markdown formatting.

    CLUSTER TOPOLOGY (traffic flows parent → children):
      node-0 → node-1, node-2
      node-2 → node-3
      node-4 (independent ingress)
    FAILED nodes have outflow=0 — their children are starved.
    Backpressure: overloaded children reduce parent capacity.

    ACTIONS (new capacity takes 5 ticks to boot):
      SCALE_UP <node> <amount>   — add capacity (0.3-0.5 normal, 0.6-0.8 heavy surge), clears DEGRADED
      SCALE_DOWN <node> <amount>  — cancel pending boots first, then remove active capacity (0.2-0.4 safe, 0.5-0.7 aggressive)
      REROUTE_TRAFFIC <node> <fraction> — reduce THIS node capacity, redistribute to peers (0.3-0.5)
      SHED_LOAD <node> <fraction>  — drop incoming traffic (0.3-0.5), NEVER on node-0 (payment gateway)
      NO_OP                           — do nothing

    REWARD PRIORITIES (in order):
      1. Avoid SLA violations (latency > 200ms or error rate > 5%)
      2. Keep queues low (growing queues = destabilizing system)
      3. Don't over-provision (excess capacity costs money)

    REWARD SIGNAL: Each step returns a reward [0,1].
      > 0.5 = good. 0.15–0.5 = acceptable. < 0.15 = you are making things worse.
      If reward is falling, STOP the current strategy — try a different action or NO_OP.
      Repeating the same action when reward < 0.1 is always wrong.

    Scale when your observations demand it, not preemptively.
    Boot delay is 5 ticks — factor this into your timing.
    Scale back down when safe to save cost.

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


def build_user_prompt(task_id: str, step: int, obs: dict, history: List[str], demo_text: str = "") -> str:
    recent = "\n".join(history[-4:]) if history else "None"
    brief = TASK_BRIEFS.get(task_id, "Maintain SLA, stability, and efficient cost.")
    demo_section = f"\n\n{demo_text}" if demo_text else ""

    # Synthesize a 1-line cluster summary from the most important signals
    cost_hour = obs.get("current_cost_per_hour", 0.0)
    cost_dev = "low" if cost_hour < 1.2 else ("high" if cost_hour > 1.8 else "baseline")
    queue_backlog = obs.get("total_queue_backlog", 0.0)
    queue_trend = "rising" if queue_backlog > 0.3 else ("stable" if queue_backlog < 0.1 else "moderate")
    sla_violations = obs.get("sla_violations", 0)
    sla_note = f" ({sla_violations} violations)" if sla_violations > 0 else ""
    # Reward feedback from recent history
    recent_rewards = [float(h.split("reward=")[1].split()[0]) for h in history[-3:] if "reward=" in h]
    if recent_rewards:
        last_r = recent_rewards[-1]
        r_tag = "GOOD" if last_r > 0.5 else ("OK" if last_r > 0.2 else ("BAD" if last_r > 0.05 else "STOP-SCALING"))
        trend = "↓" if len(recent_rewards) > 1 and recent_rewards[-1] < recent_rewards[0] else ("↑" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else "")
        reward_feedback = f" | Reward: {last_r:.2f}={r_tag}{trend}"
    else:
        reward_feedback = ""
    cluster_summary = f"Cost: {cost_dev} (${cost_hour:.2f}/hr) | Queues: {queue_trend}{sla_note}{reward_feedback}"

    return textwrap.dedent(
        f"""
        Task: {task_id}
        Objective: {brief}
        Step: {step}
        Status: {cluster_summary}

        Current state:
        {json.dumps(obs, separators=(",", ":"))}

        Recent decisions:
        {recent}{demo_section}

        Choose the next SRE action.
        """
    ).strip()


def observation_for_model(obs) -> dict:
    """
    Build a compact observation dict for the LLM.

    DESIGN: only raw physical metrics a human SRE sees on their dashboard.
    Reward decomposition and pre-digested scoring signals are EXCLUDED —
    the LLM must reason from physics, not reverse-engineer the scorer.

    The scalar reward for past steps is already in the history (correct).
    """
    # Derive summary lists from per-node status fields (feature engineering,
    # same principle as total_queue_backlog — the agent could compute these
    # from raw per-node data but having them pre-computed speeds up reasoning).
    failed_nodes = []
    degraded_nodes = []
    for node in obs.nodes:
        s = str(getattr(node.status, "value", str(node.status)))
        if s == "failed":
            failed_nodes.append(node.node_id)
        elif s == "degraded":
            degraded_nodes.append(node.node_id)

    return {
        "task_id": obs.task_id,
        "mode": getattr(obs.mode, "value", str(obs.mode)),
        "step": obs.step,
        "max_steps": obs.max_steps,
        "failed_nodes": failed_nodes,
        "degraded_nodes": degraded_nodes,
        "average_latency_ms": obs.average_latency_ms,
        "error_rate": obs.error_rate,
        "total_queue_backlog": obs.total_queue_backlog,
        "current_cost_per_hour": getattr(obs, "current_cost_per_hour", 0.0),
        "sla_violations": obs.sla_violations,
        "invalid_action_count": obs.invalid_action_count,
        "nodes": [
            {
                "node_id": node.node_id,
                "status": getattr(node.status, "value", str(node.status)),
                "queue_depth": node.queue_depth,
                "latency_ms": node.latency_ms,
                "incoming_request_rate": node.incoming_request_rate,
                "cpu_utilization": node.cpu_utilization,
                "capacity": getattr(node, "capacity", 0.0),
                "pending_capacity": getattr(node, "pending_capacity", 0.0),
                "queue_delta": getattr(node, "queue_delta", 0.0),
                "outflow_rate": getattr(node, "outflow_rate", 0.0),
                "downstream_nodes": getattr(node, "downstream_nodes", []),
                "upstream_nodes": getattr(node, "upstream_nodes", []),
                "upstream_pressure": getattr(node, "upstream_pressure", 0.0),
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
    target_node_id = str(payload.get("target_node_id") or "node-0")
    parameter = float(payload.get("parameter") or 0.0)
    return SREAction(
        action_type=ActionType(action_type),
        target_node_id=target_node_id,
        parameter=parameter,
    )


def _action_had_effect(ack_status: str) -> bool:
    """Determine if an action had measurable effect from the ack status."""
    if not ack_status:
        return True
    if ack_status.startswith("Rejected:") or ack_status.startswith("Error:"):
        return False
    return True


async def get_model_action(client: AsyncOpenAI, task_id: str, step: int, obs: dict, history: List[str], demo_text: str = "") -> SREAction:
    prompt = build_user_prompt(task_id=task_id, step=step, obs=obs, history=history, demo_text=demo_text)
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            presence_penalty=0.3,
            response_format={"type": "json_object"},
            timeout=MODEL_TIMEOUT_S,
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


async def run_single_task(
    env: AntiAtroposEnv,
    client: AsyncOpenAI,
    task_id: str,
    temperature: float = 0.0,
    replay_buffer: Optional[EpisodeReplayBuffer] = None,
    run_seed: Optional[int] = None,
    is_baseline: bool = False,
) -> dict:
    task_seed = run_seed if run_seed is not None else _task_seed(SEED, task_id)
    result = await env.reset(task_id=task_id, mode=ENV_MODE, seed=task_seed)

    grader = EpisodeGrader(task_id=task_id)
    grader.record(result.observation)
    history: List[str] = []
    rewards: List[float] = []
    raw_steps: List[dict] = []  # For replay buffer compression
    steps_taken = 0
    # Baseline mode: no few-shot demonstrations
    demo_text = "" if is_baseline else (replay_buffer.format_demonstrations() if replay_buffer else "")

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        if result.done:
            break

        action = await get_model_action(
            client=client,
            task_id=task_id,
            step=step,
            obs=observation_for_model(result.observation),
            history=history,
            demo_text=demo_text,
        )
        result = await env.step(action)
        grader.record(result.observation)

        reward = float(result.reward or 0.0)
        rewards.append(reward)
        steps_taken = step
        action_str = _compact_action(action)

        # Determine if action had effect from ack_status
        ack_status = getattr(result.observation, "action_ack_status", "")
        had_effect = _action_had_effect(ack_status)

        # Record action for efficiency analysis
        grader.record_action(
            action_type=action.action_type.value,
            target_node_id=action.target_node_id,
            parameter=float(action.parameter),
            had_effect=had_effect,
        )

        history.append(f"step={step} action={action_str} reward={reward:.2f}")

        # Collect raw step data for replay compression
        obs = result.observation
        raw_steps.append({
            "step": step,
            "action_type": action.action_type.value,
            "target_node_id": action.target_node_id,
            "parameter": float(action.parameter),
            "reward": reward,
            "avg_latency_norm": getattr(obs, "average_latency_ms", 0.0),
            "error_rate": getattr(obs, "error_rate", 0.0),
            "queue_backlog_norm": getattr(obs, "total_queue_backlog", 0.0),
            "sla_violation": reward < 0.3,
        })

        error = ack_status if ack_status.startswith("Error:") or ack_status.startswith("Rejected:") else None
        log_step(step=step, action=action_str, reward=reward, done=bool(result.done), error=error)

    grade = grader.score()
    score = _strict_score(float(grade.composite))
    success = score >= SUCCESS_SCORE_THRESHOLD

    # Store in replay buffer if available and NOT baseline mode
    if not is_baseline and replay_buffer is not None and raw_steps:
        trajectory = compress_trajectory(
            steps=raw_steps,
            task_id=task_id,
            score=score,
            total_steps=steps_taken,
            final_sla_violations=int(grade.scores.get("violations", 0)),
            final_invalid_actions=int(grade.scores.get("invalid_actions", 0)),
        )
        replay_buffer.store(trajectory, score)

    # Serialize grade scores (convert non-serializable types)
    grade_scores = {}
    for k, v in grade.scores.items():
        if isinstance(v, float) and math.isnan(v):
            grade_scores[k] = None  # NaN → null in JSON
        else:
            grade_scores[k] = v

    return {
        "task_id": task_id,
        "success": success,
        "score": score,
        "steps": steps_taken,
        "rewards": rewards,
        "grade_scores": grade_scores,
        "composite": float(grade.composite),
    }


def _compute_aggregates(scores: List[float]) -> Dict[str, float]:
    """Compute aggregate metrics from a list of per-run composite scores."""
    if not scores:
        return {"mean": 0.0, "std": 0.0, "worst_case": 0.0, "pass_rate": 0.0, "consistency": 0.0}
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    worst_case = min(scores)
    pass_rate = sum(1 for s in scores if s >= SUCCESS_SCORE_THRESHOLD) / len(scores)
    consistency = max(0.0, 1.0 - std / mean) if mean > 0 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "worst_case": round(worst_case, 4),
        "pass_rate": round(pass_rate, 4),
        "consistency": round(consistency, 4),
    }


def _compute_leaderboard(task_composites: Dict[str, float]) -> float:
    """Weighted leaderboard score across tasks."""
    total = 0.0
    for task_id, weight in LEADERBOARD_WEIGHTS.items():
        total += weight * task_composites.get(task_id, 0.0)
    return round(total, 4)


def _write_eval_report(
    output_dir: Path,
    run_mode: str,
    task_results: Dict[str, dict],
    leaderboard_score: float,
) -> Path:
    """Write structured JSON eval report to disk and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{run_mode}_{timestamp}.json"
    filepath = output_dir / filename

    report = {
        "run_type": run_mode,
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "seed_base": SEED,
        "leaderboard_score": leaderboard_score,
        "tasks": task_results,
    }

    # Check for a previous run of the opposite mode to compute delta
    opposite_mode = "baseline" if run_mode == "trained" else "trained"
    existing = sorted(output_dir.glob(f"{opposite_mode}_*.json"))
    if existing:
        with open(existing[-1], "r") as f:
            other = json.load(f)
        delta = {}
        for tid in task_results:
            other_mean = other.get("tasks", {}).get(tid, {}).get("aggregates", {}).get("mean", 0.0)
            this_mean = task_results[tid].get("aggregates", {}).get("mean", 0.0)
            delta[tid] = round(this_mean - other_mean, 4)
        report["delta_from_" + opposite_mode] = delta

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return filepath


async def run_all_tasks(overrides: Optional[argparse.Namespace] = None) -> None:
    _seed_everything(SEED)
    all_tasks = ["task-1", "task-2", "task-3"]
    if overrides and overrides.task != "all":
        tasks_to_run = [overrides.task]
    else:
        run_single = os.getenv("ANTIATROPOS_RUN_SINGLE_TASK", "false").lower() == "true"
        task_id = TASK_NAME if TASK_NAME in set(all_tasks) else "task-1"
        tasks_to_run = [task_id] if run_single else all_tasks
    if not API_KEY:
        raise RuntimeError("Missing API key (API_KEY/HF_TOKEN/OPENAI_API_KEY).")

    # Mode: trained (with replay buffer) or baseline (no demos)
    is_baseline = getattr(overrides, "mode", "trained") == "baseline"
    eval_runs_val = getattr(overrides, "eval_runs", None)
    eval_runs = eval_runs_val if eval_runs_val is not None else EVAL_RUNS
    output_dir = Path(getattr(overrides, "output_dir", "eval_results"))

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    # Baseline mode: fresh empty replay buffer, never populated
    replay_buffer = EpisodeReplayBuffer()

    # Accumulate results for JSON report
    task_results: Dict[str, dict] = {}

    try:
        async with open_env(MESSAGE_TIMEOUT_S) as env:
            for task in tasks_to_run:
                task_scores: List[float] = []
                task_composites: List[float] = []
                task_successes: List[bool] = []
                episode_breakdown: List[dict] = []

                for run_idx in range(eval_runs):
                    # Fixed seed per (task, run_idx) so runs are reproducible
                    # and comparable across temperature conditions.
                    run_seed = SEED * 1000 + hash(task) % 100 + run_idx
                    temperature = TEMPERATURE_SWEEP[run_idx % len(TEMPERATURE_SWEEP)]

                    success = False
                    steps = 0
                    score = 0.0
                    composite = 0.0
                    rewards: List[float] = []
                    grade_scores: dict = {}
                    task_error: Optional[Exception] = None
                    log_start(task=f"{task} run={run_idx+1}/{eval_runs} temp={temperature} mode={'baseline' if is_baseline else 'trained'}", env=BENCHMARK, model=MODEL_NAME)
                    try:
                        report = await run_single_task(
                            env=env,
                            client=client,
                            task_id=task,
                            temperature=temperature,
                            replay_buffer=replay_buffer,
                            run_seed=run_seed,
                            is_baseline=is_baseline,
                        )
                        success = bool(report["success"])
                        steps = int(report["steps"])
                        score = _strict_score(float(report["score"]))
                        composite = float(report.get("composite", score))
                        rewards = list(report["rewards"])
                        grade_scores = report.get("grade_scores", {})
                        task_scores.append(score)
                        task_composites.append(composite)
                        task_successes.append(success)
                        episode_breakdown.append({
                            "run": run_idx + 1,
                            "temp": temperature,
                            "composite": round(composite, 4),
                            "score": round(score, 4),
                            "steps": steps,
                            "success": success,
                            **{k: v for k, v in grade_scores.items() if k not in ("action_distribution", "node_heatmap")},
                        })
                    except Exception as exc:
                        task_error = exc
                        score = 0.0
                    finally:
                        log_end(success=success, steps=steps, score=score, rewards=rewards)
                    if task_error is not None:
                        raise InferenceError(f"Task {task} run {run_idx+1} failed.") from task_error

                # Compute aggregate stats
                aggregates = _compute_aggregates(task_composites)
                task_results[task] = {
                    "aggregates": aggregates,
                    "episodes": episode_breakdown,
                }

                print(
                    f"[AGGREGATE] task={task} mean={aggregates['mean']:.3f} "
                    f"std={aggregates['std']:.3f} worst={aggregates['worst_case']:.3f} "
                    f"pass_rate={aggregates['pass_rate']:.2f} consistency={aggregates['consistency']:.3f} "
                    f"runs={len(task_composites)}",
                    flush=True,
                )

    finally:
        await client.close()

    # Compute leaderboard score
    task_means = {tid: task_results[tid]["aggregates"]["mean"] for tid in task_results}
    leaderboard = _compute_leaderboard(task_means)
    print(f"\n[LEADERBOARD] score={leaderboard:.3f}", flush=True)
    for tid, mean in task_means.items():
        weight = LEADERBOARD_WEIGHTS.get(tid, 0.0)
        print(f"  {tid}: mean={mean:.3f} (weight={weight})", flush=True)

    # Write JSON report
    run_mode = "baseline" if is_baseline else "trained"
    filepath = _write_eval_report(output_dir, run_mode, task_results, leaderboard)
    print(f"\n[REPORT] written to {filepath}", flush=True)

    # Check for comparison with opposite mode
    opposite = "baseline" if is_baseline else "trained"
    existing = sorted(output_dir.glob(f"{opposite}_*.json"))
    if existing:
        with open(existing[-1], "r") as f:
            other = json.load(f)
        print(f"\n[DELTA] vs {opposite} ({existing[-1].name}):", flush=True)
        for tid in task_results:
            other_mean = other.get("tasks", {}).get(tid, {}).get("aggregates", {}).get("mean", 0.0)
            this_mean = task_results[tid]["aggregates"]["mean"]
            delta = this_mean - other_mean
            sign = "+" if delta >= 0 else ""
            print(f"  {tid}: {sign}{delta:.3f} ({opposite}={other_mean:.3f} → {run_mode}={this_mean:.3f})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AntiAtropos SRE inference")
    parser.add_argument(
        "--task", "-t",
        choices=["task-1", "task-2", "task-3", "all"],
        default="all",
        help="Run a specific task or all tasks (default: all)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["trained", "baseline"],
        default="trained",
        help="Evaluation mode: 'trained' uses replay buffer demos, 'baseline' runs without (default: trained)",
    )
    parser.add_argument(
        "--eval-runs", "-n",
        type=int,
        default=None,
        help=f"Number of evaluation runs per task (default: {EVAL_RUNS})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="eval_results",
        help="Directory for JSON eval reports (default: eval_results)",
    )
    args = parser.parse_args()
    asyncio.run(run_all_tasks(overrides=args))


if __name__ == "__main__":
    main()
