import asyncio
import hashlib
import os
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

from dotenv import load_dotenv

from AntiAtropos.client import AntiAtroposEnv
from AntiAtropos.grader import EpisodeGrader
from AntiAtropos.models import ActionType, SREAction


load_dotenv()

ENV_URL = os.getenv("ANTIATROPOS_ENV_URL", "https://pranavkk-antiatropos.hf.space")
ENV_MODE = os.getenv("ANTIATROPOS_MODE", "simulated")
TASK_ID = os.getenv("ANTIATROPOS_TASK", "task-3")
SEED = int(os.getenv("ANTIATROPOS_SEED", "42"))
MAX_STEPS = int(os.getenv("ANTIATROPOS_MAX_STEPS", "30"))


@dataclass(frozen=True)
class EpisodeResult:
    transcript: str
    score: float


ACTION_CYCLE: List[SREAction] = [
    SREAction(action_type=ActionType.NO_OP, target_node_id="node-0", parameter=0.0),
    SREAction(action_type=ActionType.SCALE_UP, target_node_id="node-0", parameter=1.0),
    SREAction(action_type=ActionType.REROUTE_TRAFFIC, target_node_id="node-1", parameter=0.5),
]


def hf_web_fallback_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if host.endswith(".hf.space") and path == "":
        return base_url.rstrip("/") + "/web"
    return base_url


def compact_action(action: SREAction) -> str:
    return (
        f'{{"action_type":"{action.action_type.value}",'
        f'"target_node_id":"{action.target_node_id}",'
        f'"parameter":{float(action.parameter):.2f}}}'
    )


def compact_observation(obs) -> str:
    return (
        f"step={obs.step} "
        f"lat={obs.average_latency_ms:.4f} "
        f"backlog={obs.total_queue_backlog:.4f} "
        f"reward={obs.reward:.4f} "
        f"done={str(obs.done).lower()}"
    )


async def run_one_episode(env: AntiAtroposEnv, seed: int) -> EpisodeResult:
    result = await env.reset(task_id=TASK_ID, mode=ENV_MODE, seed=seed)
    grader = EpisodeGrader(task_id=TASK_ID)
    grader.record(result.observation)

    lines: List[str] = []
    lines.append(f"[START] task={TASK_ID} seed={seed} mode={ENV_MODE}")
    lines.append(f"[OBS] {compact_observation(result.observation)}")

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        action = ACTION_CYCLE[(step - 1) % len(ACTION_CYCLE)]
        result = await env.step(action)
        grader.record(result.observation)
        lines.append(f"[STEP] step={step} action={compact_action(action)} {compact_observation(result.observation)}")

    grade = grader.score()
    score = max(0.0, min(1.0, float(grade.composite)))
    lines.append(f"[END] score={score:.6f} steps={result.observation.step}")
    transcript = "\n".join(lines)
    return EpisodeResult(transcript=transcript, score=score)


def transcript_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def main() -> None:
    print(f"Connecting to {ENV_URL}...")
    try:
        async with AntiAtroposEnv(base_url=ENV_URL) as env:
            first = await run_one_episode(env, SEED)
            second = await run_one_episode(env, SEED)
    except ConnectionError as e:
        fallback_url = hf_web_fallback_url(ENV_URL)
        if fallback_url == ENV_URL or "404" not in str(e):
            raise
        print(f"Falling back to {fallback_url}...")
        async with AntiAtroposEnv(base_url=fallback_url) as env:
            first = await run_one_episode(env, SEED)
            second = await run_one_episode(env, SEED)

    print("\n=== Run 1 ===")
    print(first.transcript)
    print(f"hash={transcript_hash(first.transcript)}")

    print("\n=== Run 2 ===")
    print(second.transcript)
    print(f"hash={transcript_hash(second.transcript)}")

    same = first.transcript == second.transcript
    print("\n=== Comparison ===")
    if same:
        print("✅ The two runs matched exactly.")
    else:
        print("❌ The two runs differed.")
        print("This usually means the environment is not fully deterministic for the chosen mode,")
        print("or the selected task still has randomness outside the seed path.")


if __name__ == "__main__":
    asyncio.run(main())
