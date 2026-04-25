#!/usr/bin/env python3
"""
Quick autonomous agent smoke test against the running AntiAtropos FastAPI server.

This does NOT require an LLM API key.
It uses a simple heuristic policy to validate end-to-end control-plane + telemetry wiring.
"""

import asyncio
import os
from dataclasses import dataclass

try:
    from AntiAtropos.client import AntiAtroposEnv
    from AntiAtropos.models import SREAction, ActionType
except ImportError:
    from client import AntiAtroposEnv  # type: ignore
    from models import SREAction, ActionType  # type: ignore


@dataclass
class Config:
    env_url: str = os.getenv("ENV_URL", "http://localhost:8000")
    task_id: str = os.getenv("ANTIATROPOS_TASK", "task-1")
    mode: str = os.getenv("ANTIATROPOS_MODE", os.getenv("ANTIATROPOS_ENV_MODE", "live"))
    max_steps: int = int(os.getenv("ANTIATROPOS_SMOKE_STEPS", "20"))


def pick_action(obs) -> SREAction:
    # Pick node with highest queue depth as target
    target = max(obs.nodes, key=lambda n: float(getattr(n, "queue_depth", 0.0)))

    avg_latency = float(getattr(obs, "average_latency_ms", 0.0))
    backlog = float(getattr(obs, "total_queue_backlog", 0.0))

    # Heuristic policy:
    # - If stressed, scale up busiest node
    # - If very calm, scale down non-VIP node
    # - Otherwise no-op
    if avg_latency > 0.20 or backlog > 0.45:
        return SREAction(action_type=ActionType.SCALE_UP, target_node_id=target.node_id, parameter=0.6)

    non_vips = [n for n in obs.nodes if not bool(getattr(n, "is_vip", False))]
    if avg_latency < 0.08 and backlog < 0.15 and non_vips:
        down_target = max(non_vips, key=lambda n: float(getattr(n, "capacity", 0.0)))
        return SREAction(action_type=ActionType.SCALE_DOWN, target_node_id=down_target.node_id, parameter=0.4)

    return SREAction(action_type=ActionType.NO_OP, target_node_id=target.node_id, parameter=0.0)


async def main() -> None:
    cfg = Config()
    print(f"[agent-smoke] env={cfg.env_url} task={cfg.task_id} mode={cfg.mode} steps={cfg.max_steps}")

    async with AntiAtroposEnv(cfg.env_url, message_timeout_s=120) as env:
        result = await env.reset(task_id=cfg.task_id, mode=cfg.mode)
        print(f"[reset] step={result.observation.step} latency={result.observation.average_latency_ms:.3f} backlog={result.observation.total_queue_backlog:.3f}")

        rewards = []
        for i in range(1, cfg.max_steps + 1):
            action = pick_action(result.observation)
            result = await env.step(action)
            rewards.append(float(result.reward or 0.0))
            ack = getattr(result.observation, "action_ack_status", "")
            print(
                f"[step {i:02d}] {action.action_type.value} {action.target_node_id} p={action.parameter:.2f} "
                f"reward={float(result.reward or 0.0):.3f} done={bool(result.done)} ack={ack}"
            )
            if result.done:
                break

        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            print(f"[done] steps={len(rewards)} avg_reward={avg_reward:.3f} final_latency={result.observation.average_latency_ms:.3f} final_backlog={result.observation.total_queue_backlog:.3f}")
        else:
            print("[done] no steps executed")


if __name__ == "__main__":
    asyncio.run(main())
