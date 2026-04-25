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
    """Diverse heuristic: uses all 5 action types based on cluster state."""
    avg_latency = float(getattr(obs, "average_latency_ms", 0.0))
    backlog = float(getattr(obs, "total_queue_backlog", 0.0))
    error_rate = float(getattr(obs, "error_rate", 0.0))

    # Find failed and degraded nodes
    failed = [n for n in obs.nodes if str(getattr(n, "status", "")) == "FAILED"]
    degraded = [n for n in obs.nodes if str(getattr(n, "status", "")) == "DEGRADED"]
    healthy = [n for n in obs.nodes if str(getattr(n, "status", "")) not in ("FAILED", "FAILED")]
    non_vips = [n for n in healthy if not bool(getattr(n, "is_vip", False))]
    critical_nodes = {"node-0", "node-1", "node-2"}

    # 1. FAILED node → REROUTE its traffic to peers
    if failed:
        fn = failed[0]
        return SREAction(action_type=ActionType.REROUTE_TRAFFIC,
                         target_node_id=fn.node_id, parameter=0.7)

    # 2. Non-critical overloaded → SHED_LOAD
    shedding_candidates = [n for n in non_vips
                           if float(getattr(n, "queue_depth", 0.0)) > 0.5
                           and n.node_id not in critical_nodes]
    if shedding_candidates and (avg_latency > 0.15 or backlog > 0.3):
        target = max(shedding_candidates, key=lambda n: float(getattr(n, "queue_depth", 0.0)))
        return SREAction(action_type=ActionType.SHED_LOAD,
                         target_node_id=target.node_id, parameter=0.4)

    # 3. High stress → SCALE_UP busiest node (prefer downstream)
    if avg_latency > 0.20 or backlog > 0.45 or degraded:
        candidates = degraded if degraded else healthy
        downstream = [n for n in candidates if n.node_id != "node-0"]
        target = max(downstream or candidates,
                     key=lambda n: float(getattr(n, "queue_depth", 0.0)))
        param = 0.6 if float(getattr(target, "queue_depth", 0.0)) > 0.7 else 0.4
        return SREAction(action_type=ActionType.SCALE_UP,
                         target_node_id=target.node_id, parameter=param)

    # 4. Very calm → SCALE_DOWN overprovisioned non-VIP
    if avg_latency < 0.08 and backlog < 0.15 and non_vips:
        overprov = [n for n in non_vips if float(getattr(n, "capacity", 0.0)) > 0.7]
        if overprov:
            down_target = max(overprov, key=lambda n: float(getattr(n, "capacity", 0.0)))
            return SREAction(action_type=ActionType.SCALE_DOWN,
                             target_node_id=down_target.node_id, parameter=0.3)

    # 5. Default: NO_OP
    fallback = healthy[0] if healthy else obs.nodes[0]
    return SREAction(action_type=ActionType.NO_OP,
                     target_node_id=fallback.node_id, parameter=0.0)


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
