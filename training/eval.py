"""
eval.py — Evaluate base vs fine-tuned model on the OpenEnv.

Runs episodes with:
  1. The fine-tuned model (current LoRA adapter)
  2. The heuristic baseline

Compares average rewards across tasks. Results are saved locally and pushed
via push_run_files_to_hub (in train.py) under hub_model_repo/<run_id>/eval_results.json.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import torch

try:
    from .model_utils import push_to_hub
    from .openenv_loop import (
        OpenEnvClient,
        rollout_episode,
        rollout_heuristic_episode,
    )
except ImportError:
    from model_utils import push_to_hub
    from openenv_loop import (
        OpenEnvClient,
        rollout_episode,
        rollout_heuristic_episode,
    )


def evaluate(
    client: OpenEnvClient,
    model,
    tokenizer,
    cfg: Dict[str, Any],
    output_dir: str = "/tmp/antiatropos_eval",
) -> Dict[str, Any]:
    """Run evaluation: fine-tuned model vs heuristic baseline.

    Returns a dict with per-task results and overall comparison.
    """
    tasks = cfg.get("tasks", ["task-1", "task-2", "task-3"])
    eval_episodes = cfg.get("eval_episodes", 3)
    eval_max_steps = cfg.get("eval_max_steps", 60)

    # Enable inference mode
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except ImportError:
        model.eval()

    results: Dict[str, Any] = {}
    all_ft_rewards: List[float] = []
    all_heur_rewards: List[float] = []

    print(f"\n{'='*70}")
    print(f"EVALUATION — {eval_episodes} episodes per task, {eval_max_steps} steps")
    print(f"{'='*70}")

    for task_id in tasks:
        ft_rewards: List[float] = []
        heur_rewards: List[float] = []
        ft_invalid = 0

        for ep in range(eval_episodes):
            seed = 1000 + ep  # Deterministic eval seeds

            # Fine-tuned model episode
            ft_ep = rollout_episode(
                client, model, tokenizer, task_id,
                eval_max_steps, cfg, seed=seed,
            )
            ft_rewards.append(ft_ep.avg_reward)
            ft_invalid += ft_ep.num_invalid

            # Heuristic baseline episode
            heur_ep = rollout_heuristic_episode(
                client, task_id, eval_max_steps, seed=seed,
            )
            heur_rewards.append(heur_ep.avg_reward)

        ft_avg = sum(ft_rewards) / len(ft_rewards)
        heur_avg = sum(heur_rewards) / len(heur_rewards)
        all_ft_rewards.extend(ft_rewards)
        all_heur_rewards.extend(heur_rewards)

        winner = "FT WINS" if ft_avg >= heur_avg else "HEURISTIC WINS"
        results[task_id] = {
            "ft_avg_reward": ft_avg,
            "heuristic_avg_reward": heur_avg,
            "ft_wins": ft_avg >= heur_avg,
            "ft_invalid_actions": ft_invalid,
        }

        print(f"\n  {task_id}:")
        print(f"    FT model avg reward:    {ft_avg:.4f}")
        print(f"    Heuristic avg reward:   {heur_avg:.4f}")
        print(f"    Result: {winner}")
        print(f"    Invalid actions (FT): {ft_invalid}")

    # Overall summary
    tasks_won = sum(1 for r in results.values() if r["ft_wins"])
    ft_overall = sum(all_ft_rewards) / len(all_ft_rewards) if all_ft_rewards else 0
    heur_overall = sum(all_heur_rewards) / len(all_heur_rewards) if all_heur_rewards else 0

    summary = {
        "per_task": results,
        "overall_ft_avg": ft_overall,
        "overall_heuristic_avg": heur_overall,
        "tasks_won_by_ft": tasks_won,
        "total_tasks": len(tasks),
        "ft_overall_wins": ft_overall >= heur_overall,
    }

    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  FT model overall avg:    {ft_overall:.4f}")
    print(f"  Heuristic overall avg:   {heur_overall:.4f}")
    print(f"  FT wins on: {tasks_won}/{len(tasks)} tasks")
    print(f"  Overall: {'FT WINS' if ft_overall >= heur_overall else 'HEURISTIC WINS'}")

    # Save eval results
    import os
    os.makedirs(output_dir, exist_ok=True)
    eval_path = f"{output_dir}/eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [eval] Saved results → {eval_path}")

    return summary



