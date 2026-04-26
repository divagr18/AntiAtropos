#!/usr/bin/env python3
"""
train.py — AntiAtropos QLoRA Reward-Based Training (HF Jobs Edition)
=====================================================================

Training loop:  generate → evaluate → reward → update → log → checkpoint

This is NOT supervised fine-tuning. The model generates actions, the OpenEnv
environment (running on HF Spaces) evaluates them, and we use the reward
signal to update the policy via REINFORCE/GRPO/RLOO.

Architecture (from training.md):
  - GPU = compute only (ephemeral)
  - Hub = source of truth (persistent)
  - Training = reproducible + resumable
  - Metrics = structured + queryable

Usage:
  # On HF Jobs (A10G recommended):
  python training/train.py

  # Local with custom config:
  HF_SPACE_URL=http://localhost:8000 python training/train.py --config my_config.yaml

  # Override specific values via CLI:
  python training/train.py --num-iterations 100 --loss-type grpo
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

# ────────────────────────────────────────────────────────────
# Module path setup — allow imports from training/ package
# ────────────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model_utils import (
    attach_lora,
    detect_gpu_tier,
    find_latest_checkpoint,
    gpu_scaled_config,
    load_base_model,
    push_adapter_to_hub,
    push_to_hub,
    save_checkpoint,
)
from openenv_loop import (
    OpenEnvClient,
    rollout_batch,
    rollout_episode,
    rollout_heuristic_episode,
)
from eval import evaluate, push_eval_results
from plotting import (
    generate_all_plots,
    push_plots_to_hub,
    episodes_to_plot_data,
)


# ────────────────────────────────────────────────────────────
# Config Loading
# ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from YAML, apply env var overrides, GPU auto-scale."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply env var overrides (ANTIATROPOS_<KEY>=<value>)
    env_overrides = {}
    for key, value in os.environ.items():
        if key.startswith("ANTIATROPOS_"):
            cfg_key = key[len("ANTIATROPOS_"):].lower()
            env_overrides[cfg_key] = value

    for key, value in env_overrides.items():
        if key in cfg:
            orig = cfg[key]
            if isinstance(orig, bool):
                cfg[key] = value.lower() in ("true", "1", "yes")
            elif isinstance(orig, int):
                cfg[key] = int(value)
            elif isinstance(orig, float):
                cfg[key] = float(value)
            elif isinstance(orig, list):
                cfg[key] = json.loads(value)
            else:
                cfg[key] = value
            print(f"[config] Env override: {key} = {cfg[key]}")

    # GPU auto-scaling (only if not explicitly overridden)
    cfg = gpu_scaled_config(cfg)

    return cfg


# ────────────────────────────────────────────────────────────
# REINFORCE Loss (PyTorch)
# ────────────────────────────────────────────────────────────

def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted returns from a list of rewards."""
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return returns


def reinforce_baseline_loss_fn(
    model,
    tokenizer,
    episodes: List,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Compute REINFORCE with baseline loss across episodes.

    Uses per-mini-batch gradient accumulation:
      - Pre-compute ALL advantages on CPU first (enables global normalization).
      - For each mini-batch: forward → compute loss → backward() immediately.
      - Frees the computation graph after every mini-batch.
      - Returns a detached scalar; gradients already sit in model.parameters().grad.

    This keeps peak VRAM to ONE forward pass worth of activations (~8-9 GiB)
    instead of accumulating all mini-batch graphs simultaneously (which caused
    OOM when 3 batches × ~8.9 GiB each = 26+ GiB were held concurrently).

    Caller (train.py) must check `if loss.requires_grad` before calling
    loss.backward() — this function returns requires_grad=False so the
    caller's backward() is skipped cleanly.
    """
    import math as _math
    gamma = cfg.get("reward_gamma", 0.99)
    normalize_adv = cfg.get("advantage_normalize", True)
    loss_batch_size = cfg.get("loss_batch_size", 1)
    max_seq_len_cap = cfg.get("max_seq_length", 512)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ── Phase 1: Collect all (transition, return) pairs (CPU) ──────────────
    all_pairs: List[Tuple] = []
    for ep in episodes:
        if not ep.transitions:
            continue
        rewards = [t.reward for t in ep.transitions]
        returns = compute_returns(rewards, gamma)
        for trans, ret in zip(ep.transitions, returns):
            if trans.input_ids is not None:
                all_pairs.append((trans, ret))

    if not all_pairs:
        return torch.tensor(0.0, device=model.device)

    # ── Phase 2: Compute normalized advantages on CPU (global normalization) ─
    raw_returns = torch.tensor([p[1] for p in all_pairs], dtype=torch.float32)
    if normalize_adv and len(raw_returns) > 1:
        advantages = (raw_returns - raw_returns.mean()) / (raw_returns.std() + 1e-8)
    else:
        advantages = raw_returns
    # advantages stays on CPU until we move per-batch slices to GPU

    # ── Phase 3: Gradient accumulation — one forward/backward per mini-batch ─
    # Each iteration: build batch → forward → loss → backward → del graph.
    # Only one forward pass worth of activations lives in VRAM at any time.
    n_batches = _math.ceil(len(all_pairs) / loss_batch_size)
    total_loss_val = 0.0

    for batch_idx, batch_start in enumerate(range(0, len(all_pairs), loss_batch_size)):
        batch = all_pairs[batch_start:batch_start + loss_batch_size]
        batch_advs = advantages[batch_start:batch_start + loss_batch_size]  # CPU tensor

        batch_ids = [p[0].input_ids for p in batch]
        batch_masks = [p[0].attention_mask for p in batch]

        # Truncate outlier-length sequences (tail keeps action tokens)
        batch_ids  = [ids[-max_seq_len_cap:]  if ids.shape[0]  > max_seq_len_cap else ids  for ids  in batch_ids]
        batch_masks = [m[-max_seq_len_cap:]   if m.shape[0]    > max_seq_len_cap else m    for m    in batch_masks]

        # Left-pad to same length within mini-batch
        max_len = max(ids.shape[0] for ids in batch_ids)
        padded_ids, padded_masks = [], []
        for ids, mask in zip(batch_ids, batch_masks):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids.append(torch.cat([torch.full((pad_len,), pad_id, device=ids.device), ids]))
                padded_masks.append(torch.cat([torch.zeros(pad_len, device=mask.device, dtype=mask.dtype), mask]))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)

        input_ids     = torch.stack(padded_ids)
        attention_mask = torch.stack(padded_masks)

        # ── Forward pass ─────────────────────────────────────────────────────
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            free, total_mem = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            print(f"  [loss_fwd b{batch_idx+1}/{n_batches}] "
                  f"shape={input_ids.shape} alloc={alloc:.2f}GiB "
                  f"free={free/1024**3:.1f}/{total_mem/1024**3:.1f}GiB", flush=True)

        outputs = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            use_cache=False,  # No KV-cache needed for single-pass loss forward.
        )

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            free2, _ = torch.cuda.mem_get_info()
            print(f"  [loss_fwd b{batch_idx+1}/{n_batches}] "
                  f"post-fwd peak={peak:.2f}GiB free={free2/1024**3:.1f}GiB", flush=True)

        # ── Memory-efficient log-prob via fused cross_entropy kernel ──────────
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels  = input_ids[:, 1:].contiguous()
        B, S_minus1 = shift_labels.shape

        token_nll = torch.nn.functional.cross_entropy(
            shift_logits.view(B * S_minus1, -1).to(model.device),
            shift_labels.view(B * S_minus1).to(model.device),
            reduction="none",
            ignore_index=-100,
        ).view(B, S_minus1)
        token_log_probs = -token_nll

        shift_mask = attention_mask[:, 1:].to(model.device)
        token_log_probs = token_log_probs * shift_mask
        seq_log_probs = token_log_probs.sum(dim=1)  # (B,)

        # Free logits immediately — cross_entropy output is all we need
        del outputs, logits, shift_logits, token_nll, token_log_probs
        torch.cuda.empty_cache()

        # ── Per-mini-batch loss scaled for correct gradient averaging ─────────
        # Divide by n_batches so that summing gradients across all mini-batches
        # equals the mean over the full batch (standard gradient accumulation).
        batch_advs_gpu = batch_advs.to(model.device)
        batch_loss = -(batch_advs_gpu * seq_log_probs).mean() / n_batches

        # ── Backward immediately — frees entire computation graph ─────────────
        # Gradients accumulate in model.parameters().grad across mini-batches.
        # The caller in train.py checks `if loss.requires_grad` and skips
        # its backward() call because we return a detached tensor.
        batch_loss.backward()

        total_loss_val += batch_loss.item() * n_batches  # unscale for logging
        del batch_loss, seq_log_probs, batch_advs_gpu
        torch.cuda.empty_cache()

    # Return detached scalar for logging (requires_grad=False → caller skips backward)
    return torch.tensor(total_loss_val / n_batches, device=model.device)




def grpo_loss_fn(
    model,
    tokenizer,
    episodes: List,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Compute GRPO (Group Relative Policy Optimization) loss.

    GRPO uses a group of K rollouts from the same initial state,
    then computes advantages relative to the group mean. This eliminates
    the need for a learned value function.

    Uses batched forward passes for GPU efficiency.
    """
    gamma = cfg.get("reward_gamma", 0.99)
    k = cfg.get("grpo_k", 4)
    loss_batch_size = cfg.get("loss_batch_size", 32)

    # Group episodes by (task_id, initial_seed)
    groups: Dict[tuple, List] = {}
    for ep in episodes:
        key = (ep.task_id, id(ep) % 1000)  # Approximate grouping
        groups.setdefault(key, []).append(ep)

    all_pairs: List[Tuple] = []  # (transition, advantage)

    for key, group in groups.items():
        # Compute group-level returns
        group_returns = []
        for ep in group:
            rewards = [t.reward for t in ep.transitions]
            returns = compute_returns(rewards, gamma)
            ep_return = returns[0] if returns else 0.0
            group_returns.append(ep_return)

        # Group advantage = return - group_mean
        group_mean = sum(group_returns) / len(group_returns)
        group_std = (sum((r - group_mean)**2 for r in group_returns)
                     / len(group_returns)) ** 0.5 + 1e-8

        for ep, ep_return in zip(group, group_returns):
            advantage = (ep_return - group_mean) / group_std
            for trans in ep.transitions:
                if trans.input_ids is not None:
                    all_pairs.append((trans, advantage))

    if not all_pairs:
        return torch.tensor(0.0, requires_grad=True, device=model.device)

    all_log_probs = []
    all_advantages = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Process transitions in batches
    for batch_start in range(0, len(all_pairs), loss_batch_size):
        batch = all_pairs[batch_start:batch_start + loss_batch_size]
        batch_ids = [p[0].input_ids for p in batch]
        batch_masks = [p[0].attention_mask for p in batch]
        batch_advs = [p[1] for p in batch]

        max_len = max(ids.shape[0] for ids in batch_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(batch_ids, batch_masks):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids.append(torch.cat([
                    torch.full((pad_len,), pad_id, device=ids.device), ids
                ]))
                padded_masks.append(torch.cat([
                    torch.zeros(pad_len, device=mask.device, dtype=mask.dtype), mask
                ]))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)

        input_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(padded_masks)

        # Forward pass WITH gradient
        outputs = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        )
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs_all.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        shift_mask = attention_mask[:, 1:]
        token_log_probs = token_log_probs * shift_mask
        seq_log_probs = token_log_probs.sum(dim=1)

        all_log_probs.append(seq_log_probs)
        all_advantages.extend(batch_advs)

    if not all_log_probs:
        return torch.tensor(0.0, requires_grad=True, device=model.device)

    log_probs = torch.cat(all_log_probs)
    advantages = torch.tensor(all_advantages, device=model.device).detach()
    loss = -(advantages * log_probs).mean()
    return loss


# ────────────────────────────────────────────────────────────
# Metrics Push
# ────────────────────────────────────────────────────────────

def push_train_metrics(
    metrics: Dict[str, Any],
    hub_dataset: str,
) -> None:
    """Push training metrics row to the Hub dataset."""
    if not hub_dataset:
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        jsonl_path = os.path.join(tmp_dir, "metrics.jsonl")

        try:
            api.hf_hub_download(
                repo_id=hub_dataset,
                filename="metrics.jsonl",
                repo_type="dataset",
                local_dir=tmp_dir,
            )
        except Exception:
            pass

        with open(jsonl_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        api.upload_file(
            path_or_fileobj=jsonl_path,
            path_in_repo="metrics.jsonl",
            repo_id=hub_dataset,
            repo_type="dataset",
            commit_message=f"train metrics — {metrics.get('run_id')} iter {metrics.get('iteration')}",
        )
    except Exception as e:
        print(f"[train] Metrics push failed: {e}")


# ────────────────────────────────────────────────────────────
# Main Training Loop
# ────────────────────────────────────────────────────────────

def _log_vram(where: str) -> None:
    """Print CUDA memory usage at key diagnostic points."""
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  [VRAM @{where}]  "
          f"alloc={alloc:6.2f}GiB  reserved={reserved:6.2f}GiB  "
          f"peak={peak:6.2f}GiB  free={free/1024**3:.1f}/{total/1024**3:.1f}GiB",
          flush=True)


def train(cfg: Dict[str, Any]) -> None:
    """Main training loop."""

    # ---- Reproducibility ----
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_id = cfg.get("run_id", "exp_001")
    output_dir = Path(cfg.get("output_dir", "/tmp/antiatropos_train"))
    output_dir.mkdir(parents=True, exist_ok=True)

    hub_model_repo = cfg.get("hub_model_repo", "")
    hub_metrics_dataset = cfg.get("hub_metrics_dataset", "")
    push_to_hub_flag = cfg.get("push_to_hub", True)

    # ---- Verify environment ----
    env_url = cfg.get("env_url", "https://pranavkk-antiatropos.hf.space")
    client = OpenEnvClient(env_url)
    if not client.verify():
        print("[train] FATAL: Cannot reach environment. Aborting.")
        sys.exit(1)

    # ---- Load model ----
    print("\n[train] Loading model...")
    model, tokenizer = load_base_model(cfg)
    _log_vram("model_loaded")

    # ---- Check for resume ----
    start_iteration = 0
    ckpt_path = find_latest_checkpoint(hub_model_repo) if hub_model_repo else None
    if ckpt_path:
        local_ckpt = download_checkpoint(hub_model_repo, ckpt_path)
        model = load_from_checkpoint(model, tokenizer, local_ckpt)
        try:
            start_iteration = int(ckpt_path.split("-")[1])
        except (ValueError, IndexError):
            start_iteration = 0
        print(f"[train] Resuming from iteration {start_iteration}")
    else:
        model = attach_lora(model, cfg, seed=seed)
        # Unsloth's attach_lora already enables gradient checkpointing via
        # use_gradient_checkpointing="unsloth" in get_peft_model().
        # Do NOT call gradient_checkpointing_enable() again — it conflicts
        # with Unsloth's custom implementation and can increase VRAM usage.

    # ---- Optimizer ----
    lr = cfg.get("learning_rate", 2e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    optim_name = cfg.get("optim", "adamw_8bit")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ---- Loss function ----
    loss_type = cfg.get("loss_type", "reinforce_baseline")
    loss_fns = {
        "reinforce_baseline": reinforce_baseline_loss_fn,
        "grpo": grpo_loss_fn,
    }
    loss_fn = loss_fns.get(loss_type, reinforce_baseline_loss_fn)
    print(f"[train] Loss function: {loss_type}")

    # ---- Config ----
    num_iterations = cfg.get("num_iterations", 500)
    num_episodes = cfg.get("num_episodes_per_iteration", 4)
    max_steps = cfg.get("max_steps_per_episode", 60)
    tasks = cfg.get("tasks", ["task-1", "task-2", "task-3"])
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    checkpoint_interval = cfg.get("checkpoint_interval", 25)
    eval_interval = cfg.get("eval_interval", 50)
    push_interval = cfg.get("push_interval", 10)
    plot_interval = cfg.get("plot_interval", 25)

    # ---- Training loop ----
    print(f"\n{'='*70}")
    print(f"ANTIATROPOS QLORA TRAINING")
    print(f"{'='*70}")
    print(f"  Run ID:        {run_id}")
    print(f"  Loss type:     {loss_type}")
    print(f"  Iterations:    {num_iterations}")
    print(f"  Episodes/iter: {num_episodes}")
    print(f"  Tasks:         {tasks}")
    print(f"  Max steps:     {max_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Hub model:     {hub_model_repo or '(not configured)'}")
    print(f"  Hub metrics:   {hub_metrics_dataset or '(not configured)'}")
    print(f"  Output dir:    {output_dir}")
    print(f"{'='*70}\n")

    # Keep model in eval mode during rollout to minimise VRAM pressure.
    # for_training() is called only right before the loss forward pass.
    model.eval()
    _log_vram("eval_after_attach")

    metrics_buffer: List[Dict] = []
    eval_metrics_history: List[Dict] = []
    recent_episodes_data: List[Dict] = []  # For plotting action distributions

    for iteration in range(start_iteration, num_iterations):
        iter_start = time.time()

        # ---- Collect rollouts (parallel batch) ----
        task_ids = [tasks[ep_idx % len(tasks)] for ep_idx in range(num_episodes)]
        seeds = [seed + iteration * 1000 + ep_idx for ep_idx in range(num_episodes)]

        _log_vram(f"i{iteration}_pre_rollout")

        try:
            use_parallel = cfg.get("parallel_episodes", True)
            if use_parallel and num_episodes > 1:
                episodes = rollout_batch(
                    env_url, model, tokenizer, task_ids,
                    max_steps, cfg, seeds,
                )
            else:
                # Fallback: sequential rollout (for debugging)
                episodes = []
                for ep_idx in range(num_episodes):
                    task_id = tasks[ep_idx % len(tasks)]
                    seed_ep = seed + iteration * 1000 + ep_idx
                    ep = rollout_episode(
                        client, model, tokenizer, task_id,
                        max_steps, cfg, seed=seed_ep,
                    )
                    episodes.append(ep)
        except Exception as e:
            print(f"  [iter {iteration}] Batch rollout failed: {e}")
            continue

        # ---- Clear VRAM before loss (generation KV-cache on GPU) ----
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        # Move rollout tensors to CPU — loss will move them back in batches
        for ep in episodes:
            for t in ep.transitions:
                if t.input_ids is not None:
                    t.input_ids = t.input_ids.cpu()
                if t.attention_mask is not None:
                    t.attention_mask = t.attention_mask.cpu()
        _log_vram(f"i{iteration}_after_offload")

        # ---- Compute loss (standard train mode — base 4-bit stays frozen, only LoRA needs gradients) ----
        model.train()
        _log_vram(f"i{iteration}_after_train")
        loss = loss_fn(model, tokenizer, episodes, cfg)

        # ---- Backward + update ----
        if loss.requires_grad:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
        else:
            grad_norm = 0.0

        # Clear training intermediates and return to eval for next rollout
        torch.cuda.empty_cache()
        model.eval()
        _log_vram(f"i{iteration}_post_grad")

        # ---- Compute iteration metrics ----
        avg_reward = sum(ep.avg_reward for ep in episodes) / len(episodes)
        total_invalid = sum(ep.num_invalid for ep in episodes)
        iter_time = time.time() - iter_start

        print(f"  [iter {iteration:4d}] loss={loss.item():.4f}  "
              f"avg_reward={avg_reward:.4f}  "
              f"invalid={total_invalid}  "
              f"grad_norm={grad_norm:.4f}  "
              f"time={iter_time:.1f}s")

        # ---- Buffer metrics ----
        metrics_row = {
            "run_id": run_id,
            "iteration": iteration,
            "type": "train",
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "invalid_actions": total_invalid,
            "grad_norm": grad_norm if isinstance(grad_norm, float)
                         else grad_norm.item() if torch.is_tensor(grad_norm)
                         else 0.0,
            "num_episodes": len(episodes),
            "iter_time_s": iter_time,
        }
        metrics_buffer.append(metrics_row)

        # Store episode data for plotting (keep recent window)
        ep_data = episodes_to_plot_data(episodes)
        recent_episodes_data.extend(ep_data)
        if len(recent_episodes_data) > 200:  # Keep last ~200 episodes
            recent_episodes_data = recent_episodes_data[-200:]

        # ---- Push metrics ----
        if (iteration + 1) % push_interval == 0 and hub_metrics_dataset:
            for row in metrics_buffer:
                push_train_metrics(row, hub_metrics_dataset)
            metrics_buffer.clear()

        # ---- Checkpoint ----
        if (iteration + 1) % checkpoint_interval == 0:
            ckpt_dir = save_checkpoint(
                model, tokenizer, str(output_dir), iteration
            )
            if push_to_hub_flag and hub_model_repo:
                push_to_hub(ckpt_dir, hub_model_repo,
                           f"checkpoint-{iteration}")

        # ---- Evaluation ----
        if (iteration + 1) % eval_interval == 0:
            eval_results = evaluate(
                client, model, tokenizer, cfg,
                output_dir=str(output_dir / "eval"),
            )
            eval_row = {
                "run_id": run_id,
                "step": iteration,
                "type": "eval",
            }
            # Flatten eval results for plotting
            for k, v in eval_results.items():
                if not isinstance(v, dict):
                    eval_row[f"eval_{k}"] = v
            for tid, tv in eval_results.get("per_task", {}).items():
                for mk, mv in tv.items():
                    eval_row[f"eval_{tid}_{mk}"] = mv
            eval_metrics_history.append(eval_row)

            if hub_metrics_dataset:
                push_eval_results(
                    eval_results, hub_metrics_dataset, run_id, iteration
                )
            # Re-enable training mode
            model.train()

        # ---- Plotting ----
        if (iteration + 1) % plot_interval == 0:
            try:
                plot_paths = generate_all_plots(
                    train_metrics=metrics_buffer,
                    eval_metrics=eval_metrics_history,
                    episodes_data=recent_episodes_data,
                    output_dir=str(output_dir),
                    cfg=cfg,
                )
                if push_to_hub_flag and hub_model_repo:
                    push_plots_to_hub(plot_paths, hub_model_repo, iteration)
            except Exception as e:
                print(f"  [iter {iteration}] Plotting failed: {e}")

    # ────────────────────────────────────────────────────────
    # Final save + push
    # ────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")

    # Save final adapter
    final_dir = str(output_dir / "final_adapter")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train] Final adapter saved to {final_dir}")

    # Push to Hub
    if push_to_hub_flag and hub_model_repo:
        push_to_hub(final_dir, hub_model_repo,
                   f"AntiAtropos QLoRA final — {run_id}")

    # Flush remaining metrics
    if hub_metrics_dataset and metrics_buffer:
        for row in metrics_buffer:
            push_train_metrics(row, hub_metrics_dataset)
        metrics_buffer.clear()

    # Final evaluation
    final_eval = evaluate(
        client, model, tokenizer, cfg,
        output_dir=str(output_dir / "final_eval"),
    )

    if hub_metrics_dataset:
        push_eval_results(
            final_eval, hub_metrics_dataset, run_id, num_iterations
        )

    # Final plots (full training history)
    try:
        final_eval_row = {
            "run_id": run_id,
            "step": num_iterations,
            "type": "eval",
        }
        for k, v in final_eval.items():
            if not isinstance(v, dict):
                final_eval_row[f"eval_{k}"] = v
        for tid, tv in final_eval.get("per_task", {}).items():
            for mk, mv in tv.items():
                final_eval_row[f"eval_{tid}_{mk}"] = mv
        eval_metrics_history.append(final_eval_row)

        plot_paths = generate_all_plots(
            train_metrics=metrics_buffer,
            eval_metrics=eval_metrics_history,
            episodes_data=recent_episodes_data,
            output_dir=str(output_dir),
            cfg=cfg,
        )
        if push_to_hub_flag and hub_model_repo:
            push_plots_to_hub(plot_paths, hub_model_repo, num_iterations)
    except Exception as e:
        print(f"[train] Final plotting failed: {e}")

    print(f"\n[train] All done. Final adapter: {final_dir}")
    if hub_model_repo:
        print(f"[train] Hub repo: https://huggingface.co/{hub_model_repo}")


# ────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AntiAtropos QLoRA Training — HF Jobs Edition"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(TRAINING_DIR / "config.yaml"),
        help="Path to config.yaml (default: training/config.yaml)",
    )

    # ---- Quick overrides for smoke runs ----
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="Total training iterations (default: from config)")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Episodes per iteration (default: from config)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per episode (default: from config)")
    parser.add_argument("--loss-type", type=str, default=None,
                        choices=["reinforce_baseline", "grpo"],
                        help="Loss function type")
    parser.add_argument("--env-mode", type=str, default=None,
                        choices=["simulated", "hybrid", "live"],
                        help="Environment mode (default: from config)")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Evaluate every N iterations")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Checkpoint every N iterations")
    parser.add_argument("--plot-interval", type=int, default=None,
                        help="Generate plots every N iterations")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Unique run identifier")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Local output directory")
    parser.add_argument("--no-push", action="store_true",
                        help="Disable all Hub pushes (model + metrics + plots)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run: 10 iters, 2 episodes, 20 steps, "
                             "no push, eval/ckpt/plot every 5")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # ---- Smoke run preset ----
    if args.smoke:
        cfg["num_iterations"] = 10
        cfg["num_episodes_per_iteration"] = 2
        cfg["max_steps_per_episode"] = 40
        cfg["eval_interval"] = 5
        cfg["checkpoint_interval"] = 5
        cfg["plot_interval"] = 5
        cfg["push_to_hub"] = False
        cfg["eval_episodes"] = 1
        if not args.run_id:
            cfg["run_id"] = "smoke_test"
        if not args.output_dir:
            cfg["output_dir"] = "/tmp/antiatropos_smoke"
        print("[SMOKE MODE] 10 iters x 2 episodes x 40 steps — no Hub push")

    # ---- CLI overrides (explicit args beat smoke preset) ----
    if args.num_iterations is not None:
        cfg["num_iterations"] = args.num_iterations
    if args.num_episodes is not None:
        cfg["num_episodes_per_iteration"] = args.num_episodes
    if args.max_steps is not None:
        cfg["max_steps_per_episode"] = args.max_steps
    if args.loss_type is not None:
        cfg["loss_type"] = args.loss_type
    if args.env_mode is not None:
        cfg["env_mode"] = args.env_mode
    if args.eval_interval is not None:
        cfg["eval_interval"] = args.eval_interval
    if args.checkpoint_interval is not None:
        cfg["checkpoint_interval"] = args.checkpoint_interval
    if args.plot_interval is not None:
        cfg["plot_interval"] = args.plot_interval
    if args.run_id is not None:
        cfg["run_id"] = args.run_id
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.no_push:
        cfg["push_to_hub"] = False
        cfg["hub_model_repo"] = ""
        cfg["hub_metrics_dataset"] = ""

    # Allow HF_TOKEN from env
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    train(cfg)


if __name__ == "__main__":
    main()
