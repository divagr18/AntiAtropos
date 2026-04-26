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
from eval import evaluate
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

        # Build per-transition action masks: only compute log-probs over
        # the GENERATED action tokens, not the prompt tokens.
        # prompt_len is stored on each transition; after possible truncation
        # we need to recompute the mask offset.
        batch_action_masks = []
        for ids, p in zip(batch_ids, batch):
            plen = p[0].prompt_len  # original prompt length before truncation
            seq_len = ids.shape[0]
            # If sequence was truncated from the left, adjust prompt_len:
            # the kept portion starts at max(0, original_len - max_seq_len_cap)
            original_len = p[0].input_ids.shape[0] if not isinstance(p[0].input_ids, int) else seq_len
            if isinstance(p[0].input_ids, torch.Tensor) and p[0].input_ids.shape[0] > max_seq_len_cap:
                offset = p[0].input_ids.shape[0] - max_seq_len_cap
                plen = max(0, plen - offset)
            amask = torch.zeros(seq_len, dtype=torch.long)
            if plen < seq_len:
                amask[plen:] = 1  # action tokens after prompt
            batch_action_masks.append(amask)

        # Left-pad to same length within mini-batch
        max_len = max(ids.shape[0] for ids in batch_ids)
        padded_ids, padded_masks, padded_action_masks = [], [], []
        for ids, mask, amask in zip(batch_ids, batch_masks, batch_action_masks):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids.append(torch.cat([torch.full((pad_len,), pad_id, device=ids.device), ids]))
                padded_masks.append(torch.cat([torch.zeros(pad_len, device=mask.device, dtype=mask.dtype), mask]))
                # Action mask: left-pad with zeros (padding tokens are never action tokens)
                padded_action_masks.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), amask]))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)
                padded_action_masks.append(amask)

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

        # ── Memory-efficient NLL via fused cross_entropy ─────────────────────
        # F.cross_entropy(reduction='none') uses a single fused CUDA kernel:
        # it never materialises the full [B, S-1, V] log-prob matrix (~623 MiB
        # at V=151936, batch=1, seq=512) — it computes log-softmax + NLL in one
        # pass, keeping only the per-token scalar result.
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()   # (B, S-1, V)
        shift_labels  = input_ids[:, 1:].contiguous()   # (B, S-1)
        shift_labels  = shift_labels.to(model.device)
        shift_mask    = attention_mask[:, 1:].to(model.device)  # (B, S-1)

        # token_nll: (B, S-1), zero for padded positions
        token_nll = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.clamp(min=0).view(-1),
            reduction="none",
        ).view(shift_labels.shape)  # (B, S-1)
        token_nll = token_nll * shift_mask

        # ── Action-only log-probs for REINFORCE ──
        # Only sum NLL over action tokens (past prompt_len), not the prompt.
        # This is critical: log π(action | prompt) ≠ log π(prompt+action).
        # Masking out prompt tokens prevents the gradient from pushing on
        # tokens the model can't control and eliminates a massive source of
        # noise and variance in the REINFORCE gradient.
        stacked_action_masks = torch.stack(padded_action_masks)  # (B, S)
        shift_action_mask = stacked_action_masks[:, 1:].to(model.device)  # (B, S-1)
        # Zero out NLL for prompt positions — only keep action token NLL
        action_nll = token_nll * shift_action_mask
        seq_log_probs = -(action_nll.sum(dim=1))  # (B,) sum of action-token log-probs only
        # Number of action tokens per sequence (for optional normalization)
        n_action_tokens = shift_action_mask.sum(dim=1).clamp(min=1)  # (B,)

        # ── Chunked vocab entropy (avoids materialising full [B, S, V]) ────────
        # logsumexp over V gives the log-normaliser (1 scalar per token, ~4 MiB).
        # We then accumulate -sum(p*log_p) chunk-by-chunk: each chunk is CHUNK_V
        # columns → peak extra alloc ≈ B×S×CHUNK_V×4B = 1×511×4096×4 ≈ 8 MiB
        # instead of 623 MiB for the full V=151936 matrix.
        CHUNK_V = 4096
        # Exact single-pass entropy without materialising [B,S,V]:
        # logsumexp over V gives the normaliser; we then compute -sum(p*log_p) chunk-by-chunk.
        log_Z = shift_logits.logsumexp(dim=-1, keepdim=True)   # (B, S-1, 1)
        entropy_per_token = torch.zeros(shift_logits.shape[:2], device=model.device)
        for v_start in range(0, shift_logits.size(-1), CHUNK_V):
            chunk_logits = shift_logits[:, :, v_start:v_start + CHUNK_V]  # (B, S-1, c)
            log_p_chunk  = chunk_logits - log_Z                           # log-prob for this slice
            p_chunk      = log_p_chunk.exp()                              # prob for this slice
            entropy_per_token += -(p_chunk * log_p_chunk).sum(dim=-1)    # accumulate (B, S-1)
        del log_Z

        # Free logits immediately before backward
        del outputs, logits, shift_logits, token_nll
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            free2, _ = torch.cuda.mem_get_info()
            print(f"  [loss_fwd b{batch_idx+1}/{n_batches}] "
                  f"post-fwd peak={peak:.2f}GiB free={free2/1024**3:.1f}GiB", flush=True)

        # ── Per-mini-batch loss: REINFORCE + entropy bonus ────────────────────
        ent_coef = cfg.get("entropy_coef", 0.001)
        n_valid_tokens = shift_mask.sum(dim=1).clamp(min=1)  # (B,)
        # Only compute entropy over action tokens (same region as log-probs)
        n_action_valid = (shift_action_mask * shift_mask).sum(dim=1).clamp(min=1)  # (B,)
        avg_token_entropy = ((entropy_per_token * shift_action_mask * shift_mask).sum(dim=1) / n_action_valid).mean()

        print(f"    [entropy b{batch_idx+1}/{n_batches}] "
              f"avg_token_entropy={avg_token_entropy.item():.3f}nats  "
              f"ent_coef={ent_coef}  "
              f"reinforce={-(batch_advs.to(model.device) * seq_log_probs).mean().item():.4f}  "
              f"ent_bonus={ent_coef * avg_token_entropy.item():.4f}", flush=True)

        batch_advs_gpu = batch_advs.to(model.device)
        # Normalize log-probs by number of action tokens to prevent
        # length-dependent gradient scaling. Without this, sequences with
        # more action tokens get disproportionately large gradients.
        norm_seq_log_probs = seq_log_probs / n_action_tokens  # (B,)
        batch_loss = (
            -(batch_advs_gpu * norm_seq_log_probs).mean()  # REINFORCE (length-normalized)
            - ent_coef * avg_token_entropy             # per-token entropy bonus
        ) / n_batches

        # ── Backward immediately — frees entire computation graph ─────────────
        batch_loss.backward()

        total_loss_val += batch_loss.item() * n_batches
        del batch_loss, seq_log_probs, batch_advs_gpu, avg_token_entropy, entropy_per_token
        torch.cuda.empty_cache()

    # Return detached scalar for logging (requires_grad=False → caller skips backward)
    return torch.tensor(total_loss_val / n_batches, device=model.device)




def grpo_loss_fn(
    model,
    tokenizer,
    episodes: List,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """GRPO (Group Relative Policy Optimization) loss.

    Requires episodes to be structured as K groups of same-(task_id, seed) rollouts.
    Each group's advantages are normalised relative to that group's mean/std,
    eliminating the need for a value-function baseline.

    Uses the same OOM-safe per-mini-batch backward() as reinforce_baseline_loss_fn.
    """
    import math as _math
    gamma           = cfg.get("reward_gamma", 0.99)
    loss_batch_size = cfg.get("loss_batch_size", 1)
    max_seq_len_cap = cfg.get("max_seq_length", 512)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ── Phase 1: Group episodes by (task_id, seed), compute group advantages ─
    # Key: (task_id, seed) — both must match for same-state comparison.
    # Seed is stored on Episode.seed (set during rollout collection).
    groups: Dict[tuple, List] = {}
    for ep in episodes:
        key = (ep.task_id, ep.seed)  # EXACT grouping — not id() approximation
        groups.setdefault(key, []).append(ep)

    all_pairs: List[Tuple] = []  # (transition, advantage)

    for key, group in groups.items():
        if len(group) == 1:
            # Group of 1 — advantage=0, no gradient signal.
            # Happens if num_episodes is not a multiple of grpo_k.
            print(f"  [grpo] WARNING: group {key} has only 1 episode — "
                  f"num_episodes must be grpo_k × num_tasks", flush=True)

        # Episode-level returns (discounted sum from step 0)
        group_returns = []
        for ep in group:
            rewards  = [t.reward for t in ep.transitions]
            returns  = compute_returns(rewards, gamma)
            group_returns.append(returns[0] if returns else 0.0)

        group_mean = sum(group_returns) / len(group_returns)
        group_std  = (sum((r - group_mean) ** 2 for r in group_returns)
                      / max(len(group_returns) - 1, 1)) ** 0.5 + 1e-8  # Bessel-corrected

        for ep, ep_return in zip(group, group_returns):
            advantage = (ep_return - group_mean) / group_std
            for trans in ep.transitions:
                if trans.input_ids is not None:
                    all_pairs.append((trans, advantage))

    if not all_pairs:
        return torch.tensor(0.0, device=model.device)

    # ── Phase 2: Advantage tensor on CPU ────────────────────────────────────
    advantages = torch.tensor([p[1] for p in all_pairs], dtype=torch.float32)

    # ── Phase 3: OOM-safe mini-batch forward/backward ───────────────────────
    n_batches     = _math.ceil(len(all_pairs) / loss_batch_size)
    total_loss_val = 0.0
    ent_coef       = cfg.get("entropy_coef", 0.001)
    CHUNK_V        = 4096

    for batch_idx, batch_start in enumerate(range(0, len(all_pairs), loss_batch_size)):
        batch      = all_pairs[batch_start:batch_start + loss_batch_size]
        batch_advs = advantages[batch_start:batch_start + loss_batch_size]

        batch_ids   = [p[0].input_ids    for p in batch]
        batch_masks = [p[0].attention_mask for p in batch]

        # Truncate + left-pad
        batch_ids   = [ids[-max_seq_len_cap:] if ids.shape[0] > max_seq_len_cap else ids
                       for ids  in batch_ids]
        batch_masks = [m[-max_seq_len_cap:]   if m.shape[0]   > max_seq_len_cap else m
                       for m    in batch_masks]

        # Build action masks (same as reinforce_baseline_loss_fn)
        batch_action_masks = []
        for ids, p in zip(batch_ids, batch):
            plen = p[0].prompt_len
            seq_len = ids.shape[0]
            if isinstance(p[0].input_ids, torch.Tensor) and p[0].input_ids.shape[0] > max_seq_len_cap:
                offset = p[0].input_ids.shape[0] - max_seq_len_cap
                plen = max(0, plen - offset)
            amask = torch.zeros(seq_len, dtype=torch.long)
            if plen < seq_len:
                amask[plen:] = 1
            batch_action_masks.append(amask)
        max_len = max(ids.shape[0] for ids in batch_ids)
        padded_ids, padded_masks, padded_action_masks = [], [], []
        for ids, mask, amask in zip(batch_ids, batch_masks, batch_action_masks):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids.append(torch.cat(
                    [torch.full((pad_len,), pad_id, device=ids.device), ids]))
                padded_masks.append(torch.cat(
                    [torch.zeros(pad_len, device=mask.device, dtype=mask.dtype), mask]))
                padded_action_masks.append(torch.cat(
                    [torch.zeros(pad_len, dtype=torch.long), amask]))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)
                padded_action_masks.append(amask)

        input_ids      = torch.stack(padded_ids)
        attention_mask = torch.stack(padded_masks)

        torch.cuda.empty_cache()
        outputs = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            use_cache=False,
        )

        # Fused cross_entropy NLL (no [B,S,V] materialisation)
        logits       = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()   # (B, S-1, V)
        shift_labels = input_ids[:, 1:].contiguous().to(model.device)
        shift_mask_g = attention_mask[:, 1:].to(model.device)

        token_nll = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.clamp(min=0).view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        token_nll  = token_nll * shift_mask_g

        # ── Action-only log-probs for GRPO ──
        stacked_action_masks = torch.stack(padded_action_masks)  # (B, S)
        shift_action_mask = stacked_action_masks[:, 1:].to(model.device)  # (B, S-1)
        action_nll = token_nll * shift_action_mask
        seq_log_probs = -(action_nll.sum(dim=1))  # (B,)

        # Chunked entropy
        log_Z = shift_logits.logsumexp(dim=-1, keepdim=True)
        entropy_per_token = torch.zeros(shift_logits.shape[:2], device=model.device)
        for v_start in range(0, shift_logits.size(-1), CHUNK_V):
            chunk    = shift_logits[:, :, v_start:v_start + CHUNK_V] - log_Z
            p_chunk  = chunk.exp()
            entropy_per_token += -(p_chunk * chunk).sum(dim=-1)
        del log_Z, outputs, logits, shift_logits, token_nll
        torch.cuda.empty_cache()

        n_valid = (shift_action_mask * shift_mask_g).sum(dim=1).clamp(min=1)
        avg_entropy = ((entropy_per_token * shift_action_mask * shift_mask_g).sum(dim=1) / n_valid).mean()

        batch_advs_gpu = batch_advs.to(model.device)
        # Length-normalized log-probs (same as reinforce_baseline_loss_fn)
        n_action_tokens_grpo = shift_action_mask.sum(dim=1).clamp(min=1)  # (B,)
        norm_seq_log_probs = seq_log_probs / n_action_tokens_grpo
        batch_loss = (
            -(batch_advs_gpu * norm_seq_log_probs).mean()
            - ent_coef * avg_entropy
        ) / n_batches
        batch_loss.backward()

        total_loss_val += batch_loss.item() * n_batches
        del batch_loss, seq_log_probs, batch_advs_gpu, avg_entropy, entropy_per_token
        torch.cuda.empty_cache()

    return torch.tensor(total_loss_val / n_batches, device=model.device)



# ────────────────────────────────────────────────────────────
# Run Files Push (to hub_model_repo/<run_id>/)
# ────────────────────────────────────────────────────────────


def push_run_files_to_hub(
    run_id: str,
    output_dir: Path,
    hub_model_repo: str,
    iteration: int,
) -> None:
    """Upload step_metrics.jsonl, iter_metrics.jsonl, training.log, and eval results.

    Files are uploaded under <run_id>/ in the model repo alongside checkpoints.
    Called every checkpoint_interval iterations and at the end of training.
    """
    if not hub_model_repo:
        return

    files_to_push = [
        ("step_metrics.jsonl",  f"{run_id}/step_metrics.jsonl"),
        ("iter_metrics.jsonl",  f"{run_id}/iter_metrics.jsonl"),
        ("training.log",        f"{run_id}/training.log"),
        ("run_info.json",       f"{run_id}/run_info.json"),
    ]

    # Also push eval results if they exist
    eval_path = output_dir / "eval" / "eval_results.json"
    if eval_path.exists():
        files_to_push.append(("eval/eval_results.json", f"{run_id}/eval_results.json"))
    final_eval_path = output_dir / "final_eval" / "eval_results.json"
    if final_eval_path.exists():
        files_to_push.append(("final_eval/eval_results.json", f"{run_id}/final_eval_results.json"))

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        pushed = []
        for local_name, hub_path in files_to_push:
            local_path = output_dir / local_name
            if not local_path.exists():
                continue
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=hub_path,
                    repo_id=hub_model_repo,
                    repo_type="model",
                    commit_message=f"[{run_id}] iter {iteration}: {local_name}",
                )
                pushed.append(local_name)
            except Exception as e:
                print(f"  [push] Failed to push {local_name}: {e}")
        if pushed:
            print(f"  [push] \u2192 HF model {hub_model_repo}/{run_id}/: {', '.join(pushed)}",
                  flush=True)
    except Exception as e:
        print(f"[train] Hub file push failed: {e}")


# ────────────────────────────────────────────────────────────
# Local JSONL Metrics Writers
# ────────────────────────────────────────────────────────────

def write_step_metrics(
    run_id: str,
    iteration: int,
    episode_idx: int,
    task_id: str,
    step: int,
    transition,          # Transition dataclass from openenv_loop
    output_dir: Path,
) -> None:
    """Append one per-step row to step_metrics.jsonl.

    Each row captures the full cluster state at that step:
    action chosen, reward received, and all per-node + cluster-level
    metrics so you can graph queue depth, latency, cost, SLA violations,
    action distribution, etc. over time.
    """
    obs = transition.obs_dict or {}
    action = transition.action

    row: Dict[str, Any] = {
        # ── Identity ──────────────────────────────────────────────
        "run_id":        run_id,
        "iteration":     iteration,
        "episode_idx":   episode_idx,
        "task_id":       task_id,
        "step":          step,
        "ts":            __import__("datetime").datetime.utcnow().isoformat() + "Z",

        # ── Action ────────────────────────────────────────────────
        "action_type":   action.action_type,
        "target_node":   action.target_node_id,
        "parameter":     round(action.parameter, 4),
        "is_valid":      action.is_valid,

        # ── Reward ────────────────────────────────────────────────
        "reward":        round(transition.reward, 6),

        # ── Cluster-level metrics ──────────────────────────────────
        "avg_latency_ms":      round(obs.get("average_latency_ms", 0.0), 3),
        "error_rate":          round(obs.get("error_rate", 0.0), 6),
        "total_queue_backlog": round(obs.get("total_queue_backlog", 0.0), 4),
        "cost_per_hour":       round(obs.get("current_cost_per_hour", 0.0), 4),
        "sla_violations":      obs.get("sla_violations", 0),
    }

    # ── Per-node metrics (flat columns: n0_q, n0_l, n0_s, ...) ──
    for node in obs.get("nodes", []):
        nid = node.get("node_id", "")
        key = nid.replace("-", "")  # "node-0" → "node0"
        row[f"{key}_status"]   = node.get("status", "")[:1]   # H/D/F
        row[f"{key}_queue"]    = round(node.get("queue_depth", 0.0), 4)
        row[f"{key}_latency"]  = round(node.get("latency_ms", 0.0), 2)
        row[f"{key}_inflow"]   = round(node.get("incoming_request_rate", 0.0), 2)
        row[f"{key}_outflow"]  = round(node.get("outflow_rate", 0.0), 2)
        row[f"{key}_capacity"] = round(node.get("capacity", 0.0), 4)
        row[f"{key}_pending"]  = round(node.get("pending_capacity", 0.0), 4)

    path = output_dir / "step_metrics.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_iter_metrics(
    run_id: str,
    iteration: int,
    loss: float,
    avg_reward: float,
    grad_norm: float,
    total_invalid: int,
    num_episodes: int,
    iter_time_s: float,
    output_dir: Path,
) -> None:
    """Append one per-iteration row to iter_metrics.jsonl."""
    row = {
        "run_id":        run_id,
        "iteration":     iteration,
        "ts":            __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "loss":          round(loss, 6),
        "avg_reward":    round(avg_reward, 6),
        "grad_norm":     round(grad_norm, 4),
        "invalid_actions": total_invalid,
        "num_episodes":  num_episodes,
        "iter_time_s":   round(iter_time_s, 2),
    }
    path = output_dir / "iter_metrics.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")



class _TeeLogger:
    """Duplicates writes to both the original stream and a log file.

    Activated at the start of train() so that every print() — VRAM stats,
    step logs, entropy, iteration summaries, tracebacks — goes to both
    the HF job terminal stream AND a persistent training.log on disk.
    """
    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._file = open(log_path, "a", buffering=1, encoding="utf-8")  # line-buffered

    def write(self, data: str) -> None:
        self._stream.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def fileno(self) -> int:
        return self._stream.fileno()  # subprocess / os compatibility

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

    @property
    def original_stream(self):
        return self._stream


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

    # ── Run-specific output directory ───────────────────────────────────────
    # Structure: <base_output_dir>/<run_id>/
    #   checkpoint-0010/   ← saved every checkpoint_interval iters
    #   checkpoint-0020/
    #   ...
    #   final_adapter/     ← saved at end of training
    #   run_info.json      ← written at startup; identifies this run
    #
    # Using run_id as a subfolder means multiple runs never overwrite each other.
    base_output_dir = Path(cfg.get("output_dir", "/workspace/antiatropos_checkpoints"))
    output_dir = base_output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Activate full-run logging to disk ────────────────────────────────────
    # Every print() from here on is tee'd to training.log (line-buffered).
    # This mirrors the HF job terminal stream to a persistent file so you
    # can inspect the full log even after the job completes or crashes.
    log_path = output_dir / "training.log"
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = _TeeLogger(sys.stdout, log_path)
    sys.stderr = _TeeLogger(sys.stderr, log_path)
    print(f"[train] Full log: {log_path}")
    # Write run manifest so checkpoints are always identifiable
    import json as _json
    run_info = {
        "run_id": run_id,
        "started_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    run_info_path = output_dir / "run_info.json"
    run_info_path.write_text(_json.dumps(run_info, indent=2, default=str))
    print(f"[train] Run directory: {output_dir}")
    print(f"[train] Run manifest:  {run_info_path}")

    hub_model_repo = cfg.get("hub_model_repo", "")
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
    checkpoint_interval = cfg.get("checkpoint_interval", 10)  # default: every 10 iters
    eval_interval = cfg.get("eval_interval", 50)
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
        # GRPO requires K episodes per task from the SAME seed so grpo_loss_fn
        # can group by (task_id, seed) and compute within-group advantages.
        # REINFORCE uses unique seeds per episode for diversity.
        if loss_type == "grpo":
            k = cfg.get("grpo_k", 2)
            # Validate: num_episodes must be k * num_tasks
            expected = k * len(tasks)
            if num_episodes != expected:
                print(f"  [grpo] WARNING: num_episodes={num_episodes} ≠ "
                      f"grpo_k({k}) × num_tasks({len(tasks)})={expected}. "
                      f"Forcing to {expected}.", flush=True)
                num_episodes = expected
            # Each task gets k copies with the same per-task seed
            task_ids = [tasks[t] for t in range(len(tasks)) for _ in range(k)]
            task_seeds = [seed + iteration * 100 + t for t in range(len(tasks))]
            seeds    = [task_seeds[t] for t in range(len(tasks)) for _ in range(k)]
        else:
            task_ids = [tasks[ep_idx % len(tasks)] for ep_idx in range(num_episodes)]
            seeds    = [seed + iteration * 1000 + ep_idx for ep_idx in range(num_episodes)]

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

        # ---- Optimizer step (loss_fn already called .backward() per mini-batch) ----
        grad_norm = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()

        # Clear training intermediates and return to eval for next rollout
        torch.cuda.empty_cache()
        model.eval()
        _log_vram(f"i{iteration}_post_grad")

        # ---- Compute iteration metrics ----
        avg_reward = sum(ep.avg_reward for ep in episodes) / len(episodes)
        total_invalid = sum(ep.num_invalid for ep in episodes)
        iter_time = time.time() - iter_start

        # ---- Write per-step metrics (one row per episode step) ----
        # Done here (post-training) because training may mutate episode objects.
        for ep_idx, ep in enumerate(episodes):
            for step_idx, t in enumerate(ep.transitions):
                write_step_metrics(
                    run_id=run_id,
                    iteration=iteration,
                    episode_idx=ep_idx,
                    task_id=ep.task_id,
                    step=step_idx + 1,
                    transition=t,
                    output_dir=output_dir,
                )

        # ---- Write per-iteration metrics ----
        _grad_norm_val = (
            grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        )
        write_iter_metrics(
            run_id=run_id,
            iteration=iteration,
            loss=loss.item(),
            avg_reward=avg_reward,
            grad_norm=_grad_norm_val,
            total_invalid=total_invalid,
            num_episodes=len(episodes),
            iter_time_s=iter_time,
            output_dir=output_dir,
        )

        print(f"  [iter {iteration:4d}] loss={loss.item():.4f}  "
              f"avg_reward={avg_reward:.4f}  "
              f"invalid={total_invalid}  "
              f"grad_norm={_grad_norm_val:.4f}  "
              f"time={iter_time:.1f}s")

        # Store episode data for plotting (keep recent window)
        ep_data = episodes_to_plot_data(episodes)
        recent_episodes_data.extend(ep_data)
        if len(recent_episodes_data) > 200:  # Keep last ~200 episodes
            recent_episodes_data = recent_episodes_data[-200:]

        # ---- Checkpoint + push run files ----
        if (iteration + 1) % checkpoint_interval == 0:
            # Pad iteration number so ls sorts correctly: checkpoint-0010, checkpoint-0050, ...
            ckpt_name = f"checkpoint-{iteration + 1:04d}"
            ckpt_dir = output_dir / ckpt_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            # Write a small metadata file so you know exactly what's in each checkpoint
            ckpt_meta = {
                "run_id": run_id,
                "iteration": iteration + 1,
                "avg_reward": avg_reward,
                "loss": loss.item(),
                "saved_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            }
            (ckpt_dir / "checkpoint_meta.json").write_text(
                _json.dumps(ckpt_meta, indent=2)
            )
            print(f"  [ckpt] Saved \u2192 {ckpt_dir}  "
                  f"(reward={avg_reward:.4f}  loss={loss.item():.4f})", flush=True)
            if push_to_hub_flag and hub_model_repo:
                push_to_hub(
                    str(ckpt_dir),
                    hub_model_repo,
                    commit_message=f"[{run_id}] {ckpt_name}",
                    path_in_repo=f"{run_id}/{ckpt_name}",
                )
                # Push run files (metrics, logs) alongside checkpoint
                push_run_files_to_hub(run_id, output_dir, hub_model_repo, iteration + 1)

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
                    push_plots_to_hub(plot_paths, hub_model_repo, iteration, run_id=run_id)
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

    # Push to Hub — scoped under run_id/final_adapter/ so it never overwrites other runs
    if push_to_hub_flag and hub_model_repo:
        push_to_hub(
            final_dir,
            hub_model_repo,
            commit_message=f"[{run_id}] final_adapter",
            path_in_repo=f"{run_id}/final_adapter",
        )


    # Final evaluation
    final_eval = evaluate(
        client, model, tokenizer, cfg,
        output_dir=str(output_dir / "final_eval"),
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
            push_plots_to_hub(plot_paths, hub_model_repo, num_iterations, run_id=run_id)
    except Exception as e:
        print(f"[train] Final plotting failed: {e}")

    print(f"\n[train] All done. Final adapter: {final_dir}")
    if hub_model_repo:
        print(f"[train] Hub repo: https://huggingface.co/{hub_model_repo}")
    
    # \u2500\u2500 Final push of all run files
    if hub_model_repo:
        push_run_files_to_hub(run_id, output_dir, hub_model_repo, num_iterations)

    # ── Flush and close the TeeLogger ────────────────────────────────────────
    # Restore original stdout/stderr so any code after train() works normally.
    print(f"[train] Full training log saved to: {log_path}", flush=True)
    if isinstance(sys.stdout, _TeeLogger):
        sys.stdout.close()
        sys.stdout = _orig_stdout
    if isinstance(sys.stderr, _TeeLogger):
        sys.stderr.close()
        sys.stderr = _orig_stderr


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

    # Allow HF_TOKEN from env
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    train(cfg)


if __name__ == "__main__":
    main()
