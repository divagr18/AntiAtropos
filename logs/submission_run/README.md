# Submission Run — `run_0011`

**This is the official AntiAtropos submission run.** All results, metrics, and behavioral analysis discussed in the README and blog post are derived from this training run.

## Overview

| Field | Value |
|-------|-------|
| Run ID | `run_0011` |
| Loss Type | REINFORCE + baseline |
| Base Model | Qwen/Qwen3.5-4B |
| QLoRA Rank | 64 |
| Iterations | 500 |
| Episodes per Iteration | 6 (2 per task for curriculum balance) |
| Max Steps per Episode | 20 |
| Training Time | ~2 hours on A10G (24 GiB) |
| Cost | ~$0.68 (a10g-large at $0.34/hr) |
| Results | Task-1: 0.88 / Task-2: 0.82 / Task-3: 0.94 |

## What's in This Folder

| File | Description |
|------|-------------|
| `SUBMISSION_TRAINING_RUN.log` | Full stderr/stdout log from the training container |
| `step_metrics.jsonl` | Per-step telemetry across all episodes (queue depth, latency, reward, actions, etc.) |
| `run_info.json` | Configuration snapshot for full reproducibility |

## Hub Mirror

All artifacts are also available on the Hugging Face Hub:
- **Model checkpoints, metrics, plots:** [Keshav051/antiatropos-qlora/run_0011](https://huggingface.co/Keshav051/antiatropos-qlora/tree/main/run_0011)

## Training Command

This run was launched via Hugging Face Jobs using:

```bash
python training/launch_train.py --run-id run_0011 --num-iterations 500 --num-episodes 6
```

See the [Training section in README](../../README.md#training-rl-with-unsloth--hugging-face-jobs) for detailed launch instructions.
