# GRPO Experiment — `grpo_run_001`

> ⚠️ **IMPORTANT: This is NOT the submission run.**  
> This folder contains an **experimental GRPO training run** that was explored during research but is **not** part of the final AntiAtropos submission. All submission results come from `run_0011` (REINFORCE + baseline), located in the [`logs/submission_run/`](../submission_run/) directory.

## Why This Exists

During development we experimented with **Group Relative Policy Optimization (GRPO)** as an alternative to REINFORCE. This run documents that exploration for full transparency and reproducibility.

## Key Differences from Submission

| Aspect | This Run (GRPO) | Submission (REINFORCE) |
|--------|-----------------|----------------------|
| Loss function | GRPO (K=4) | REINFORCE + baseline |
| Training iterations | 15 (short test) | 500 (full convergence) |
| Result quality | Not comparable — truncated diagnostic run | Converged policy |
| Status | **Experimental only** | **Canonical submission** |

## What's in This Folder

| File | Description |
|------|-------------|
| `NON_RECORD_GRPO.log` | Stderr/stdout log from the GRPO experiment |

## Hub Mirror

Also available on the Hugging Face Hub at [Keshav051/antiatropos-qlora/grpo_run_001](https://huggingface.co/Keshav051/antiatropos-qlora/tree/main/grpo_run_001).

## Final Takeaway

GRPO showed potential but was ultimately **not used in the submission**. The REINFORCE baseline approach produced equivalent or better results with a faster, more stable training loop. See [The Road Not Taken](../../Blog.md#the-road-not-taken-grpo-experimentation) section in the blog for the full analysis.
