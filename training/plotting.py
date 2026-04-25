"""
plotting.py — Comprehensive training visualization for AntiAtropos.

Generates publication-quality plots covering EVERY aspect of training:
  1. Reward curve (train + eval, per-task)
  2. Loss curve
  3. Gradient norm (training health)
  4. Action type distribution (over time)
  5. Invalid action rate
  6. Per-task reward comparison (FT vs heuristic)
  7. Episode length distribution
  8. Reward distribution histogram
  9. Iteration time (throughput)
  10. Summary dashboard (all-in-one)

All plots are saved locally and pushed to Hub.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ────────────────────────────────────────────────
# Style Configuration
# ────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 9,
})

ACTION_COLORS = {
    "NO_OP": "#8b949e",
    "SCALE_UP": "#3fb950",
    "SCALE_DOWN": "#f85149",
    "REROUTE_TRAFFIC": "#58a6ff",
    "SHED_LOAD": "#d2a8ff",
}

TASK_COLORS = {
    "task-1": "#58a6ff",
    "task-2": "#f0883e",
    "task-3": "#3fb950",
}

PRIMARY_COLOR = "#58a6ff"
ACCENT_COLOR = "#f0883e"
SUCCESS_COLOR = "#3fb950"
DANGER_COLOR = "#f85149"


def _smooth(data: List[float], window: int = 20) -> List[float]:
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid").tolist()


# ────────────────────────────────────────────────
# Individual Plot Functions
# ────────────────────────────────────────────────

def plot_reward_curve(
    train_metrics: List[Dict],
    eval_metrics: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    if train_metrics:
        iters = [m["iteration"] for m in train_metrics]
        rewards = [m["avg_reward"] for m in train_metrics]
        ax.plot(iters, rewards, color=PRIMARY_COLOR, alpha=0.3,
                linewidth=0.8, label="Train (raw)")
        if len(rewards) > 10:
            w = min(20, len(rewards) // 3)
            sm = _smooth(rewards, w)
            ax.plot(iters[w-1:], sm, color=PRIMARY_COLOR,
                    linewidth=2, label=f"Train (MA-{w})")

    if eval_metrics:
        ei = [m["step"] for m in eval_metrics if m.get("type") == "eval"]
        ef = [m.get("overall_ft_avg", 0) for m in eval_metrics
              if m.get("type") == "eval"]
        eh = [m.get("overall_heuristic_avg", 0) for m in eval_metrics
              if m.get("type") == "eval"]
        if ei:
            ax.plot(ei, ef, "o-", color=SUCCESS_COLOR,
                    linewidth=2, markersize=6, label="FT (eval)")
            ax.plot(ei, eh, "s--", color=ACCENT_COLOR,
                    linewidth=2, markersize=6, label="Heuristic (eval)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Reward")
    ax.set_title("Reward Curve - FT vs Heuristic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(
    train_metrics: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if not train_metrics:
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    iters = [m["iteration"] for m in train_metrics]
    losses = [m["loss"] for m in train_metrics]
    ax.plot(iters, losses, color=ACCENT_COLOR, alpha=0.4, linewidth=0.8)
    if len(losses) > 10:
        w = min(20, len(losses) // 3)
        sm = _smooth(losses, w)
        ax.plot(iters[w-1:], sm, color=ACCENT_COLOR,
                linewidth=2, label=f"MA-{w}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (REINFORCE)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_norm(
    train_metrics: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not train_metrics:
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    iters = [m["iteration"] for m in train_metrics]
    gn = [m.get("grad_norm", 0) for m in train_metrics]
    ax.semilogy(iters, gn, color="#d2a8ff", alpha=0.5, linewidth=0.8)
    if len(gn) > 10:
        w = min(20, len(gn) // 3)
        sm = _smooth(gn, w)
        ax.semilogy(iters[w-1:], sm, color="#d2a8ff", linewidth=2,
                     label=f"MA-{w}")
    ax.axhline(y=1.0, color=DANGER_COLOR, linestyle="--", alpha=0.5,
               label="Clip threshold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Grad Norm (log scale)")
    ax.set_title("Gradient Norm - Training Stability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_action_distribution(
    train_metrics: List[Dict],
    episodes_data: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    action_counts: Counter = Counter()
    for ep in episodes_data:
        for t in ep.get("transitions", []):
            at = t.get("action", {}).get("action_type", "UNKNOWN")
            action_counts[at] += 1

    if action_counts:
        labels = list(action_counts.keys())
        sizes = list(action_counts.values())
        colors = [ACTION_COLORS.get(l, "#8b949e") for l in labels]
        axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                    startangle=90, pctdistance=0.85)
        axes[0].set_title("Action Distribution (Overall)")

    if train_metrics:
        iters = [m["iteration"] for m in train_metrics]
        invalid_rates = []
        for m in train_metrics:
            n_ep = max(m.get("num_episodes", 1), 1)
            invalid_rates.append(m.get("invalid_actions", 0) / n_ep)
        axes[1].fill_between(iters, invalid_rates, alpha=0.3,
                             color=DANGER_COLOR)
        axes[1].plot(iters, invalid_rates, color=DANGER_COLOR,
                     linewidth=1.5, label="Invalid rate")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Invalid Actions / Episode")
        axes[1].set_title("Invalid Action Rate")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_per_task_rewards(
    eval_metrics: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, task_id in enumerate(["task-1", "task-2", "task-3"]):
        ax = axes[idx]
        ft_key = f"eval_{task_id}_ft_avg_reward"
        heur_key = f"eval_{task_id}_heuristic_avg_reward"
        ft_vals, heur_vals, steps = [], [], []
        for m in eval_metrics:
            if m.get("type") != "eval":
                continue
            if ft_key in m:
                steps.append(m["step"])
                ft_vals.append(m[ft_key])
                heur_vals.append(m.get(heur_key, 0))
        if steps:
            ax.plot(steps, ft_vals, "o-", color=TASK_COLORS[task_id],
                    linewidth=2, markersize=5, label="FT")
            ax.plot(steps, heur_vals, "s--", color="#8b949e",
                    linewidth=2, markersize=5, label="Heuristic")
        ax.set_title(task_id)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Avg Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Per-Task Reward: Fine-Tuned vs Heuristic", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_iteration_time(
    train_metrics: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not train_metrics:
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    iters = [m["iteration"] for m in train_metrics]
    times = [m.get("iter_time_s", 0) for m in train_metrics]
    ax.bar(iters, times, color=PRIMARY_COLOR, alpha=0.6, width=1.0)
    if times:
        ax.axhline(y=np.mean(times), color=ACCENT_COLOR, linestyle="--",
                   label=f"Avg: {np.mean(times):.1f}s")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (s)")
    ax.set_title("Iteration Wall-Clock Time")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_episode_length_distribution(
    episodes_data: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    lengths = [len(ep.get("transitions", [])) for ep in episodes_data]
    if not lengths:
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    ax.hist(lengths, bins=range(min(lengths), max(lengths) + 2),
            color=PRIMARY_COLOR, alpha=0.7, edgecolor="#30363d")
    ax.axvline(x=np.mean(lengths), color=ACCENT_COLOR, linestyle="--",
               label=f"Mean: {np.mean(lengths):.1f}")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_reward_distribution(
    train_metrics: List[Dict],
    episodes_data: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_rewards = []
    for ep in episodes_data:
        for t in ep.get("transitions", []):
            all_rewards.append(t.get("reward", 0))
    if not all_rewards:
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    ax.hist(all_rewards, bins=50, color="#d2a8ff", alpha=0.7,
            edgecolor="#30363d")
    ax.axvline(x=np.mean(all_rewards), color=SUCCESS_COLOR, linestyle="--",
               label=f"Mean: {np.mean(all_rewards):.3f}")
    ax.axvline(x=np.median(all_rewards), color=ACCENT_COLOR, linestyle=":",
               label=f"Median: {np.median(all_rewards):.3f}")
    ax.set_xlabel("Step Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution (all steps)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────
# Summary Dashboard
# ────────────────────────────────────────────────

def plot_dashboard(
    train_metrics: List[Dict],
    eval_metrics: List[Dict],
    episodes_data: List[Dict],
    output_path: str,
    dpi: int = 150,
) -> None:
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Reward curve
    ax1 = fig.add_subplot(gs[0, 0])
    if train_metrics:
        iters = [m["iteration"] for m in train_metrics]
        rewards = [m["avg_reward"] for m in train_metrics]
        ax1.plot(iters, rewards, color=PRIMARY_COLOR, alpha=0.3, linewidth=0.8)
        if len(rewards) > 10:
            sm = _smooth(rewards, min(20, len(rewards)//3))
            ax1.plot(iters[len(iters)-len(sm):], sm,
                     color=PRIMARY_COLOR, linewidth=2)
    if eval_metrics:
        ei = [m["step"] for m in eval_metrics if m.get("type") == "eval"]
        ef = [m.get("overall_ft_avg", 0) for m in eval_metrics
              if m.get("type") == "eval"]
        eh = [m.get("overall_heuristic_avg", 0) for m in eval_metrics
              if m.get("type") == "eval"]
        if ei:
            ax1.plot(ei, ef, "o-", color=SUCCESS_COLOR, linewidth=2,
                     markersize=5, label="FT")
            ax1.plot(ei, eh, "s--", color=ACCENT_COLOR, linewidth=2,
                     markersize=5, label="Heuristic")
    ax1.set_title("Reward Curve")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Avg Reward")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Panel 2: Loss curve
    ax2 = fig.add_subplot(gs[0, 1])
    if train_metrics:
        iters = [m["iteration"] for m in train_metrics]
        losses = [m["loss"] for m in train_metrics]
        ax2.plot(iters, losses, color=ACCENT_COLOR, alpha=0.5, linewidth=0.8)
        if len(losses) > 10:
            sm = _smooth(losses, min(20, len(losses)//3))
            ax2.plot(iters[len(iters)-len(sm):], sm,
                     color=ACCENT_COLOR, linewidth=2)
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Gradient norm
    ax3 = fig.add_subplot(gs[0, 2])
    if train_metrics:
        iters = [m["iteration"] for m in train_metrics]
        gn = [m.get("grad_norm", 0) for m in train_metrics]
        ax3.semilogy(iters, gn, color="#d2a8ff", alpha=0.5, linewidth=0.8)
        if len(gn) > 10:
            sm = _smooth(gn, min(20, len(gn)//3))
            ax3.semilogy(iters[len(iters)-len(sm):], sm,
                          color="#d2a8ff", linewidth=2)
        ax3.axhline(y=1.0, color=DANGER_COLOR, linestyle="--", alpha=0.5)
    ax3.set_title("Gradient Norm")
    ax3.set_xlabel("Iteration"); ax3.set_ylabel("Grad Norm (log)")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Per-task rewards
    ax4 = fig.add_subplot(gs[1, 0])
    for task_id in ["task-1", "task-2", "task-3"]:
        ft_key = f"eval_{task_id}_ft_avg_reward"
        ft_vals, steps = [], []
        for m in eval_metrics:
            if m.get("type") != "eval":
                continue
            if ft_key in m:
                steps.append(m["step"])
                ft_vals.append(m[ft_key])
        if steps:
            ax4.plot(steps, ft_vals, "o-", color=TASK_COLORS[task_id],
                     linewidth=1.5, markersize=4, label=f"{task_id} FT")
    ax4.set_title("Per-Task FT Reward")
    ax4.set_xlabel("Iteration"); ax4.set_ylabel("Avg Reward")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # Panel 5: Action distribution
    ax5 = fig.add_subplot(gs[1, 1])
    action_counts: Counter = Counter()
    for ep in episodes_data:
        for t in ep.get("transitions", []):
            at = t.get("action", {}).get("action_type", "UNKNOWN")
            action_counts[at] += 1
    if action_counts:
        labels = list(action_counts.keys())
        sizes = list(action_counts.values())
        colors = [ACTION_COLORS.get(l, "#8b949e") for l in labels]
        ax5.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, pctdistance=0.85)
    ax5.set_title("Action Distribution")

    # Panel 6: Iteration time
    ax6 = fig.add_subplot(gs[1, 2])
    if train_metrics:
        iters = [m["iteration"] for m in train_metrics]
        times = [m.get("iter_time_s", 0) for m in train_metrics]
        ax6.bar(iters, times, color=PRIMARY_COLOR, alpha=0.6, width=1.0)
        if times:
            ax6.axhline(y=np.mean(times), color=ACCENT_COLOR, linestyle="--",
                        label=f"Avg: {np.mean(times):.1f}s")
    ax6.set_title("Iteration Time")
    ax6.set_xlabel("Iteration"); ax6.set_ylabel("Seconds")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3, axis="y")

    fig.suptitle("AntiAtropos QLoRA Training Dashboard", fontsize=16,
                 fontweight="bold", color="#f0f6fc")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────

def generate_all_plots(
    train_metrics: List[Dict],
    eval_metrics: List[Dict],
    episodes_data: List[Dict],
    output_dir: str,
    cfg: Dict[str, Any],
) -> List[str]:
    dpi = cfg.get("plot_dpi", 150)
    fmt = cfg.get("plot_format", "png")
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    def _save(name, plot_fn, *args):
        path = str(plot_dir / f"{name}.{fmt}")
        try:
            plot_fn(*args, path, dpi)
            paths.append(path)
            print(f"[plotting] Saved {path}")
        except Exception as e:
            print(f"[plotting] Failed {name}: {e}")

    _save("reward_curve", plot_reward_curve, train_metrics, eval_metrics)
    _save("loss_curve", plot_loss_curve, train_metrics)
    _save("gradient_norm", plot_gradient_norm, train_metrics)
    _save("action_distribution", plot_action_distribution,
          train_metrics, episodes_data)
    _save("per_task_rewards", plot_per_task_rewards, eval_metrics)
    _save("iteration_time", plot_iteration_time, train_metrics)
    _save("episode_length", plot_episode_length_distribution, episodes_data)
    _save("reward_distribution", plot_reward_distribution,
          train_metrics, episodes_data)
    _save("dashboard", plot_dashboard,
          train_metrics, eval_metrics, episodes_data)

    return paths


def push_plots_to_hub(
    plot_paths: List[str],
    hub_repo: str,
    iteration: int,
) -> None:
    if not hub_repo or not plot_paths:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for path in plot_paths:
            filename = Path(path).name
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"plots/iter_{iteration}/{filename}",
                repo_id=hub_repo,
                repo_type="model",
                commit_message=f"Training plots - iteration {iteration}",
            )
        print(f"[plotting] Pushed {len(plot_paths)} plots to {hub_repo}")
    except Exception as e:
        print(f"[plotting] Push failed: {e}")


def episodes_to_plot_data(episodes: List) -> List[Dict]:
    data = []
    for ep in episodes:
        transitions = []
        for t in ep.transitions:
            transitions.append({
                "action": {
                    "action_type": t.action.action_type,
                    "target_node_id": t.action.target_node_id,
                    "parameter": t.action.parameter,
                    "is_valid": t.action.is_valid,
                },
                "reward": t.reward,
            })
        data.append({
            "task_id": ep.task_id,
            "avg_reward": ep.avg_reward,
            "total_reward": ep.total_reward,
            "num_invalid": ep.num_invalid,
            "transitions": transitions,
        })
    return data
