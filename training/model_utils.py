"""
model_utils.py — Model loading, LoRA setup, GPU detection, checkpoint resume.

All persistent state lives on Hugging Face Hub.
GPU is ephemeral — we only ever write to Hub, never assume local disk survives.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from huggingface_hub import HfApi, snapshot_download

# Reduce CUDA allocator fragmentation — critical on A10G where free memory
# is non-contiguous after generation KV-cache eviction.
# Must be set before any CUDA allocation.
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ────────────────────────────────────────────────
# GPU Detection
# ────────────────────────────────────────────────

def detect_gpu_tier() -> str:
    """Return 'a100', 'a10g', or 't4' based on GPU name and VRAM."""
    if not torch.cuda.is_available():
        print("[model_utils] No CUDA detected — will be extremely slow")
        return "t4"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0).lower()
    if "a100" in name or vram_gb >= 70:
        return "a100"
    elif "a10" in name or vram_gb >= 20:
        return "a10g"
    else:
        return "t4"


def gpu_scaled_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-scale LoRA rank and seq_length based on GPU tier.

    SRE actions are short (~40 tokens output). The bottleneck is
    env API latency, not context length. So we keep seq_len tight
    (1024 is plenty) and spend VRAM on rank + episodes instead.
    """
    tier = detect_gpu_tier()

    # GPU-tier safe DEFAULTS — applied only if the config value is BELOW these floors.
    # These are not caps: if the user deliberately sets a higher value (e.g. lora_rank=64),
    # it is preserved. The override only kicks in to prevent unsafe low values.
    #
    # A10G headroom (verified empirically):
    #   base model: 3.3 GiB  |  one forward (bf16, seq=512, batch=1): 8.9 GiB
    #   peak with loss_batch_size=2: ~17.8 GiB  |  22.3 GiB total  →  ~4.5 GiB headroom
    #   lora_rank=64 vs 32: +83 MiB difference — negligible
    if tier == "a100":
        safe_defaults = {
            "max_seq_length": 2048,
            "lora_rank": 48,
            "lora_alpha": 48,
            "per_device_train_batch_size": 4,
            "loss_batch_size": 4,
        }
    elif tier == "a10g":
        safe_defaults = {
            "max_seq_length": 512,    # keep tight: logits at 1024 blow past 22 GiB
            "lora_rank": 64,          # verified safe: +83 MiB vs rank-32
            "lora_alpha": 64,         # keep == lora_rank
            "per_device_train_batch_size": 2,
            "loss_batch_size": 1,     # proven safe. New fused-CE path may allow 2 — test after validating.
        }
        # Hard floors — values below these will always OOM on A10G
        hard_floors = {
            "max_seq_length": 256,
            "loss_batch_size": 1,
        }
    else:  # t4 / unknown
        safe_defaults = {
            "max_seq_length": 512,
            "lora_rank": 16,
            "lora_alpha": 16,
            "per_device_train_batch_size": 1,
            "loss_batch_size": 1,
        }
        hard_floors = {}

    # Apply: only override if config is missing the key entirely, or below the hard floor.
    # Env vars (ANTIATROPOS_<KEY>=<value>) always win over everything.
    hard_floors = locals().get("hard_floors", {})
    for key, default_val in safe_defaults.items():
        env_key = f"ANTIATROPOS_{key.upper()}"

        # Env var always wins
        if env_key in os.environ:
            val = os.environ[env_key]
            cfg[key] = int(val) if isinstance(default_val, int) else float(val)
            print(f"[model_utils] env override: {key} = {cfg[key]}")
            continue

        current = cfg.get(key)
        floor = hard_floors.get(key)

        if current is None:
            # Key not set at all — apply safe default
            cfg[key] = default_val
            print(f"[model_utils] GPU {tier}: setting {key} = {default_val} (not in config)")
        elif floor is not None and current < floor:
            # Below hard floor — unsafe, must override
            print(f"[model_utils] GPU {tier}: {key}={current} is below safe floor {floor}, "
                  f"raising to {floor}")
            cfg[key] = floor
        else:
            # User set a value >= floor — respect it
            if current != default_val:
                print(f"[model_utils] GPU {tier}: keeping user config {key}={current} "
                      f"(default was {default_val})")

    return cfg


# ────────────────────────────────────────────────
# Model Loading
# ────────────────────────────────────────────────

def load_base_model(cfg: Dict[str, Any]):
    """Load base model with Unsloth QLoRA. Returns (model, tokenizer)."""
    from unsloth import FastLanguageModel

    model_name = cfg["base_model"]
    max_seq_length = cfg.get("max_seq_length", 1024)
    load_in_4bit = cfg.get("load_in_4bit", True)

    print(f"[model_utils] Loading {model_name} "
          f"(seq_len={max_seq_length}, 4bit={load_in_4bit})")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect bf16/fp16
        trust_remote_code=True,
    )

    return model, tokenizer


def attach_lora(model, cfg: Dict[str, Any], seed: int = 42):
    """Attach LoRA adapters to the base model."""
    from unsloth import FastLanguageModel

    rank = cfg.get("lora_rank", 32)
    alpha = cfg.get("lora_alpha", 32)
    dropout = cfg.get("lora_dropout", 0.0)
    target_modules = cfg.get("lora_target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    print(f"[model_utils] Attaching LoRA: rank={rank}, alpha={alpha}, "
          f"dropout={dropout}, targets={len(target_modules)} modules")

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[model_utils] VRAM: {vram_used:.2f} / {vram_total:.2f} GiB")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model_utils] Trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    return model


# ────────────────────────────────────────────────
# Checkpoint Resume
# ────────────────────────────────────────────────

def find_latest_checkpoint(hub_repo: str) -> Optional[str]:
    """Check Hub for the latest checkpoint subfolder.

    Checkpoints are stored as: <hub_repo>/checkpoint-<step>/
    Returns the path to download, or None if no checkpoint exists.
    """
    if not hub_repo:
        return None

    try:
        api = HfApi()
        # List all files in the repo, find checkpoint dirs
        files = api.list_repo_files(hub_repo, repo_type="model")
        checkpoint_dirs = set()
        for f in files:
            # checkpoint-123/adapter_model.safetensors
            parts = f.split("/")
            if len(parts) >= 2 and parts[0].startswith("checkpoint-"):
                try:
                    step = int(parts[0].split("-")[1])
                    checkpoint_dirs.add(step)
                except (ValueError, IndexError):
                    continue

        if not checkpoint_dirs:
            return None

        latest_step = max(checkpoint_dirs)
        ckpt_path = f"checkpoint-{latest_step}"
        print(f"[model_utils] Found Hub checkpoint: {hub_repo}/{ckpt_path}")
        return ckpt_path

    except Exception as e:
        print(f"[model_utils] Could not check Hub for checkpoints: {e}")
        return None


def download_checkpoint(hub_repo: str, checkpoint_path: str,
                        local_dir: str = "/tmp/antiatropos_ckpt") -> str:
    """Download a checkpoint from Hub to local disk.

    Returns the local path containing adapter files.
    """
    print(f"[model_utils] Downloading checkpoint {hub_repo}/{checkpoint_path}...")
    snapshot_download(
        repo_id=hub_repo,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=[f"{checkpoint_path}/*"],
    )
    return str(Path(local_dir) / checkpoint_path)


def load_from_checkpoint(model, tokenizer, ckpt_local_path: str):
    """Load LoRA weights from a local checkpoint directory."""
    from peft import PeftModel

    print(f"[model_utils] Loading adapter from {ckpt_local_path}")
    # For Unsloth models, we reload the adapter
    model.load_adapter(ckpt_local_path)
    return model


# ────────────────────────────────────────────────
# Save & Push
# ────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, output_dir: str, step: int) -> str:
    """Save adapter + tokenizer locally. Returns the checkpoint path."""
    ckpt_dir = str(Path(output_dir) / f"checkpoint-{step}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"[model_utils] Checkpoint saved: {ckpt_dir}")
    return ckpt_dir


def push_to_hub(
    local_dir: str,
    hub_repo: str,
    commit_message: str = "",
    path_in_repo: str = "",
) -> None:
    """Push a local directory to a Hub model repo.

    Args:
        local_dir:     Local folder to upload.
        hub_repo:      HF model repo (e.g. 'Keshav051/antiatropos-qlora').
        commit_message: Commit message.
        path_in_repo:  Subfolder INSIDE the Hub repo (e.g. 'run_001/checkpoint-0005').
                       If empty, uploads to the repo root (legacy behaviour).
    """
    if not hub_repo:
        print("[model_utils] No hub_model_repo configured, skipping push")
        return

    try:
        from huggingface_hub import upload_folder

        kwargs = dict(
            folder_path=local_dir,
            repo_id=hub_repo,
            repo_type="model",
            commit_message=commit_message or "Upload from AntiAtropos training",
        )
        if path_in_repo:
            kwargs["path_in_repo"] = path_in_repo

        upload_folder(**kwargs)
        dest = f"{hub_repo}/{path_in_repo}" if path_in_repo else hub_repo
        print(f"[model_utils] Pushed → {dest}")
    except Exception as e:
        print(f"[model_utils] Push failed: {e}")


def push_adapter_to_hub(model, tokenizer, hub_repo: str,
                        step: int, output_dir: str = "/tmp/antiatropos_final") -> None:
    """Save final adapter and push to Hub."""
    if not hub_repo:
        print("[model_utils] No hub_model_repo configured, skipping final push")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    push_to_hub(output_dir, hub_repo, f"AntiAtropos QLoRA step {step}")
