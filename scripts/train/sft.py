"""
sft.py — Full fine-tuning of JarvisVLA (Qwen2-VL 7B) on skill-annotated VQA data.

Trains on the JSONL dataset produced by prepare_sft_data.py.  Uses the same
VLAMultimodalChatDataCollatorforVLM from JarvisVLA, so image pre-processing,
label masking, and special-token handling are identical to the original SFT.

The model learns to produce either:
  • <think>\\n{reasoning}\\n</think>{action_tokens}   (skill-annotated steps)
  • {action_tokens}                                    (unannotated steps)

The <think> and </think> tokens are already in the extended JarvisVLA vocabulary
(special_token.json indices 12–13), so no new tokens are needed.

Usage — single GPU:
    python scripts/sft.py \\
        --data-dir   sft_data \\
        --model-path models/jarvis_vla_qwen2_vl_7b_sft \\
        --output-dir models/jarvis_vla_skill_sft

Usage — multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=2 scripts/sft.py \\
        --data-dir   sft_data \\
        --model-path models/jarvis_vla_qwen2_vl_7b_sft \\
        --output-dir models/jarvis_vla_skill_sft \\
        --batch-size 2 --grad-accum 8

Key flags:
    --freeze-vision    Freeze visual encoder weights (saves ~30 % memory).
    --max-seq-length   Token budget per example (default 2048).
    --lr               Peak learning rate (default 2e-5).
    --epochs           Training epochs (default 3).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/root/autodl-tmp/JarvisVLA")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import load_dataset
from rich import print
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from jarvisvla import QWEN_SPECIAL_TOKENS
except ImportError:
    QWEN_SPECIAL_TOKENS = "/root/autodl-tmp/JarvisVLA/assets/special_token.json"

from jarvisvla.train.data_collator import make_collator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main    = local_rank in {0, -1}

    # ── Load processor & add special tokens ──────────────────────────────────
    if is_main:
        print(f"[cyan]Loading processor from {args.model_path}[/cyan]")
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path,
        do_rescale=False,
        patch_size=14,
        vision_feature_select_strategy="default",
    )
    with open(QWEN_SPECIAL_TOKENS, "r") as f:
        special_tokens = json.load(f)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    if is_main:
        print(f"[cyan]Loading model from {args.model_path}[/cyan]")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # Resize token embeddings to match the extended vocabulary
    model.resize_token_embeddings(len(processor.tokenizer))

    # ── Optional: freeze visual encoder ──────────────────────────────────────
    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
                frozen += param.numel()
        if is_main:
            print(f"  [dim]Frozen visual encoder params: {frozen:,}[/dim]")

    if is_main:
        trainable = _count_trainable(model)
        total     = _count_total(model)
        print(f"  Trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.1f} %)")

    # ── Data collator ─────────────────────────────────────────────────────────
    # Resolve image_folder: prefer meta.json from data-dir, fall back to arg
    meta_file = Path(args.data_dir) / "meta.json"
    if meta_file.exists():
        meta        = json.loads(meta_file.read_text())
        image_folder = Path(meta.get("trajectory_dir", args.trajectory_dir))
    else:
        image_folder = Path(args.trajectory_dir)

    if is_main:
        print(f"  image_folder → {image_folder}")

    collator = make_collator(
        "VLAMultimodalChatDataCollatorforVLM",
        processor    = processor,
        model_path   = "qwen2_vl",
        image_folder = image_folder,
        max_seq_length = args.max_seq_length,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_dir   = Path(args.data_dir)
    raw        = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "valid": str(data_dir / "valid.jsonl"),
        },
    )
    if is_main:
        print(f"  train={len(raw['train'])}  valid={len(raw['valid'])}")

    # ── Training arguments ────────────────────────────────────────────────────
    output_dir  = Path(args.output_dir)
    has_ckpt    = bool(list(output_dir.glob("checkpoint-*")))

    training_args = TrainingArguments(
        output_dir                    = str(output_dir),
        num_train_epochs              = args.epochs,
        per_device_train_batch_size   = args.batch_size,
        per_device_eval_batch_size    = args.batch_size,
        gradient_accumulation_steps   = args.grad_accum,
        learning_rate                 = args.lr,
        weight_decay                  = 0.01,
        warmup_ratio                  = 0.03,
        lr_scheduler_type             = "cosine",
        bf16                          = True,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        logging_steps                 = 10,
        save_steps                    = args.save_steps,
        eval_steps                    = args.save_steps,
        eval_strategy                 = "steps",
        save_strategy                 = "steps",
        save_total_limit              = 2,
        dataloader_num_workers        = 4,
        remove_unused_columns         = False,
        report_to                     = "none",
        ddp_find_unused_parameters    = False,
        # Required by JarvisVLA's collator — prevents TRL from re-processing data
        dataset_kwargs                = {"skip_prepare_dataset": True},
    )
    # dataset_text_field is read by SFTTrainer; Trainer ignores it but we set it
    # for forwards-compatibility if someone swaps in SFTTrainer later.
    training_args.dataset_text_field = "text"

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model          = model,
        args           = training_args,
        data_collator  = collator,
        train_dataset  = raw["train"],
        eval_dataset   = raw["valid"],
        tokenizer      = processor.tokenizer,
    )

    if is_main:
        think_frac = sum(
            1 for ex in raw["train"]
            if any(
                "<think>" in item.get("text", "")
                for turn in ex["conversations"]
                for item in turn["content"]
            )
        ) / max(1, len(raw["train"]))
        print(f"  Think-annotated steps: {think_frac:.1%} of train set")

    # ── Train ─────────────────────────────────────────────────────────────────
    if has_ckpt:
        if is_main:
            print("[yellow]Resuming from existing checkpoint …[/yellow]")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    if is_main:
        print(f"[bold green]Saved to {output_dir}[/bold green]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MineAgent skill SFT")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data-dir", type=str,
                        default="/root/autodl-tmp/MineAgent/sft_data",
                        help="Directory with train.jsonl / valid.jsonl (from prepare_sft_data.py)")
    parser.add_argument("--trajectory-dir", type=str,
                        default="/root/autodl-tmp/MineAgent/trajectories",
                        help="Root directory of trajectories (for resolving image paths). "
                             "Overridden by meta.json inside --data-dir if present.")

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model-path", type=str,
                        default="/root/autodl-tmp/MineAgent/models/jarvis_vla_qwen2_vl_7b_sft",
                        help="Path to base model checkpoint (Qwen2-VL 7B SFT)")
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/MineAgent/models/jarvis_vla_skill_sft",
                        help="Where to save the fine-tuned checkpoint")
    parser.add_argument("--freeze-vision", action="store_true",
                        help="Freeze the visual encoder (saves ~30 %% GPU memory)")

    # ── Training hyper-parameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch-size",     type=int,   default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum",     type=int,   default=16,
                        help="Gradient accumulation steps (effective batch = batch_size × gpus × grad_accum)")
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int,   default=2048)
    parser.add_argument("--save-steps",     type=int,   default=200)

    main(parser.parse_args())
