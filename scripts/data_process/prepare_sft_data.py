"""
prepare_sft_data.py — Convert MineAgent trajectories to HuggingFace SFT dataset.

Reads skill_trajectory.json (preferred) or trajectory.json from each trajectory
directory and produces JSONL files consumable by sft.py.

Output format per example:
  id            : "{run_tag}_step_{N:05d}"
  conversations : [
    {role: user,      content: [{type:text, text:"<instr>\\nobservation: "}, {type:image}]},
    {role: assistant, content: [{type:text, text:"<think>\\n...\\n</think><action_tokens>"}]}
  ]
  image         : ["<traj_name>/frames/<N>.png"]   (relative to --trajectory-dir)

Rules:
  • Steps with model_from_cache=True are skipped (duplicate action chunks).
  • Steps with empty model_raw_output are skipped.
  • skill_trajectory.json takes precedence over trajectory.json.
  • Failed trajectories are skipped unless --include-failed.
  • When frame_path is missing, frames are extracted from video.mp4 on the fly
    and saved to <traj_dir>/frames/ for reuse.

Usage:
    python scripts/prepare_sft_data.py \\
        --trajectory-dir trajectories \\
        --output-dir     sft_data \\
        --valid-split    0.05

    # Include failed runs too:
    python scripts/prepare_sft_data.py \\
        --trajectory-dir trajectories \\
        --output-dir     sft_data \\
        --include-failed
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

import cv2
from PIL import Image
from rich import print

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.rag import format_skills_for_llm


# ─────────────────────────────────────────────────────────────────────────────
# Skill context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_skills_context(skills: list) -> str:
    """Format skill records as a [Retrieved Skills] block for the user prompt."""
    if not skills:
        return ""
    return f"[Retrieved Skills]\n{format_skills_for_llm(skills)}"


def _get_step_skills(meta_skills: list, retrieved_ids: list) -> list:
    """
    Return the subset of meta_skills matching the given IDs.
    Falls back to all meta_skills if none of the IDs are found.
    """
    if not retrieved_ids:
        return meta_skills
    id_map = {s["skill_id"]: s for s in meta_skills}
    matched = [id_map[sid] for sid in retrieved_ids if sid in id_map]
    return matched if matched else meta_skills


# ─────────────────────────────────────────────────────────────────────────────
# Frame utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_frame(traj_dir: Path, step_data: dict, video_path: Path) -> Optional[str]:
    """
    Return a frame path relative to traj_dir.parent, ensuring the file exists.
    Falls back to extracting the frame from video.mp4 if frame_path is absent.
    """
    # Use existing frame if recorded
    if step_data.get("frame_path"):
        full = traj_dir.parent / step_data["frame_path"]
        if full.exists():
            return step_data["frame_path"]

    # Extract from video
    if not video_path.exists():
        return None

    frames_dir = traj_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    step_idx = step_data["step"]
    out_path  = frames_dir / f"{step_idx:05d}.png"

    if not out_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, step_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(str(out_path))

    return str(out_path.relative_to(traj_dir.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory processing
# ─────────────────────────────────────────────────────────────────────────────

def process_trajectory(
    traj_dir: Path,
    include_failed: bool = False,
    skill_persist_steps: int = 0,
) -> tuple:
    """Convert one trajectory directory into a list of SFT examples.

    For skill trajectories: every step's user prompt includes a [Retrieved Skills]
    block derived from meta.skills (step-specific for annotated steps, episode-level
    for all others). The [Active Skill] persistence mechanism is not used for skill
    trajectories — it is only kept as a fallback for plain trajectory.json files.

    Returns (examples, skill_retrieved_count) where skill_retrieved_count is the
    number of examples that received a [Retrieved Skills] context block.
    """
    skill_file = traj_dir / "skill_trajectory.json"
    plain_file = traj_dir / "trajectory.json"

    if skill_file.exists():
        traj      = json.loads(skill_file.read_text(encoding="utf-8"))
        has_skill = True
    elif plain_file.exists():
        traj      = json.loads(plain_file.read_text(encoding="utf-8"))
        has_skill = False
    else:
        return [], 0

    meta = traj["meta"]
    if not meta.get("success") and not include_failed:
        return [], 0

    run_tag    = meta.get("run", traj_dir.name)
    steps      = traj["trajectory"]
    video_path = traj_dir / "video.mp4"
    examples   = []

    # Pre-compute episode-level skill context from meta (for skill trajectories)
    episode_skills     = meta.get("skills", []) if has_skill else []
    episode_skill_block = _format_skills_context(episode_skills)

    # Legacy [Active Skill] state — only used for plain trajectory.json files
    active_skill_text      = ""
    active_skill_remaining = 0
    skill_retrieved_count  = 0

    for step_data in steps:
        skill_think = step_data.get("skill_think", "") if has_skill else ""

        # Legacy [Active Skill] persistence — only for plain trajectories
        if not has_skill:
            if skill_think:
                active_skill_text      = skill_think
                active_skill_remaining = skill_persist_steps
                in_persist_window      = False
            elif active_skill_remaining > 0:
                in_persist_window       = True
                active_skill_remaining -= 1
            else:
                in_persist_window = False
        else:
            in_persist_window = False  # unused for skill trajectories

        # Skip duplicate cached action chunks
        if step_data.get("model_from_cache", False):
            continue

        raw_output = step_data.get("model_raw_output", "")
        if not raw_output:
            continue

        frame_rel = _ensure_frame(traj_dir, step_data, video_path)
        if frame_rel is None:
            continue

        instruction = step_data.get("instruction", "")

        # Build user text
        if has_skill and episode_skill_block:
            # Use step-specific skills for annotated steps, episode-level otherwise.
            step_retrieved_ids = step_data.get("retrieved_skills", [])
            step_skills        = _get_step_skills(episode_skills, step_retrieved_ids)
            skill_block        = _format_skills_context(step_skills)
            user_text          = f"{instruction}\n{skill_block}\nobservation: "
            skill_retrieved_count += 1
        elif in_persist_window and active_skill_text:
            # Legacy [Active Skill] for plain trajectories
            user_text = f"{instruction}\nobservation: \n[Active Skill] {active_skill_text}"
        else:
            user_text = f"{instruction}\nobservation: "

        # Build the target assistant text
        if skill_think:
            assistant_text = f"<think>\n{skill_think}\n</think>{raw_output}"
        else:
            assistant_text = raw_output

        examples.append({
            "id": f"{run_tag}_step_{step_data['step']:05d}",
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": user_text},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_text},
                    ],
                },
            ],
            "image": [frame_rel],
        })

    return examples, skill_retrieved_count


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    traj_base  = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_dirs = sorted(p for p in traj_base.iterdir() if p.is_dir())
    print(f"Found {len(traj_dirs)} trajectory directories in {traj_base}")

    all_examples: List[dict] = []
    total_retrieved = 0
    for traj_dir in traj_dirs:
        examples, retrieved = process_trajectory(
            traj_dir,
            include_failed=args.include_failed,
            skill_persist_steps=args.skill_persist_steps,
        )
        if examples:
            is_skill = (traj_dir / 'skill_trajectory.json').exists()
            tag = f"  skill_retrieved={retrieved}" if (is_skill and retrieved) else ""
            print(f"  {traj_dir.name}: {len(examples)} examples "
                  f"({'skill' if is_skill else 'plain'}){tag}")
        all_examples.extend(examples)
        total_retrieved += retrieved

    if not all_examples:
        print("[red]No examples produced — check that trajectories exist and have frames.[/red]")
        return

    def _has_think(ex: dict) -> bool:
        return any("<think>" in item["text"]
                   for turn in ex["conversations"]
                   for item in turn["content"]
                   if item["type"] == "text")

    def _has_retrieved(ex: dict) -> bool:
        return any("[Retrieved Skills]" in item["text"]
                   for turn in ex["conversations"]
                   for item in turn["content"]
                   if item["type"] == "text")

    # ── Split first (on original examples), then oversample train only ────────
    random.seed(args.seed)
    random.shuffle(all_examples)

    n_valid = max(1, int(len(all_examples) * args.valid_split))
    valid   = all_examples[:n_valid]          # kept clean — no oversampling
    train   = list(all_examples[n_valid:])    # oversampling applied below

    # ── Think-only mode ───────────────────────────────────────────────────────
    if args.think_only:
        train = [e for e in train if _has_think(e)]
        print(f"  [think-only] train filtered to {len(train)} think-annotated examples")

    # ── Think oversample ──────────────────────────────────────────────────────
    elif args.think_oversample > 1:
        think_train = [e for e in train if _has_think(e)]
        plain_train = [e for e in train if not _has_think(e)]
        # Add (N-1) extra copies of think examples
        extra = think_train * (args.think_oversample - 1)
        train = train + extra          # original ratio preserved, think boosted
        random.shuffle(train)
        think_after = sum(1 for e in train if _has_think(e))
        print(f"  [oversample ×{args.think_oversample}]  "
              f"think: {len(think_train)} → {think_after}  "
              f"plain: {len(plain_train)}  "
              f"total train: {len(train)}  "
              f"think ratio: {think_after/len(train):.1%}")

    def _write_jsonl(path: Path, records: List[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "valid.jsonl", valid)

    # Write metadata so sft.py can find the trajectory dir
    skill_train    = sum(1 for e in train if _has_think(e))
    skill_valid    = sum(1 for e in valid if _has_think(e))
    retrieved_train = sum(1 for e in train if _has_retrieved(e))
    meta = {
        "trajectory_dir":              str(traj_base.resolve()),
        "total_examples":              len(all_examples),
        "train_examples":              len(train),
        "valid_examples":              len(valid),
        "skill_examples_train":        skill_train,
        "skill_examples_valid":        skill_valid,
        "think_ratio_train":           round(skill_train / max(1, len(train)), 4),
        "skill_retrieved_examples":    retrieved_train,
        "think_oversample":            args.think_oversample,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[bold green]Done[/bold green]  "
          f"train={len(train)} (think={skill_train}, ratio={skill_train/max(1,len(train)):.1%})  "
          f"valid={len(valid)} (think={skill_valid})")
    print(f"Output → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MineAgent SFT dataset")

    parser.add_argument("--trajectory-dir", "-T", type=str,
                        default="/root/autodl-tmp/MineAgent/trajectories",
                        help="Root directory containing trajectory subdirs")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="/root/autodl-tmp/MineAgent/sft_data",
                        help="Where to write train.jsonl / valid.jsonl")
    parser.add_argument("--valid-split", type=float, default=0.05,
                        help="Fraction of examples held out for validation (default 0.05)")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include unsuccessful trajectories")
    parser.add_argument("--skill-persist-steps", type=int, default=12,
                        help="Steps after a skill annotation that carry [Active Skill] context (default 12)")
    parser.add_argument("--seed", type=int, default=42)

    # ── Think-signal boosting ──────────────────────────────────────────────────
    parser.add_argument("--think-oversample", type=int, default=1,
                        help="Repeat think-annotated examples N times in the train set "
                             "to boost skill-reasoning signal (e.g. 8 → ~41%% think ratio)")
    parser.add_argument("--think-only", action="store_true",
                        help="Train only on think-annotated examples (ignores --think-oversample)")

    main(parser.parse_args())
