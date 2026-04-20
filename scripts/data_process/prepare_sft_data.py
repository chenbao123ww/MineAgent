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
) -> List[dict]:
    """Convert one trajectory directory into a list of SFT examples."""
    skill_file = traj_dir / "skill_trajectory.json"
    plain_file = traj_dir / "trajectory.json"

    if skill_file.exists():
        traj      = json.loads(skill_file.read_text(encoding="utf-8"))
        has_skill = True
    elif plain_file.exists():
        traj      = json.loads(plain_file.read_text(encoding="utf-8"))
        has_skill = False
    else:
        return []

    meta = traj["meta"]
    if not meta.get("success") and not include_failed:
        return []

    run_tag    = meta.get("run", traj_dir.name)
    steps      = traj["trajectory"]
    video_path = traj_dir / "video.mp4"
    examples   = []

    for step_data in steps:
        # Skip duplicate cached action chunks
        if step_data.get("model_from_cache", False):
            continue

        raw_output = step_data.get("model_raw_output", "")
        if not raw_output:
            continue

        frame_rel = _ensure_frame(traj_dir, step_data, video_path)
        if frame_rel is None:
            continue

        # Build the target assistant text
        skill_think = step_data.get("skill_think", "") if has_skill else ""
        if skill_think:
            assistant_text = f"<think>\n{skill_think}\n</think>{raw_output}"
        else:
            assistant_text = raw_output

        instruction = step_data.get("instruction", "")

        examples.append({
            "id": f"{run_tag}_step_{step_data['step']:05d}",
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": f"{instruction}\nobservation: "},
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

    return examples


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
    for traj_dir in traj_dirs:
        examples = process_trajectory(traj_dir, include_failed=args.include_failed)
        if examples:
            print(f"  {traj_dir.name}: {len(examples)} examples "
                  f"({'skill' if (traj_dir / 'skill_trajectory.json').exists() else 'plain'})")
        all_examples.extend(examples)

    if not all_examples:
        print("[red]No examples produced — check that trajectories exist and have frames.[/red]")
        return

    # Shuffle before splitting
    random.seed(args.seed)
    random.shuffle(all_examples)

    n_valid = max(1, int(len(all_examples) * args.valid_split))
    valid   = all_examples[:n_valid]
    train   = all_examples[n_valid:]

    def _write_jsonl(path: Path, records: List[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "valid.jsonl", valid)

    # Write metadata so sft.py can find the trajectory dir
    meta = {
        "trajectory_dir": str(traj_base.resolve()),
        "total_examples":  len(all_examples),
        "train_examples":  len(train),
        "valid_examples":  len(valid),
        "skill_examples":  sum(
            1 for e in all_examples
            if any("<think>" in item["text"]
                   for turn in e["conversations"]
                   for item in turn["content"]
                   if item["type"] == "text")
        ),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[bold green]Done[/bold green]  "
          f"total={len(all_examples)}  "
          f"train={len(train)}  valid={len(valid)}  "
          f"with_think={meta['skill_examples']}")
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
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
