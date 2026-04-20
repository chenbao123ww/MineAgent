"""
gen_skill_vqa.py — Annotate MineAgent trajectories with skill-based chain-of-thought.

Architecture (video-first):
  1. Send ONE request: inline video + full trajectory log + RAG-retrieved skill context.
  2. Gemini watches the video, reads the log, and returns which steps need
     skill-based annotations + the obs/think content for each.
  3. RAG retrieval is driven by skill *preconditions* matched against the
     current env_state at each candidate step.
  4. The returned annotations are injected into trajectory.json →
     skill_trajectory.json.

Output format per step (added fields):
  "skill_think"     : "" | "[skill_id] <reasoning chain>"
  "skill_trigger"   : "" | reason string (from LLM)
  "retrieved_skills": [] | [skill_id, ...]

Meta block gets:
  "skills_header"         : "[id] Name; ..."
  "skills"                : [{full skill record}, ...]
  "skill_annotated_steps" : N

Usage:
    python scripts/gen_skill_vqa.py \\
        --trajectory trajectories/kill_zombie-20260420_010903 \\
        --skillbank  skillbank.json \\
        --api-key    sk-xxx \\
        --include-failed
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# utils/ lives at the project root (parent of scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich import print

from utils.api    import GeminiClient
from utils.rag    import (retrieve_skills_rag, precond_check_text,
                           format_skills_header, format_skills_for_llm)
from utils.parser import describe_env_state, build_traj_summary


# ─────────────────────────────────────────────────────────────────────────────
# Video-level annotation — single LLM call for the whole episode
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_FILE = Path(__file__).parent.parent.parent / "prompt" / "use_skill.md"


def _load_system_prompt() -> str:
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"System prompt not found: {_PROMPT_FILE}")


def annotate_with_video(
    client: GeminiClient,
    traj_dir: Path,
    steps: List[dict],
    task: str,
    skillbank: list,
) -> Tuple[List[dict], List[dict]]:
    """
    Build trajectory summary + RAG skill context, then call Gemini once with
    the inline video + text to get annotations for all selected steps.

    Returns (annotations, skills):
      annotations — [{step, obs, think, skill_ids}, ...]
      skills      — list of retrieved skill dicts
    """
    video_path = traj_dir / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"No video.mp4 in {traj_dir}")

    init_state = describe_env_state(steps[0].get("env_state", {}))
    skills     = retrieve_skills_rag(init_state, task, skillbank)
    if not skills:
        raise ValueError(f"No skills matched for task '{task}'")

    user_msg = (
        f"TASK: {task}\n"
        f"TOTAL STEPS: {len(steps)}\n\n"
        f"=== SKILL CONTEXT (RAG from preconditions) ===\n"
        f"{format_skills_for_llm(skills)}\n\n"
        f"=== INITIAL PRECONDITION CHECK ===\n"
        f"{precond_check_text(init_state, skills)}\n\n"
        f"=== TRAJECTORY LOG (video frame index = step index) ===\n"
        f"{build_traj_summary(steps)}\n\n"
        "Watch the video — each video frame corresponds to the step index above. "
        "Generate skill-based annotations as described in the system prompt."
    )

    sz_kb = video_path.stat().st_size // 1024
    print(f"  [dim]sending video inline ({sz_kb} KB) + trajectory to {client.model}…[/dim]")
    raw = client.chat_with_video(
        system     = _load_system_prompt(),
        user_text  = user_msg,
        video_path = video_path,
        temperature = 0.2,
        max_tokens  = 8192,
    ).strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    annotations = json.loads(raw)
    if not isinstance(annotations, list):
        raise ValueError(f"Expected JSON array, got: {type(annotations)}")

    return annotations, skills


# ─────────────────────────────────────────────────────────────────────────────
# Episode processing
# ─────────────────────────────────────────────────────────────────────────────

def process_trajectory(
    traj_dir: Path,
    skillbank: list,
    client: GeminiClient,
    include_failed: bool = False,
) -> Optional[int]:
    """
    Reads trajectory.json, annotates via video, writes skill_trajectory.json.
    Returns number of annotated steps, or None if skipped.
    """
    traj_file = traj_dir / "trajectory.json"
    if not traj_file.exists():
        print(f"[yellow]skip {traj_dir.name}: no trajectory.json[/yellow]")
        return None

    traj = json.loads(traj_file.read_text(encoding="utf-8"))
    if not traj["meta"].get("success") and not include_failed:
        print(f"[yellow]skip {traj_dir.name}: unsuccessful run (use --include-failed)[/yellow]")
        return None

    steps = traj["trajectory"]
    task  = steps[0]["instruction"]
    print(f"[cyan]{traj_dir.name}[/cyan]: {len(steps)} steps | task={task}")

    try:
        annotations, traj_skills = annotate_with_video(
            client, traj_dir, steps, task, skillbank
        )
    except Exception as e:
        print(f"[red]  Video annotation failed: {e}[/red]")
        return None

    anno_by_step: Dict[int, dict] = {a["step"]: a for a in annotations}
    print(f"  LLM selected {len(anno_by_step)} steps to annotate")

    all_skills_seen: Dict[str, dict] = {s["skill_id"]: s for s in traj_skills}
    n_annotated    = 0
    enriched_steps = []

    for step_data in steps:
        new_step = dict(step_data)
        new_step["skill_think"]      = ""
        new_step["skill_trigger"]    = ""
        new_step["retrieved_skills"] = []

        anno = anno_by_step.get(step_data["step"])
        if anno:
            think     = anno.get("think", "")
            skill_ids = anno.get("skill_ids", [])
            new_step["skill_think"]      = think
            new_step["skill_trigger"]    = "llm_selected"
            new_step["retrieved_skills"] = skill_ids
            for sid in skill_ids:
                for s in traj_skills:
                    if s["skill_id"] == sid:
                        all_skills_seen[sid] = s
            n_annotated += 1
            print(f"  step {step_data['step']:>4d}  skills={skill_ids}  {think[:90]}")

        enriched_steps.append(new_step)

    skills_in_meta = list(all_skills_seen.values())
    skill_traj = {
        "meta": {
            **traj["meta"],
            "skills_header":         format_skills_header(skills_in_meta),
            "skills": [
                {
                    "skill_id":            s["skill_id"],
                    "skill_name":          s["skill_name"],
                    "description":         s.get("description",         ""),
                    "preconditions":       s.get("preconditions",        ""),
                    "sub_tasks":           s.get("sub_tasks",            []),
                    "key_action_patterns": s.get("key_action_patterns",  []),
                }
                for s in skills_in_meta
            ],
            "skill_annotated_steps": n_annotated,
        },
        "trajectory": enriched_steps,
    }

    out_path = traj_dir / "skill_trajectory.json"
    out_path.write_text(
        json.dumps(skill_traj, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[green]  → {out_path}  ({n_annotated} annotated steps)[/green]")
    return n_annotated


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    skillbank_path = Path(args.skillbank)
    if not skillbank_path.exists():
        print(f"[red]skillbank not found: {skillbank_path}[/red]")
        sys.exit(1)
    skillbank = json.loads(skillbank_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(skillbank)} skill(s) from {skillbank_path}")

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    client  = GeminiClient(host=args.host, api_key=api_key, model=args.model)

    traj_dirs: List[Path] = []
    if args.trajectory:
        traj_dirs = [Path(args.trajectory)]
    elif args.trajectory_dir:
        traj_dirs = sorted(p for p in Path(args.trajectory_dir).iterdir() if p.is_dir())
    if not traj_dirs:
        print("[red]Provide --trajectory or --trajectory-dir[/red]")
        sys.exit(1)

    n_episodes, n_steps = 0, 0
    for traj_dir in traj_dirs:
        try:
            result = process_trajectory(
                traj_dir, skillbank, client,
                include_failed=args.include_failed,
            )
            if result is not None:
                n_episodes += 1
                n_steps    += result
        except Exception as e:
            print(f"[red]Error on {traj_dir.name}: {e}[/red]")

    print(f"\n[bold green]Done — {n_episodes} episode(s), {n_steps} annotated steps[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MineAgent skill VQA data generator")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--trajectory",     "-t", type=str,
                     help="Single trajectory directory")
    src.add_argument("--trajectory-dir", "-T", type=str,
                     help="Directory of trajectory subdirectories")

    parser.add_argument("--skillbank",   "-s", type=str,
                        default="/root/autodl-tmp/MineAgent/skillbank.json")
    parser.add_argument("--api-key",           type=str, default=None)
    parser.add_argument("--host",              type=str, default="poloai.top")
    parser.add_argument("--model",             type=str,
                        default="gemini-3-flash-preview")
    parser.add_argument("--include-failed",    action="store_true",
                        help="Also annotate unsuccessful trajectories")

    main(parser.parse_args())
