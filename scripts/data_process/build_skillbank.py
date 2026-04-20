"""
build_skillbank.py — Extract reusable skills from MineAgent trajectory data.

Loads a trajectory JSON + video, sends both to an expert LLM, and persists
the returned skill JSON to skillbank.json.  System prompt is read from
prompt/skillbank.md so it can be edited without touching this file.

Evolution flags (all off by default to keep test runs clean):
  --evo             Update usage_count / success_count for existing skills
                    that are relevant to the current trajectory's task type.
                    Applied to ALL complete trajectories, including failed ones.
  --prune           After processing, remove skills whose success rate is
                    below --min-success-rate (requires usage_count >= --prune-threshold).
  --no-novelty-check
                    Skip the novelty gate; add the skill even if it is
                    very similar to an existing one.

Usage:
    python scripts/build_skillbank.py \\
        --trajectory trajectories/kill_zombie-20260419_215315 \\
        --api-key sk-xxx

    # Evolution mode: track stats, novelty-filter new skills, prune weak ones
    python scripts/build_skillbank.py \\
        --trajectory-dir trajectories \\
        --api-key sk-xxx \\
        --evo --prune

    # Batch, skip novelty gate
    python scripts/build_skillbank.py \\
        --trajectory-dir trajectories \\
        --api-key sk-xxx \\
        --no-novelty-check
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

# utils/ lives at the project root (parent of scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich import print

from utils.api       import GeminiClient
from utils.parser    import (_summarise_action, _action_phase,
                              _non_empty, _first_nonempty_env, _inventory_snapshot)
from utils.skillbank import update_usage, prune_skills, is_novel


# ─────────────────────────────────────────────────────────────────────────────
# Skill bank I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_bank(path: Path) -> list:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_bank(bank: list, path: Path) -> None:
    path.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory summarisation
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(traj: dict) -> dict:
    meta  = traj["meta"]
    steps = traj["trajectory"]
    n     = len(steps)

    phase_counts: Counter       = Counter()
    action_combo_counts: Counter = Counter()
    cam_pitch_deltas, cam_yaw_deltas = [], []

    for s in steps:
        act = s["action"]
        phase_counts[_action_phase(act)] += 1
        action_combo_counts[_summarise_action(act)] += 1
        cam = act.get("camera", [0.0, 0.0])
        cam_pitch_deltas.append(cam[0])
        cam_yaw_deltas.append(cam[1])

    top_combos = [{"actions": combo, "count": cnt, "pct": round(cnt / n * 100, 1)}
                  for combo, cnt in action_combo_counts.most_common(10)]
    phase_dist = {phase: {"count": cnt, "pct": round(cnt / n * 100, 1)}
                  for phase, cnt in phase_counts.most_common()}
    cam_summary = {
        "mean_pitch_delta": round(sum(cam_pitch_deltas) / n, 3),
        "mean_yaw_delta":   round(sum(cam_yaw_deltas) / n, 3),
        "max_pitch_delta":  round(max(cam_pitch_deltas), 3),
        "max_yaw_delta":    round(max(cam_yaw_deltas), 3),
    }

    state_transitions = []
    prev_kill = prev_mine = prev_pickup = {}
    for s in steps:
        es = s["env_state"]
        kill   = es.get("kill_entity", {})
        mine   = es.get("mine_block",  {})
        pickup = es.get("pickup",      {})
        if kill   != prev_kill:
            state_transitions.append({"step": s["step"], "event": "kill_entity", "value": kill})
            prev_kill = dict(kill)
        if mine   != prev_mine:
            state_transitions.append({"step": s["step"], "event": "mine_block",  "value": mine})
            prev_mine = dict(mine)
        if pickup != prev_pickup:
            state_transitions.append({"step": s["step"], "event": "pickup",      "value": pickup})
            prev_pickup = dict(pickup)

    reward_steps  = [s["step"] for s in steps if s["reward"] > 0]
    first_attack  = next((s["step"] for s in steps if s["action"].get("attack",  0)), None)
    first_forward = next((s["step"] for s in steps if s["action"].get("forward", 0)), None)
    first_sprint  = next((s["step"] for s in steps if s["action"].get("sprint",  0)), None)

    milestones = {
        "first_forward_step": first_forward,
        "first_sprint_step":  first_sprint,
        "first_attack_step":  first_attack,
        "reward_steps":       reward_steps,
        "success_step":       reward_steps[0] if reward_steps else None,
    }

    sample_indices = set(range(0, n, max(1, n // 15)))
    sample_indices.update(reward_steps[:3])
    if first_attack is not None:
        sample_indices.add(first_attack)
    sampled_steps = []
    for s in steps:
        if s["step"] not in sample_indices:
            continue
        es = s["env_state"]
        sampled_steps.append({
            "step":           s["step"],
            "action_summary": _summarise_action(s["action"]),
            "camera_delta":   s["action"].get("camera", [0.0, 0.0]),
            "reward":         s["reward"],
            "kill_entity":    _non_empty(es.get("kill_entity", {})),
            "mine_block":     _non_empty(es.get("mine_block",  {})),
            "damage_dealt":   _non_empty(es.get("damage_dealt",{})),
            "use_item":       _non_empty(es.get("use_item",    {})),
            "location": {k: round(v, 2)
                         for k, v in es.get("location_stats", {}).items()
                         if k in ("xpos","ypos","zpos","yaw","pitch","light_level")},
        })

    start_inv   = _inventory_snapshot(steps[0]["env_state"]["inventory"])
    end_inv     = _inventory_snapshot(steps[-1]["env_state"]["inventory"])
    final_es    = steps[-1]["env_state"]

    return {
        "meta":               meta,
        "task_instruction":   steps[0]["instruction"],
        "total_steps":        n,
        "success":            meta["success"],
        "total_reward":       meta["total_reward"],
        "phase_distribution": phase_dist,
        "top_action_combos":  top_combos,
        "camera_summary":     cam_summary,
        "milestones":         milestones,
        "state_transitions":  state_transitions,
        "sampled_steps":      sampled_steps,
        "start_inventory":    start_inv,
        "end_inventory":      end_inv,
        "start_equipped":     steps[0]["env_state"].get("equipped_items", {}),
        "end_equipped":       steps[-1]["env_state"].get("equipped_items", {}),
        "cumulative_kill_entity":  _non_empty(final_es.get("kill_entity",  {})),
        "cumulative_mine_block":   _non_empty(final_es.get("mine_block",   {})),
        "cumulative_use_item":     _non_empty(final_es.get("use_item",     {})),
        "cumulative_pickup":       _non_empty(final_es.get("pickup",       {})),
        "cumulative_damage_dealt": _non_empty(final_es.get("damage_dealt", {})),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_FILE = Path(__file__).parent.parent.parent / "prompt" / "skillbank.md"


def _load_system_prompt() -> str:
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"System prompt not found: {_PROMPT_FILE}")


def build_user_prompt(summary: dict) -> str:
    return (
        "Below is the structured trajectory summary. "
        "Analyse it carefully and produce the skill JSON.\n\n"
        "```json\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n```\n\n"
        "Return only the skill JSON object, no other text."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def call_expert_llm(summary: dict, client: GeminiClient,
                    video_path: Optional[Path] = None) -> dict:
    system   = _load_system_prompt()
    user_msg = build_user_prompt(summary)
    raw = client.chat(system=system, user_text=user_msg,
                      video_path=video_path, temperature=0.2, max_tokens=2048)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def process_trajectory(
    traj_dir: Path,
    client: GeminiClient,
    skillbank_path: Path,
    evo: bool               = False,
    prune: bool             = False,
    prune_threshold: int    = 5,
    min_success_rate: float = 0.3,
    novelty_threshold: float = 0.65,
    no_novelty_check: bool  = False,
) -> Optional[dict]:
    """
    Process one trajectory directory.

    Steps (in order):
      1. Load trajectory.  If missing → skip.
      2. If --evo: update usage/success counts for relevant existing skills and save.
      3. Skip skill extraction for failed trajectories.
      4. Extract skill via LLM.
      5. Novelty gate: skip if too similar to an existing skill (unless --no-novelty-check).
      6. Append skill to bank (initialising usage_count / success_count = 0) and save.
      7. If --prune: prune under-performing skills and save.

    Returns the newly added skill dict, or None if nothing was added.
    """
    traj_file = traj_dir / "trajectory.json"
    if not traj_file.exists():
        print(f"[yellow]skip {traj_dir.name}: no trajectory.json[/yellow]")
        return None

    traj    = json.loads(traj_file.read_text(encoding="utf-8"))
    success = traj["meta"].get("success", False)
    task    = traj["trajectory"][0]["instruction"]

    bank = _load_bank(skillbank_path)

    # ── Step 2: usage tracking (all complete trajectories) ───────────────────
    if evo:
        n_updated = update_usage(bank, task, success)
        _save_bank(bank, skillbank_path)
        print(f"  [dim][evo] updated usage for {n_updated} skill(s) "
              f"(task={task}, success={success})[/dim]")

    # ── Step 3: only extract skills from successful trajectories ────────────
    if not success:
        print(f"[yellow]skip {traj_dir.name}: unsuccessful trajectory "
              f"(usage stats still updated if --evo)[/yellow]")
        return None

    print(f"[cyan]Processing {traj_dir.name} …[/cyan]")
    summary    = build_summary(traj)
    video_path = traj_dir / "video.mp4"

    # ── Step 4: LLM extraction ───────────────────────────────────────────────
    skill = call_expert_llm(summary, client, video_path=video_path)

    canonical = ["skill_id", "skill_name", "description",
                 "preconditions", "sub_tasks", "key_action_patterns"]
    skill = {k: skill[k] for k in canonical if k in skill}

    # Initialise tracking counters for the new skill
    skill["usage_count"]   = 0
    skill["success_count"] = 0

    # ── Step 5: novelty gate ─────────────────────────────────────────────────
    if not no_novelty_check:
        novel, max_sim = is_novel(skill, bank, max_similarity=novelty_threshold)
        if not novel:
            print(f"[yellow]  skill '{skill.get('skill_id')}' rejected: "
                  f"max similarity {max_sim:.2f} ≥ threshold {novelty_threshold} "
                  f"(use --no-novelty-check to force)[/yellow]")
            # Still fall through to pruning if requested
            skill = None
        else:
            print(f"  [dim]novelty OK (max_sim={max_sim:.2f})[/dim]")

    # ── Step 6: append to bank ───────────────────────────────────────────────
    if skill is not None:
        existing_ids = {s.get("skill_id") for s in bank}
        if skill.get("skill_id") in existing_ids:
            print(f"[yellow]  skill_id '{skill['skill_id']}' already in bank, skipping[/yellow]")
            skill = None
        else:
            bank.append(skill)
            _save_bank(bank, skillbank_path)
            print(f"[green]  skill '{skill.get('skill_id')}' appended "
                  f"→ {skillbank_path} ({len(bank)} total)[/green]")

    # ── Step 7: pruning ──────────────────────────────────────────────────────
    if prune:
        bank = _load_bank(skillbank_path)   # reload in case step 6 wrote
        pruned, removed_ids = prune_skills(bank, min_usage=prune_threshold,
                                            min_success_rate=min_success_rate)
        if removed_ids:
            _save_bank(pruned, skillbank_path)
            print(f"[magenta]  [prune] removed {len(removed_ids)} skill(s): "
                  f"{removed_ids}[/magenta]")
        else:
            print(f"  [dim][prune] no skills pruned "
                  f"(threshold={prune_threshold}, min_rate={min_success_rate})[/dim]")

    return skill


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    skillbank_path = Path(args.output)

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    client  = GeminiClient(host=args.host, api_key=api_key, model=args.model)

    traj_dirs: List[Path] = []
    if args.trajectory:
        traj_dirs = [Path(args.trajectory)]
    elif args.trajectory_dir:
        traj_dirs = sorted(Path(args.trajectory_dir).iterdir())

    if not traj_dirs:
        print("[red]Provide --trajectory or --trajectory-dir[/red]")
        sys.exit(1)

    added = 0
    for d in traj_dirs:
        if not d.is_dir():
            continue
        try:
            skill = process_trajectory(
                d, client, skillbank_path,
                evo              = args.evo,
                prune            = args.prune,
                prune_threshold  = args.prune_threshold,
                min_success_rate = args.min_success_rate,
                novelty_threshold = args.novelty_threshold,
                no_novelty_check  = args.no_novelty_check,
            )
            if skill:
                added += 1
        except Exception as e:
            print(f"[red]Error on {d.name}: {e}[/red]")

    print(f"\n[bold]Done — {added} new skill(s) added to {skillbank_path}[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MineAgent skill bank builder")

    # ── Source ────────────────────────────────────────────────────────────────
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--trajectory",     "-t", type=str,
                     help="Path to a single trajectory directory")
    src.add_argument("--trajectory-dir", "-T", type=str,
                     help="Directory containing multiple trajectory subdirectories")

    # ── API ───────────────────────────────────────────────────────────────────
    parser.add_argument("--host",    type=str, default="poloai.top")
    parser.add_argument("--model",   type=str, default="gemini-3-flash-preview")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output",  type=str,
                        default="/root/autodl-tmp/MineAgent/skillbank.json")

    # ── Evolution / tracking ──────────────────────────────────────────────────
    parser.add_argument(
        "--evo", action="store_true",
        help="Update usage_count / success_count for existing skills that match "
             "the trajectory task type.  Applied to ALL complete trajectories "
             "(including failed ones).  Off by default to keep test runs clean.",
    )

    # ── Novelty filtering ─────────────────────────────────────────────────────
    parser.add_argument(
        "--novelty-threshold", type=float, default=0.65, metavar="SIM",
        help="Maximum weighted-Jaccard similarity allowed for a new skill "
             "to be added (default: 0.65).  A new skill scoring ≥ this "
             "against any existing skill is rejected as non-novel.",
    )
    parser.add_argument(
        "--no-novelty-check", action="store_true",
        help="Disable the novelty gate; add the skill regardless of similarity.",
    )

    # ── Pruning ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--prune", action="store_true",
        help="After processing, remove skills whose success rate is below "
             "--min-success-rate (only when usage_count >= --prune-threshold).",
    )
    parser.add_argument(
        "--prune-threshold", type=int, default=5, metavar="N",
        help="Minimum usage_count before a skill is eligible for pruning "
             "(default: 5).",
    )
    parser.add_argument(
        "--min-success-rate", type=float, default=0.3, metavar="RATE",
        help="Skills with success_rate < RATE are pruned when they have "
             "enough usage data (default: 0.3).",
    )

    main(parser.parse_args())
