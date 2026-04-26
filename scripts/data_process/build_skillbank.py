"""
build_skillbank.py — Extract reusable skills from MineAgent trajectory data.

Groups trajectories by task category (kill / mine / craft / smelt), aggregates
statistics across all trajectories in each group, then calls the LLM once per
category to produce 3–5 representative skills.  Also produces a 'general'
category synthesised from all trajectories combined.

Usage:
    python scripts/data_process/build_skillbank.py \\
        --trajectory-dir trajectories \\
        --api-key sk-xxx

    # Single trajectory (still runs in category mode for that one traj)
    python scripts/data_process/build_skillbank.py \\
        --trajectory trajectories/kill_zombie-... \\
        --api-key sk-xxx

    # Skip novelty gate
    python scripts/data_process/build_skillbank.py \\
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
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich import print

from utils.api       import GeminiClient
from utils.parser    import (_summarise_action, _action_phase,
                              _non_empty, _inventory_snapshot)
from utils.skillbank import is_novel, update_usage, prune_skills


# ─────────────────────────────────────────────────────────────────────────────
# Skill bank I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_bank(path: Path) -> list:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_bank(bank: list, path: Path) -> None:
    path.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Category detection
# ─────────────────────────────────────────────────────────────────────────────

_PREFIX_TO_CATEGORY: Dict[str, str] = {
    "kill_entity": "kill",
    "mine_block":  "mine",
    "craft_item":  "craft",
    "smelt_item":  "smelt",
}


def detect_category(traj: dict) -> str:
    instruction = traj["trajectory"][0]["instruction"]
    prefix = instruction.split(":")[0] if ":" in instruction else instruction
    return _PREFIX_TO_CATEGORY.get(prefix, "general")


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory summary (unchanged stats extraction)
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(traj: dict) -> dict:
    meta  = traj["meta"]
    steps = traj["trajectory"]
    n     = len(steps)

    phase_counts: Counter        = Counter()
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

    state_transitions = []
    prev_kill = prev_mine = prev_pickup = {}
    for s in steps:
        es     = s["env_state"]
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

    sample_indices = set(range(0, n, max(1, n // 10)))
    sample_indices.update(reward_steps[:2])
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
            "reward":         s["reward"],
            "kill_entity":    _non_empty(es.get("kill_entity", {})),
            "mine_block":     _non_empty(es.get("mine_block",  {})),
        })

    start_inv = _inventory_snapshot(steps[0]["env_state"]["inventory"])
    end_inv   = _inventory_snapshot(steps[-1]["env_state"]["inventory"])
    final_es  = steps[-1]["env_state"]

    return {
        "meta":               meta,
        "task_instruction":   steps[0]["instruction"],
        "total_steps":        n,
        "success":            meta["success"],
        "phase_distribution": phase_dist,
        "top_action_combos":  top_combos,
        "milestones":         milestones,
        "state_transitions":  state_transitions[:12],
        "sampled_steps":      sampled_steps,
        "start_inventory":    start_inv,
        "end_inventory":      end_inv,
        "cumulative_kill_entity":  _non_empty(final_es.get("kill_entity",  {})),
        "cumulative_mine_block":   _non_empty(final_es.get("mine_block",   {})),
        "cumulative_pickup":       _non_empty(final_es.get("pickup",       {})),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Category-level aggregate summary
# ─────────────────────────────────────────────────────────────────────────────

# Type alias: (traj_dir, summary_dict, success_bool)
TrajTriple = Tuple[Path, dict, bool]


def build_category_summary(category: str, triples: List[TrajTriple]) -> dict:
    """Aggregate multiple trajectory summaries into one category-level summary."""
    summaries = [s for _, s, _ in triples]

    # Unique targets
    targets = sorted({
        s["task_instruction"].split(":")[-1] if ":" in s["task_instruction"]
        else s["task_instruction"]
        for s in summaries
    })

    success_count = sum(1 for _, _, ok in triples if ok)

    # Aggregate action combos
    combo_agg: Counter = Counter()
    for s in summaries:
        for c in s["top_action_combos"]:
            combo_agg[c["actions"]] += c["count"]

    # Aggregate phase distribution
    phase_agg: Counter = Counter()
    for s in summaries:
        for phase, stats in s["phase_distribution"].items():
            phase_agg[phase] += stats["count"]
    total_steps = sum(phase_agg.values()) or 1
    phase_dist = {
        p: {"count": n, "pct": round(n / total_steps * 100, 1)}
        for p, n in phase_agg.most_common()
    }

    # Select representative sample trajectories (prefer successful, keep ≤4)
    successful = [(d, s) for d, s, ok in triples if ok]
    failed     = [(d, s) for d, s, ok in triples if not ok]
    selected   = (successful + failed)[:4]

    return {
        "category":           category,
        "task_targets":       targets,
        "total_trajectories": len(triples),
        "success_rate":       round(success_count / len(triples), 2),
        "phase_distribution": phase_dist,
        "top_action_combos":  [{"actions": a, "count": c}
                                for a, c in combo_agg.most_common(8)],
        "sample_trajectories": [s for _, s in selected],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_FILE = Path(__file__).parent.parent.parent / "prompt" / "skillbank.md"


def _load_system_prompt() -> str:
    return _PROMPT_FILE.read_text(encoding="utf-8").strip()


def call_expert_llm(category_summary: dict, client: GeminiClient) -> List[dict]:
    """Call LLM with category-level summary. Returns list of 3–5 skill dicts."""
    system   = _load_system_prompt()
    user_msg = (
        "Below is the category-level trajectory summary. "
        "Analyse it and produce exactly 3–5 skill JSON records.\n\n"
        "```json\n"
        + json.dumps(category_summary, indent=2, ensure_ascii=False)
        + "\n```\n\n"
        "Return only the JSON array, no other text."
    )
    raw = client.chat(system=system, user_text=user_msg,
                      temperature=0.2, max_tokens=2048)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        parsed = [parsed]
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Category processing
# ─────────────────────────────────────────────────────────────────────────────

def _add_skills_to_bank(
    skills: List[dict],
    skillbank_path: Path,
    novelty_threshold: float,
    no_novelty_check: bool,
) -> int:
    """Persist a list of skill dicts to the bank. Returns count added."""
    added = 0
    for skill in skills:
        skill.setdefault("skill_type", "success")
        skill["usage_count"]   = 0
        skill["success_count"] = 0

        bank = _load_bank(skillbank_path)

        if not no_novelty_check:
            novel, max_sim = is_novel(skill, bank, max_similarity=novelty_threshold)
            if not novel:
                print(f"  [yellow]skip '{skill.get('skill_id')}': "
                      f"similarity {max_sim:.2f} ≥ {novelty_threshold}[/yellow]")
                continue

        if any(s["skill_id"] == skill.get("skill_id") for s in bank):
            print(f"  [yellow]skip '{skill.get('skill_id')}': id already in bank[/yellow]")
            continue

        bank.append(skill)
        _save_bank(bank, skillbank_path)
        added += 1
        print(f"  [green]+'{skill.get('skill_id')}' [{skill.get('skill_type')}] "
              f"→ {skillbank_path.name} ({len(bank)} total)[/green]")

    return added


def process_all_categories(
    traj_dirs: List[Path],
    client: GeminiClient,
    skillbank_path: Path,
    novelty_threshold: float = 0.65,
    no_novelty_check: bool   = False,
) -> int:
    """
    Main entry point for category-based skill extraction.

    1. Load and classify all trajectories by category.
    2. For each category: build aggregate summary → LLM → 3-5 skills.
    3. For 'general': aggregate across all trajectories → LLM → 3-5 skills.
    Returns total skills added.
    """
    category_triples: Dict[str, List[TrajTriple]] = {}
    all_triples: List[TrajTriple] = []

    for traj_dir in traj_dirs:
        if not traj_dir.is_dir():
            continue
        traj_file = traj_dir / "trajectory.json"
        if not traj_file.exists():
            print(f"[yellow]skip {traj_dir.name}: no trajectory.json[/yellow]")
            continue
        try:
            traj    = json.loads(traj_file.read_text(encoding="utf-8"))
            summary = build_summary(traj)
            success = traj["meta"].get("success", False)
            cat     = detect_category(traj)
        except Exception as e:
            print(f"[red]skip {traj_dir.name}: {e}[/red]")
            continue

        category_triples.setdefault(cat, []).append((traj_dir, summary, success))
        all_triples.append((traj_dir, summary, success))

    print(f"\nFound {len(all_triples)} valid trajectories across "
          f"{len(category_triples)} categories: {list(category_triples.keys())}")

    total_added = 0

    # ── Per-category skills ───────────────────────────────────────────────────
    for cat, triples in sorted(category_triples.items()):
        n_ok = sum(1 for _, _, ok in triples if ok)
        print(f"\n[bold cyan]── Category: {cat.upper()} "
              f"({len(triples)} traj, {n_ok} success) ──[/bold cyan]")

        cat_summary = build_category_summary(cat, triples)
        try:
            skills = call_expert_llm(cat_summary, client)
        except Exception as e:
            print(f"[red]  LLM failed for '{cat}': {e}[/red]")
            continue

        print(f"  LLM returned {len(skills)} skill(s)")
        n = _add_skills_to_bank(skills, skillbank_path,
                                 novelty_threshold, no_novelty_check)
        total_added += n
        print(f"  → {n} added for category '{cat}'")

    # ── General cross-task skills ─────────────────────────────────────────────
    if len(category_triples) > 1:
        print(f"\n[bold cyan]── Category: GENERAL (all {len(all_triples)} traj) ──[/bold cyan]")
        gen_summary = build_category_summary("general", all_triples)
        gen_summary["note"] = ("Cross-task skills applicable regardless of task type: "
                               "navigation, combat readiness, resource efficiency.")
        try:
            skills = call_expert_llm(gen_summary, client)
        except Exception as e:
            print(f"[red]  LLM failed for 'general': {e}[/red]")
            skills = []

        print(f"  LLM returned {len(skills)} skill(s)")
        n = _add_skills_to_bank(skills, skillbank_path,
                                 novelty_threshold, no_novelty_check)
        total_added += n
        print(f"  → {n} added for category 'general'")

    return total_added


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

    added = process_all_categories(
        traj_dirs,
        client,
        skillbank_path,
        novelty_threshold = args.novelty_threshold,
        no_novelty_check  = args.no_novelty_check,
    )

    bank = _load_bank(skillbank_path)

    # ── Evo: update usage/success counts per trajectory ───────────────────────
    if args.evo:
        print("\n[bold cyan]── Evo: updating usage counts ──[/bold cyan]")
        evo_updated = 0
        for traj_dir in traj_dirs:
            traj_file = traj_dir / "trajectory.json"
            if not traj_file.exists():
                continue
            try:
                traj        = json.loads(traj_file.read_text(encoding="utf-8"))
                instruction = traj["trajectory"][0]["instruction"]
                success     = traj["meta"].get("success", False)
                n           = update_usage(bank, instruction, success)
                evo_updated += n
            except Exception as e:
                print(f"[yellow]  evo skip {traj_dir.name}: {e}[/yellow]")
        _save_bank(bank, skillbank_path)
        print(f"  Updated {evo_updated} usage record(s) across {len(bank)} skills.")

    # ── Prune: remove low-success-rate skills ─────────────────────────────────
    if args.prune:
        kept, removed = prune_skills(bank)
        if removed:
            _save_bank(kept, skillbank_path)
            bank = kept
            print(f"\n[yellow]Pruned {len(removed)} skill(s): {removed}[/yellow]")
        else:
            print("\n[green]Prune: no skills removed.[/green]")

    print(f"\n[bold green]Done — {added} new skill(s) added. "
          f"Bank total: {len(bank)} skill(s).[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MineAgent skill bank builder")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--trajectory",     "-t", type=str,
                     help="Path to a single trajectory directory")
    src.add_argument("--trajectory-dir", "-T", type=str,
                     help="Directory containing multiple trajectory subdirectories")

    parser.add_argument("--host",    type=str, default="poloai.top")
    parser.add_argument("--model",   type=str, default="gemini-3-flash-preview")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output",  type=str,
                        default="/root/autodl-tmp/MineAgent/skillbank.json")
    parser.add_argument("--novelty-threshold", type=float, default=0.65)
    parser.add_argument("--no-novelty-check",  action="store_true")
    parser.add_argument("--evo",               action="store_true",
                        help="Update usage_count/success_count per skill after extraction")
    parser.add_argument("--prune",             action="store_true",
                        help="Remove skills below success-rate threshold after evo update")

    main(parser.parse_args())
