"""
build_skillbank.py — extract reusable skills from MineAgent trajectory data

Loads a trajectory JSON + video, sends both to an expert LLM, and persists
the returned skill JSON to skillbank/.  System prompt is read from
prompt/prompt.md so it can be edited without touching this file.

Usage:
    python scripts/build_skillbank.py \\
        --trajectory trajectories/kill_zombie-20260419_215315 \\
        --api-key sk-xxx

    # batch: process all successful runs under trajectories/
    python scripts/build_skillbank.py --trajectory-dir trajectories --api-key sk-xxx
"""

import argparse
import base64
import http.client
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from rich import print


# ─────────────────────────────────────────────────────────────────────────────
# Gemini-format API client  (poloai.top relay, supports inline video)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiClient:
    def __init__(self, host: str = "poloai.top", api_key: str = "",
                 model: str = "gemini-3-flash-preview"):
        self.host    = host
        self.api_key = api_key
        self.model   = model

    def chat(self, system: str, user_text: str,
             video_path: Optional[Path] = None,
             temperature: float = 0.2, max_tokens: int = 2048) -> str:
        parts = [{"text": user_text}]
        if video_path and video_path.exists():
            b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
            parts.append({"inline_data": {"mime_type": "video/mp4", "data": b64}})

        payload = json.dumps({
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }, ensure_ascii=False)

        conn = http.client.HTTPSConnection(self.host)
        conn.request("POST", f"/v1beta/models/{self.model}",
                     payload.encode("utf-8"),
                     headers={"Accept": "application/json",
                               "Authorization": self.api_key,
                               "Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected API response: {data}") from e

# ─────────────────────────────────────────────────────────────────────────────
# Trajectory summarisation
# ─────────────────────────────────────────────────────────────────────────────

# Actions that are binary (0/1); camera is handled separately
_BINARY_ACTIONS = [
    "attack", "forward", "back", "left", "right",
    "jump", "sneak", "sprint", "use",
]

def _active_hotbar(action: dict) -> Optional[int]:
    for i in range(1, 10):
        if action.get(f"hotbar.{i}", 0):
            return i
    return None


def _summarise_action(action: dict) -> str:
    active = [k for k in _BINARY_ACTIONS if action.get(k, 0)]
    cam = action.get("camera", [0.0, 0.0])
    parts = active[:]
    if abs(cam[0]) > 0.5 or abs(cam[1]) > 0.5:
        pitch_dir = "look_down" if cam[0] > 0.5 else ("look_up" if cam[0] < -0.5 else "")
        yaw_dir   = "turn_right" if cam[1] > 0.5 else ("turn_left" if cam[1] < -0.5 else "")
        parts += [d for d in [pitch_dir, yaw_dir] if d]
    hb = _active_hotbar(action)
    if hb:
        parts.append(f"hotbar.{hb}")
    return "+".join(parts) if parts else "no_op"


def _action_phase(action: dict) -> str:
    if action.get("attack", 0):
        return "attack"
    cam = action.get("camera", [0.0, 0.0])
    if abs(cam[1]) > 2.0 and not action.get("forward", 0):
        return "scan"
    if action.get("forward", 0):
        return "approach"
    if abs(cam[1]) > 0.5 or abs(cam[0]) > 0.5:
        return "orient"
    return "idle"


def _non_empty(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in ({}, [], None, 0, "")}


def _first_nonempty_env(steps: list, key: str):
    for s in steps:
        val = s["env_state"].get(key, {})
        if val:
            return val
    return {}


def _inventory_snapshot(inv: dict) -> dict:
    return {slot: {"type": item["type"], "qty": item["quantity"]}
            for slot, item in inv.items()
            if item["type"] != "none"}


def build_summary(traj: dict) -> dict:
    meta   = traj["meta"]
    steps  = traj["trajectory"]
    n      = len(steps)

    # ── Action statistics ────────────────────────────────────────────────────
    phase_counts: Counter = Counter()
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

    # ── State transitions ─────────────────────────────────────────────────────
    state_transitions = []
    prev_kill = {}
    prev_mine = {}
    prev_pickup = {}

    for s in steps:
        es = s["env_state"]
        kill   = es.get("kill_entity", {})
        mine   = es.get("mine_block", {})
        pickup = es.get("pickup", {})
        if kill != prev_kill:
            state_transitions.append({"step": s["step"], "event": "kill_entity", "value": kill})
            prev_kill = dict(kill)
        if mine != prev_mine:
            state_transitions.append({"step": s["step"], "event": "mine_block", "value": mine})
            prev_mine = dict(mine)
        if pickup != prev_pickup:
            state_transitions.append({"step": s["step"], "event": "pickup", "value": pickup})
            prev_pickup = dict(pickup)

    # ── Key milestone steps (reward steps, first attack, etc.) ───────────────
    reward_steps = [s["step"] for s in steps if s["reward"] > 0]
    first_attack = next((s["step"] for s in steps if s["action"].get("attack", 0)), None)
    first_forward = next((s["step"] for s in steps if s["action"].get("forward", 0)), None)
    first_sprint = next((s["step"] for s in steps if s["action"].get("sprint", 0)), None)

    milestones = {
        "first_forward_step": first_forward,
        "first_sprint_step":  first_sprint,
        "first_attack_step":  first_attack,
        "reward_steps":       reward_steps,
        "success_step":       reward_steps[0] if reward_steps else None,
    }

    # ── Sampled steps (every ~20 steps + reward step) ────────────────────────
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
            "step": s["step"],
            "action_summary": _summarise_action(s["action"]),
            "camera_delta": s["action"].get("camera", [0.0, 0.0]),
            "reward": s["reward"],
            "kill_entity": _non_empty(es.get("kill_entity", {})),
            "mine_block": _non_empty(es.get("mine_block", {})),
            "damage_dealt": _non_empty(es.get("damage_dealt", {})),
            "use_item": _non_empty(es.get("use_item", {})),
            "location": {k: round(v, 2) for k, v in es.get("location_stats", {}).items()
                         if k in ("xpos", "ypos", "zpos", "yaw", "pitch", "light_level")},
        })

    # ── Inventory & equipment ─────────────────────────────────────────────────
    start_inv  = _inventory_snapshot(steps[0]["env_state"]["inventory"])
    end_inv    = _inventory_snapshot(steps[-1]["env_state"]["inventory"])
    start_equip = steps[0]["env_state"].get("equipped_items", {})
    end_equip   = steps[-1]["env_state"].get("equipped_items", {})

    # ── Cumulative env at end ─────────────────────────────────────────────────
    final_es = steps[-1]["env_state"]

    return {
        "meta": meta,
        "task_instruction": steps[0]["instruction"],
        "total_steps": n,
        "success": meta["success"],
        "total_reward": meta["total_reward"],

        "phase_distribution": phase_dist,
        "top_action_combos": top_combos,
        "camera_summary": cam_summary,
        "milestones": milestones,
        "state_transitions": state_transitions,
        "sampled_steps": sampled_steps,

        "start_inventory": start_inv,
        "end_inventory": end_inv,
        "start_equipped": start_equip,
        "end_equipped": end_equip,

        "cumulative_kill_entity": _non_empty(final_es.get("kill_entity", {})),
        "cumulative_mine_block":  _non_empty(final_es.get("mine_block", {})),
        "cumulative_use_item":    _non_empty(final_es.get("use_item", {})),
        "cumulative_pickup":      _non_empty(final_es.get("pickup", {})),
        "cumulative_damage_dealt":_non_empty(final_es.get("damage_dealt", {})),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_FILE = Path(__file__).parent.parent / "prompt" / "prompt.md"

def _load_system_prompt() -> str:
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"System prompt not found: {_PROMPT_FILE}")


# kept for reference — actual prompt lives in prompt/prompt.md
SYSTEM_PROMPT = """You are an expert Minecraft AI researcher specializing in skill extraction and knowledge distillation.

Your task: given a structured summary of a Minecraft agent trajectory, extract a reusable **skill** that captures the core knowledge demonstrated in this trajectory. The skill will be stored in a skill bank and later retrieved to help a Vision-Language Model (VLM) reason about how to accomplish Minecraft tasks.

## Trajectory Summary Fields Explained

- **task_instruction**: The goal the agent was given. Format is `event_type:target` (e.g., `kill_entity:zombie`, `mine_block:iron_ore`).
- **total_steps**: How many simulator steps the episode took.
- **success** / **total_reward**: Whether the agent completed the task.
- **phase_distribution**: Fraction of steps spent in each behavior phase:
  - `attack` — agent was pressing attack (melee)
  - `approach` — moving forward toward target
  - `scan` — rotating camera to search/orient (no forward movement)
  - `orient` — small camera adjustments
  - `idle` — no significant action
- **top_action_combos**: Most frequent discrete action combinations, with counts and percentages. Action names: `attack`, `forward`, `back`, `left`, `right`, `jump`, `sneak`, `sprint`, `use`, `hotbar.N`, `turn_left/right`, `look_up/down`.
- **camera_summary**: Statistics on camera (view direction) changes per step. `mean_yaw_delta` > 0 means generally turning right. Camera uses first-person pitch/yaw deltas.
- **milestones**: Key step indices — when the agent first moved, first attacked, and when reward was received.
- **state_transitions**: Moments when observable game events changed (kills, block mines, item pickups), with the step number.
- **sampled_steps**: A representative subset of steps showing: `action_summary` (active actions), `camera_delta` [pitch_delta, yaw_delta], `reward`, and cumulative event counters at that step.
- **start/end_inventory**: Items in the 36 inventory slots (hotbar = slots 0–8) at episode start and end.
- **start/end_equipped**: Equipment in armor slots + mainhand/offhand at start and end. `damage` / `maxDamage` tracks item wear.
- **cumulative_***: Total kills, blocks mined, items used/picked up across the full episode.

## Output Format

Return a single JSON object (no markdown fences) matching this exact schema:

```
{
  "skill_id": "<snake_case_unique_id>",
  "skill_name": "<Short Human-Readable Name>",
  "task_type": "<combat | mining | crafting | smelting | exploration | building>",
  "applicable_tasks": ["task_instruction strings this skill applies to"],
  "description": "<1-2 sentence description of what the skill accomplishes and when to use it>",
  "preconditions": {
    "required_items": ["items the agent needs before starting (slot 0 = first hotbar slot)"],
    "environment": ["relevant environmental conditions: time of day, biome, mob presence, etc."],
    "player_state": ["relevant player state: equipped items, health considerations, etc."]
  },
  "sub_tasks": [
    {
      "order": 1,
      "name": "<sub-task name>",
      "goal": "<what to accomplish>",
      "key_actions": {"<action>": "<value or guidance>"},
      "visual_cues": ["<what the VLM should look for in the POV image to know this sub-task is active or done>"],
      "completion_criterion": "<how to know this sub-task is finished>"
    }
  ],
  "key_action_patterns": [
    "<reusable action pattern, e.g.: sprint forward while centering target in view>"
  ],
  "camera_strategy": "<guidance on how to move the camera for this task>",
  "success_indicators": {
    "env_state_signals": ["<env_state fields that signal success, e.g. kill_entity.zombie increments>"],
    "visual_signals": ["<visual observations in POV that indicate progress or completion>"]
  },
  "failure_modes": [
    {"mode": "<failure scenario>", "recovery": "<how to recover>"}
  ],
  "vlm_reasoning_hints": [
    "<high-level reasoning tip for the VLM, written as if advising the model in real time>"
  ],
  "efficiency_insights": "<what made this successful trajectory effective — key decisions or patterns>",
  "extracted_from": {
    "trajectory": "<run_tag>",
    "success": true,
    "total_steps": 0,
    "task": "<task_instruction>"
  }
}
```

Be specific and actionable. The VLM will read this skill and use it to decide what actions to take step-by-step. Avoid vague advice."""


def build_user_prompt(summary: dict) -> str:
    return (
        "Below is the structured trajectory summary. Analyse it carefully and produce the skill JSON.\n\n"
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
    system  = _load_system_prompt()
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

def process_trajectory(traj_dir: Path, client: GeminiClient, skillbank_path: Path) -> Optional[dict]:
    traj_file = traj_dir / "trajectory.json"
    if not traj_file.exists():
        print(f"[yellow]skip {traj_dir.name}: no trajectory.json[/yellow]")
        return None

    traj = json.loads(traj_file.read_text(encoding="utf-8"))
    if not traj["meta"].get("success"):
        print(f"[yellow]skip {traj_dir.name}: not a successful trajectory[/yellow]")
        return None

    print(f"[cyan]Processing {traj_dir.name} …[/cyan]")
    summary   = build_summary(traj)
    video_path = traj_dir / "video.mp4"

    skill = call_expert_llm(summary, client, video_path=video_path)

    # Keep only the six canonical fields
    canonical = ["skill_id", "skill_name", "description",
                 "preconditions", "sub_tasks", "key_action_patterns"]
    skill = {k: skill[k] for k in canonical if k in skill}

    # Append to skillbank.json (create if absent, skip duplicate skill_ids)
    bank: list = []
    if skillbank_path.exists():
        bank = json.loads(skillbank_path.read_text(encoding="utf-8"))
    existing_ids = {s.get("skill_id") for s in bank}
    if skill.get("skill_id") in existing_ids:
        print(f"[yellow]skill_id '{skill['skill_id']}' already in bank, skipping[/yellow]")
        return None
    bank.append(skill)
    skillbank_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[green]skill '{skill.get('skill_id')}' appended → {skillbank_path} ({len(bank)} total)[/green]")
    return skill


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
        if d.is_dir():
            try:
                skill = process_trajectory(d, client, skillbank_path)
                if skill:
                    added += 1
            except Exception as e:
                print(f"[red]Error on {d.name}: {e}[/red]")

    print(f"\n[bold]Done — {added} new skill(s) added to {skillbank_path}[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MineAgent skill bank builder")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--trajectory",     "-t", type=str,
                     help="Path to a single trajectory directory")
    src.add_argument("--trajectory-dir", "-T", type=str,
                     help="Directory containing multiple trajectory subdirectories")

    parser.add_argument("--host",      type=str, default="poloai.top",
                        help="API relay host")
    parser.add_argument("--model",     type=str, default="gemini-3-flash-preview",
                        help="Expert model name")
    parser.add_argument("--api-key",   type=str, default=None,
                        help="API key (falls back to GEMINI_API_KEY env var)")
    parser.add_argument("--output",    type=str,
                        default="/root/autodl-tmp/MineAgent/skillbank.json",
                        help="Path to the skillbank JSON file (appended each run)")

    main(parser.parse_args())
