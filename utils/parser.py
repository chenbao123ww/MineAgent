"""
utils/parser.py — Action and trajectory parsing helpers.

Shared by build_skillbank.py and gen_skill_vqa.py.
Covers: action summarisation, phase classification, env-state description,
and compact trajectory-log formatting.
"""

from typing import Dict, List, Optional


# ── Action helpers ────────────────────────────────────────────────────────────

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
    """Convert an action dict to a compact human-readable string."""
    active = [k for k in _BINARY_ACTIONS if action.get(k, 0)]
    cam    = action.get("camera", [0.0, 0.0])
    parts  = active[:]
    if abs(cam[0]) > 0.5 or abs(cam[1]) > 0.5:
        pitch_dir = "look_down" if cam[0] >  0.5 else ("look_up"    if cam[0] < -0.5 else "")
        yaw_dir   = "turn_right" if cam[1] >  0.5 else ("turn_left" if cam[1] < -0.5 else "")
        parts += [d for d in [pitch_dir, yaw_dir] if d]
    hb = _active_hotbar(action)
    if hb:
        parts.append(f"hotbar.{hb}")
    return "+".join(parts) if parts else "no_op"


def _action_phase(action: dict) -> str:
    """Classify an action as one of: attack, scan, approach, orient, idle."""
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


# ── Env-state helpers ─────────────────────────────────────────────────────────

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


def describe_env_state(es: dict) -> str:
    """Convert an env_state dict to a human-readable one-liner."""
    parts = []

    inv = es.get("inventory", {}) or {}
    items = []
    for slot, info in inv.items():
        if isinstance(info, dict) and info.get("type", "none") != "none":
            items.append(f"{info.get('quantity', 1)}x {info['type']}")
    if items:
        parts.append("Inventory: " + ", ".join(items[:6]))

    equip = es.get("equipped_items", {}) or {}
    mh = equip.get("mainhand", {}) or {}
    if mh.get("type", "air") not in ("air", "none", ""):
        parts.append(f"Equipped: {mh['type']}")

    for key, label in [("kill_entity", "kills"), ("mine_block", "mined"),
                        ("craft_item", "crafted")]:
        val = {k: v for k, v in (es.get(key) or {}).items() if v}
        if val:
            parts.append(f"{label}={val}")

    dmg = {k: v for k, v in (es.get("damage_dealt") or {}).items() if v}
    if dmg:
        parts.append(f"damage_dealt={dmg}")

    loc = es.get("location_stats", {}) or {}
    if loc:
        parts.append(
            f"pos=({loc.get('xpos', 0):.0f},{loc.get('ypos', 0):.0f},{loc.get('zpos', 0):.0f})"
            f" yaw={loc.get('yaw', 0):.0f}"
        )

    return "; ".join(parts) if parts else "No notable state."


def build_traj_summary(steps: List[dict]) -> str:
    """
    Compact per-step text log for LLM context.
    Format: step=N  action=...  eq=...  [events]  pos=(x,y,z)
    """
    lines = []
    for s in steps:
        action   = _summarise_action(s.get("action", {}))
        es       = s.get("env_state", {})
        equip    = (es.get("equipped_items", {}) or {}).get("mainhand", {}) or {}
        equipped = equip.get("type", "air")
        reward   = s.get("reward", 0.0)
        kill     = {k: v for k, v in (es.get("kill_entity") or {}).items() if v}
        mine     = {k: v for k, v in (es.get("mine_block") or {}).items() if v}
        craft    = {k: v for k, v in (es.get("craft_item") or {}).items() if v}
        loc      = es.get("location_stats", {}) or {}

        row = [f"step={s['step']:>4d}", f"action={action:<30s}", f"eq={equipped}"]
        if reward:  row.append(f"R={reward:.1f}")
        if kill:    row.append(f"kills={kill}")
        if mine:    row.append(f"mined={mine}")
        if craft:   row.append(f"crafted={craft}")
        if loc:
            row.append(
                f"pos=({loc.get('xpos', 0):.0f},"
                f"{loc.get('ypos', 0):.0f},"
                f"{loc.get('zpos', 0):.0f})"
            )
        lines.append("  ".join(row))
    return "\n".join(lines)
