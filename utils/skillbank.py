"""
utils/skillbank.py — Skill bank management: novelty filtering, usage tracking, pruning.

Three capabilities:
  1. Novelty check  — reject new skills too similar to existing ones (weighted Jaccard).
  2. Usage tracking — record usage_count / success_count per skill across trajectories.
  3. Pruning        — remove skills with persistently low success rates.

Two skill types are supported:
  "success"   — positive skills describing what to do.
  "avoidance" — negative skills describing mistakes to avoid.
Skills of different types are never compared for novelty.
"""

import re
from typing import List, Tuple


# ── Internal tokeniser (self-contained; mirrors utils/rag.py) ─────────────────

def _tok(text: str) -> set:
    return set(re.sub(r"[_:,=().\[\]{}]", " ", text.lower()).split())


def _field_tokens(skill: dict, field: str) -> set:
    val = skill.get(field, "")
    if isinstance(val, list):
        val = " ".join(str(x) for x in val)
    return _tok(str(val))


# ── Similarity ────────────────────────────────────────────────────────────────

_SIM_FIELDS_SUCCESS = [
    ("skill_name",           1.0),
    ("description",          3.0),
    ("preconditions",        3.0),
    ("sub_tasks",            2.0),
    ("key_action_patterns",  2.0),
]

_SIM_FIELDS_AVOIDANCE = [
    ("skill_name",           1.0),
    ("description",          3.0),
    ("preconditions",        3.0),
    ("common_mistakes",      2.0),
    ("corrective_actions",   2.0),
]


def _sim_fields_for(skill: dict) -> List[Tuple[str, float]]:
    if skill.get("skill_type") == "avoidance":
        return _SIM_FIELDS_AVOIDANCE
    return _SIM_FIELDS_SUCCESS


def skill_similarity(s1: dict, s2: dict) -> float:
    """
    Weighted Jaccard similarity between two skill records across key text fields.
    Returns a value in [0, 1]; 1 means all fields are identical.
    Skills of different types always return 0.0 (not comparable).
    Both-empty fields contribute full weight (treated as identical).
    One-empty fields contribute nothing.
    """
    if s1.get("skill_type", "success") != s2.get("skill_type", "success"):
        return 0.0
    fields       = _sim_fields_for(s1)
    total_weight = sum(w for _, w in fields)
    score        = 0.0
    for field, weight in fields:
        t1 = _field_tokens(s1, field)
        t2 = _field_tokens(s2, field)
        if not t1 and not t2:
            score += weight
        elif t1 and t2:
            score += weight * len(t1 & t2) / len(t1 | t2)
    return score / total_weight


def is_novel(new_skill: dict, bank: List[dict],
             max_similarity: float = 0.65) -> Tuple[bool, float]:
    """
    Check whether new_skill is sufficiently different from every skill in bank.

    Only compares against skills of the same type.

    Returns (novel: bool, max_sim: float).
      novel=True  → the skill should be added.
      novel=False → it is too similar to an existing skill; skip addition.
    max_sim is the highest similarity score found (for logging).
    """
    max_sim = 0.0
    for existing in bank:
        sim = skill_similarity(new_skill, existing)
        if sim > max_sim:
            max_sim = sim
        if sim >= max_similarity:
            return False, max_sim
    return True, max_sim


# ── Usage tracking ────────────────────────────────────────────────────────────

_TASK_KEYWORDS: dict = {
    "kill_entity": {"kill", "combat", "attack", "melee", "hostile", "mob"},
    "mine_block":  {"mine", "mining", "block", "ore", "pickaxe"},
    "craft_item":  {"craft", "crafting", "recipe", "table"},
    "smelt_item":  {"smelt", "smelting", "furnace", "coal"},
}


def update_usage(bank: List[dict], task_instruction: str,
                 success: bool) -> int:
    """
    For every skill in bank that is relevant to task_instruction, increment:
      usage_count   by 1 unconditionally.
      success_count by 1 only when success=True.

    Relevance: the task-type keyword set (derived from the task prefix, e.g.
    "kill_entity") has at least one token in common with the skill's
    name + description + preconditions text.

    Returns the number of skills updated.
    New fields are initialised to 0 if not already present.
    """
    task_type = task_instruction.split(":")[0] if ":" in task_instruction else task_instruction
    keywords  = _TASK_KEYWORDS.get(task_type, set())
    if not keywords:
        return 0

    updated = 0
    for skill in bank:
        text = " ".join([
            skill.get("skill_name",    ""),
            skill.get("description",   ""),
            skill.get("preconditions", ""),
        ])
        if keywords & _tok(text):
            skill["usage_count"]   = skill.get("usage_count",   0) + 1
            if success:
                skill["success_count"] = skill.get("success_count", 0) + 1
            updated += 1
    return updated


# ── Pruning ───────────────────────────────────────────────────────────────────

def prune_skills(bank: List[dict], min_usage: int = 5,
                 min_success_rate: float = 0.3) -> Tuple[List[dict], List[str]]:
    """
    Remove under-performing skills from bank.

    A skill is pruned when BOTH conditions hold:
      usage_count   >= min_usage          (enough data to judge)
      success_rate   < min_success_rate   (too many failures)

    Skills with insufficient usage data are always kept.

    Returns (kept_bank, list_of_removed_skill_ids).
    """
    kept: List[dict] = []
    removed_ids: List[str] = []

    for skill in bank:
        usage = skill.get("usage_count",   0)
        succ  = skill.get("success_count", 0)
        if usage >= min_usage and (succ / usage) < min_success_rate:
            removed_ids.append(skill.get("skill_id", "?"))
        else:
            kept.append(skill)

    return kept, removed_ids
