"""
utils/rag.py — Skill-bank RAG retrieval helpers.

Provides lightweight keyword-overlap retrieval driven by skill *preconditions*,
plus formatting utilities for injecting skill context into LLM prompts.

Two skill types are handled:
  "success"   — positive skills describing what to do.
  "avoidance" — negative skills describing mistakes to avoid.
Each type is retrieved and formatted independently so both always appear in context.
"""

import re
from typing import Dict, List


def _tokenise(text: str) -> set:
    """Lower-case word tokens, stripping common punctuation."""
    return set(re.sub(r"[_:,=().\[\]{}]", " ", text.lower()).split())


def _score_skills(
    skills: List[dict],
    task_tokens: set,
    hint_tokens: set,
    state_tokens: set,
) -> List[dict]:
    """Score and sort a list of skills by relevance; return sorted list."""
    scored = []
    for skill in skills:
        full_text      = " ".join([skill.get("skill_name",    ""),
                                    skill.get("description",   ""),
                                    skill.get("preconditions", "")])
        skill_tokens   = _tokenise(full_text)
        precond_tokens = _tokenise(skill.get("preconditions", ""))
        desc_tokens    = _tokenise(skill.get("description",   ""))

        score = (
            len((task_tokens | hint_tokens) & skill_tokens) * 2
            + len(state_tokens & precond_tokens) * 3
            + len(state_tokens & desc_tokens)
        )
        scored.append((score, skill))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored]


def retrieve_skills_rag(
    env_state_text: str,
    task: str,
    skillbank: list,
    max_success: int = 2,
    max_avoidance: int = 2,
) -> List[dict]:
    """
    Score each skill by how well its *preconditions* match the current state.
    Returns up to max_success success skills and max_avoidance avoidance skills,
    combined into a single list (success first, avoidance second).

    Scoring (additive):
      task_overlap    × 2  — task tokens ∩ (name + desc + precond)
      precond_overlap × 3  — state tokens ∩ preconditions tokens  ← main RAG signal
      desc_overlap    × 1  — state tokens ∩ description tokens
    """
    task_tokens  = _tokenise(task)
    state_tokens = _tokenise(env_state_text)

    _TYPE_HINTS: Dict[str, set] = {
        "kill_entity": {"combat", "attack", "melee", "hostile", "mob", "weapon"},
        "mine_block":  {"mine", "mining", "pickaxe", "ore", "block"},
        "craft_item":  {"craft", "crafting", "table", "recipe", "wood"},
        "smelt_item":  {"smelt", "smelting", "furnace", "ore", "coal"},
    }
    task_type   = task.split(":")[0] if ":" in task else task
    hint_tokens = _TYPE_HINTS.get(task_type, set())

    success_pool   = [s for s in skillbank if s.get("skill_type", "success") == "success"]
    avoidance_pool = [s for s in skillbank if s.get("skill_type") == "avoidance"]

    result  = _score_skills(success_pool,   task_tokens, hint_tokens, state_tokens)[:max_success]
    result += _score_skills(avoidance_pool, task_tokens, hint_tokens, state_tokens)[:max_avoidance]
    return result


def precond_check_text(env_state_text: str, skills: List[dict]) -> str:
    """
    Returns a short human-readable summary of which precondition keywords
    appear in the current env_state_text, per skill.
    """
    lines = []
    state_tokens = _tokenise(env_state_text)
    for s in skills:
        precond = s.get("preconditions", "")
        if not precond:
            continue
        matched = [w for w in _tokenise(precond) if len(w) > 3 and w in state_tokens]
        pct  = int(100 * len(matched) / max(1, len(_tokenise(precond))))
        tag  = f"[{s.get('skill_type', 'success')}]"
        lines.append(
            f"  [{s['skill_id']}]{tag} ~{pct}% met  "
            f"(keywords: {', '.join(matched[:6]) or 'none'})"
        )
    return "\n".join(lines) or "  (none)"


def format_skills_header(skills: List[dict]) -> str:
    """Format skills as '[id] Name; [id] Name; ...' for episode-level meta."""
    return "; ".join(f"[{s['skill_id']}] {s['skill_name']}" for s in skills)


def format_skills_for_llm(skills: List[dict]) -> str:
    """
    Full RAG context block split into SUCCESS and AVOIDANCE sections.
    Each section only includes the fields relevant to that skill type.
    """
    success_skills   = [s for s in skills if s.get("skill_type", "success") == "success"]
    avoidance_skills = [s for s in skills if s.get("skill_type") == "avoidance"]

    lines = []

    if success_skills:
        lines.append("=== SUCCESS SKILLS — what to do ===")
        for s in success_skills:
            lines.append(f"[{s['skill_id']}] {s['skill_name']}")
            lines.append(f"  Description   : {s.get('description', '')}")
            lines.append(f"  Preconditions : {s.get('preconditions', '')}")
            sub = s.get("sub_tasks", [])
            if sub:
                lines.append("  Sub-tasks     : " + " → ".join(sub))
            kap = s.get("key_action_patterns", [])
            if kap:
                lines.append("  Key actions   : " + "; ".join(kap))
            lines.append("")

    if avoidance_skills:
        lines.append("=== AVOIDANCE SKILLS — what NOT to do ===")
        for s in avoidance_skills:
            lines.append(f"[{s['skill_id']}] {s['skill_name']}")
            lines.append(f"  Description        : {s.get('description', '')}")
            lines.append(f"  Preconditions      : {s.get('preconditions', '')}")
            cm = s.get("common_mistakes", [])
            if cm:
                lines.append("  Common mistakes    : " + "; ".join(cm))
            ca = s.get("corrective_actions", [])
            if ca:
                lines.append("  Corrective actions : " + "; ".join(ca))
            lines.append("")

    return "\n".join(lines).rstrip()
