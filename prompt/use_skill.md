You are generating skill-annotated Minecraft VQA training data.

You receive:
  1. A video of a Minecraft agent (one video frame = one trajectory step, 0-indexed).
  2. A step-by-step trajectory log: step index, action, equipped item, position, events.
  3. A "skill_context" block retrieved via RAG based on skill preconditions:
       [skill_id] Name
         Description   : what the skill achieves
         Preconditions : what must be true BEFORE applying this skill
         Sub-tasks     : ordered execution steps
         Key actions   : concrete button/camera patterns

YOUR TASK:
  Watch the video and read the trajectory log together.
  Select the steps that most deserve skill-based chain-of-thought annotation.
  Aim for roughly 15–25% of total steps.  Prioritise:
    • The moment skill preconditions first become satisfied
    • Phase transitions (searching → approaching → attacking → post-event)
    • Steps immediately before/after a significant event (kill, mine, craft, reward)
    • Periodic coverage (~every 30 steps) when nothing notable is happening

For each selected step generate:
  "obs"  : ONE sentence — perceptual facts visible in that video frame +
            the single most relevant game-state fact from the log.
  "think": 2–3 sentences structured EXACTLY as:
             (a) Precondition check: which preconditions are now met / not yet met
             (b) [skill_id] + which sub-task is currently active
             (c) Immediate next action and why (cite key_action_patterns)

OUTPUT: a JSON array and NOTHING else (no markdown, no prose):
[
  {"step": <int>, "obs": "...", "think": "...", "skill_ids": ["skill_id"]},
  ...
]
Sorted by step ascending.
