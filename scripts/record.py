"""
record.py — MineAgent trajectory recorder

Per-step captures:
  - POV frame (PNG)
  - Model raw LLM output string
  - Decoded action dict
  - Full env state (inventory, location, event counters, etc.)
  - Reward

Saves:
  trajectories/{model}-{task}-{timestamp}/
      trajectory.json   — full step-by-step data
      frames/00000.png  — per-step POV images
      video.mp4         — assembled video

Usage:
    python scripts/record.py \\
        --env-config kill/kill_zombie \\
        --base-url http://localhost:8000/v1
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import hydra
import numpy as np
from rich import print

# ── Make JarvisVLA importable ─────────────────────────────────────────────────
sys.path.insert(0, "/root/autodl-tmp/JarvisVLA")

from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.callbacks import (
    CommandsCallback,
    FastResetCallback,
    InitInventoryCallback,
    RewardsCallback,
    SpeedTestCallback,
    SummonMobsCallback,
    TaskCallback,
)
from jarvisvla.evaluate import agent_wrapper
from jarvisvla.evaluate.env_helper.craft_agent import CraftWorker
from jarvisvla.evaluate.env_helper.smelt_agent import SmeltWorker


# ─────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_python(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


# info keys that carry meaningful state (exclude pov which is saved as PNG)
_INFO_KEYS = [
    "inventory",
    "isGuiOpen",
    "location_stats",
    "kill_entity",
    "mine_block",
    "craft_item",
    "damage_dealt",
    "pickup",
    "use_item",
    "equipped_items",
]


def _extract_env_state(info: dict) -> dict:
    return {k: _to_python(info[k]) for k in _INFO_KEYS if k in info}


# ─────────────────────────────────────────────────────────────────────────────
# Recording agent — captures raw LLM output alongside each action
# ─────────────────────────────────────────────────────────────────────────────

class RecordingAgent(agent_wrapper.VLLM_AGENT):
    """Wraps VLLM_AGENT to expose per-step raw model output and cache status."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_raw_output: str = ""        # raw LLM token string
        self.last_action_tokens: Any = None   # token ids before decoding
        self.last_from_cache: bool = False    # True when action came from chunk cache

    def forward(self, observations, instructions, verbos=False, need_crafting_table=False):
        # ── Return from chunk cache without calling the API ──────────────────
        if self.actions:
            self.last_from_cache = True
            if len(self.actions) > 1:
                return self.actions.pop(0)
            else:
                action = self.actions[0]
                self.actions = []
                return action

        # ── API call ─────────────────────────────────────────────────────────
        self.last_from_cache = False
        messages = []
        image = self.processor_wrapper.create_image_input(observations[0])
        method = self.method_map[need_crafting_table]
        private_instruction = self.create_instruction(instructions[0], method=method)
        thought = self.create_thought(instructions[0]) if self.instruction_type == "recipe" else ""

        if self.history_num:
            if not self.history:
                null_tok = self.action_tokenizer.null_token()
                self.history = [(image, null_tok, copy.copy(thought), 0)] * self.history_num
            new_history = [None] * self.history_num
            new_history[:-1] = self.history[1:]
            for hdx, (im, ac, past_thought, _) in enumerate(self.history):
                if self.instruction_type == "recipe":
                    prompt_input = "\nthought: " + past_thought + "\nobservation: "
                else:
                    prompt_input = "\nobservation: "
                if not hdx:
                    prompt_input = private_instruction + prompt_input
                messages.append(self.processor_wrapper.create_message_vllm(
                    role="user", input_type="image", prompt=[prompt_input], image=[im]))
                messages.append(self.processor_wrapper.create_message_vllm(
                    role="assistant", input_type="text", prompt=[ac]))

        if self.instruction_type == "recipe":
            prompt_input = "\nthought: " + thought + "\nobservation: "
        else:
            prompt_input = "\nobservation: "
        if not self.history_num:
            prompt_input = private_instruction + prompt_input
        messages.append(self.processor_wrapper.create_message_vllm(
            role="user", input_type="image", prompt=[prompt_input], image=[image]))

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
            top_p=0.99,
            logprobs=False,
            extra_body={"skip_special_tokens": False, "top_k": -1},
        )

        raw_output = chat_completion.choices[0].message.content
        self.last_raw_output = raw_output

        if self.history_num:
            new_history[-1] = (image, raw_output, thought, self.history[-1][-1] + 1)
            self.history = new_history

        if self.LLM_backbone in {"qwen2_vl"}:
            token_ids = self.tokenizer(raw_output)["input_ids"]
        else:
            token_ids = raw_output
        self.last_action_tokens = token_ids

        actions = self.action_tokenizer.decode(token_ids)
        len_action = min(self.action_chunk_len, len(actions))
        self.actions = list(actions[:len_action])
        return self.actions.pop(0)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory recorder — writes frames and JSON
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryRecorder:

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.steps: List[dict] = []

    def record(
        self,
        step: int,
        pov: np.ndarray,
        instruction: str,
        raw_output: str,
        from_cache: bool,
        action_tokens: Any,
        action: dict,
        reward: float,
        info: dict,
    ) -> None:
        frame_name = f"{step:05d}.png"
        cv2.imwrite(
            str(self.frames_dir / frame_name),
            cv2.cvtColor(pov.astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
        self.steps.append({
            "step": step,
            "timestamp": time.time(),
            "frame": frame_name,
            "instruction": instruction,
            # model outputs
            "model_raw_output": raw_output,
            "model_from_cache": from_cache,    # True = repeated from chunk, no API call
            "action_tokens": _to_python(action_tokens),
            "action": _to_python(action),
            # environment feedback
            "reward": float(reward),
            "env_state": _extract_env_state(info),
        })

    def save_json(self, meta: dict) -> str:
        path = self.output_dir / "trajectory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "trajectory": self.steps}, f, indent=2, ensure_ascii=False)
        print(f"[green]trajectory.json → {path}[/green]")
        return str(path)

    def save_video(self, fps: int = 20) -> Optional[str]:
        frame_files = sorted(self.frames_dir.glob("*.png"))
        if not frame_files:
            return None
        sample = cv2.imread(str(frame_files[0]))
        h, w = sample.shape[:2]
        path = self.output_dir / "video.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fp in frame_files:
            writer.write(cv2.imread(str(fp)))
        writer.release()
        print(f"[green]video.mp4       → {path}[/green]")
        return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Core record function
# ─────────────────────────────────────────────────────────────────────────────

def record(args):
    # ── Load Hydra config ─────────────────────────────────────────────────────
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    cfg_path = Path(f"{args.env_config}.yaml")
    abs_dir = str(Path(args.config_base) / cfg_path.parent)
    if os.path.isabs(abs_dir):
        hydra.initialize_config_dir(config_dir=abs_dir, version_base="1.3")
    else:
        hydra.initialize(config_path=abs_dir, version_base="1.3")
    cfg = hydra.compose(config_name=cfg_path.stem)

    # ── Output directory ──────────────────────────────────────────────────────
    model_tag = Path(args.checkpoints).name
    task_tag  = args.env_config.replace("/", "_")
    run_tag   = f"{model_tag}-{task_tag}-{time.strftime('%Y%m%d_%H%M%S')}"
    recorder  = TrajectoryRecorder(str(Path(args.output_dir) / run_tag))
    print(f"[cyan]Output dir: {recorder.output_dir}[/cyan]")

    # ── Build environment ─────────────────────────────────────────────────────
    camera_cfg = CameraConfig(**cfg.camera_config)
    callbacks = [
        FastResetCallback(
            biomes=cfg.candidate_preferred_spawn_biome,
            random_tp_range=cfg.random_tp_range,
            start_time=cfg.start_time,
        ),
        SpeedTestCallback(50),
        TaskCallback(getattr(cfg, "task_conf", None)),
        RewardsCallback(getattr(cfg, "reward_conf", None)),
        InitInventoryCallback(
            cfg.init_inventory,
            distraction_level=cfg.inventory_distraction_level,
        ),
        CommandsCallback(getattr(cfg, "command", [])),
    ]
    if cfg.mobs:
        callbacks.append(SummonMobsCallback(cfg.mobs))

    env = MinecraftSim(
        action_type="env",
        seed=cfg.seed,
        obs_size=cfg.origin_resolution,
        render_size=cfg.resize_resolution,
        camera_config=camera_cfg,
        preferred_spawn_biome=getattr(cfg, "preferred_spawn_biome", None),
        callbacks=callbacks,
    )
    obs, info = env.reset()

    # ── Optional GUI pre-setup (craft/smelt tasks) ────────────────────────────
    pre_agent = None
    need_crafting_table = False
    worker_type = getattr(cfg, "worker", None)
    if worker_type == "craft":
        pre_agent = CraftWorker(env, if_discrete=True)
    elif worker_type == "smelt":
        pre_agent = SmeltWorker(env, if_discrete=True)

    if getattr(cfg, "need_gui", False):
        need_crafting_table = getattr(cfg, "need_crafting_table", False)
        need_furnace = getattr(cfg, "need_furnace", False)
        try:
            if need_crafting_table:
                pre_agent.open_crating_table_wo_recipe()
            elif need_furnace:
                pre_agent.open_furnace_wo_recipe()
            else:
                pre_agent._null_action(1)
                if not pre_agent.info["isGuiOpen"]:
                    pre_agent._call_func("inventory")
        except AssertionError as e:
            env.close()
            print(f"[red]GUI setup failed: {e}[/red]")
            return False

    env.action_type = "agent"

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = RecordingAgent(
        checkpoint_path=args.checkpoints,
        base_url=args.base_url,
        temperature=args.temperature,
        history_num=args.history_num,
        instruction_type=args.instruction_type,
        action_chunk_len=args.action_chunk_len,
    )
    instructions = [item["text"] for item in cfg.task_conf]
    instruction_str = instructions[0]
    print(f"[cyan]Task: {instruction_str} | max_frames={args.max_frames}[/cyan]")

    # ── Step loop ─────────────────────────────────────────────────────────────
    success = False
    total_reward = 0.0
    step = 0

    for step in range(args.max_frames):
        pov = info["pov"].copy()
        action = agent.forward([pov], instructions, need_crafting_table=need_crafting_table)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        recorder.record(
            step=step,
            pov=pov,
            instruction=instruction_str,
            raw_output=agent.last_raw_output,
            from_cache=agent.last_from_cache,
            action_tokens=agent.last_action_tokens,
            action=action,
            reward=reward,
            info=info,
        )

        if reward > 0:
            success = True
            print(f"[green]✓ Success at step {step}[/green]")
            # record extra steps after success for richer data
            for extra in range(1, args.extra_steps + 1):
                pov = info["pov"].copy()
                action = agent.forward([pov], instructions, need_crafting_table=need_crafting_table)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                recorder.record(
                    step=step + extra,
                    pov=pov,
                    instruction=instruction_str,
                    raw_output=agent.last_raw_output,
                    from_cache=agent.last_from_cache,
                    action_tokens=agent.last_action_tokens,
                    action=action,
                    reward=reward,
                    info=info,
                )
            break

    env.close()

    # ── Persist ───────────────────────────────────────────────────────────────
    meta = {
        "task":            args.env_config,
        "model":           args.checkpoints,
        "success":         success,
        "total_steps":     len(recorder.steps),
        "total_reward":    total_reward,
        "instruction":     instruction_str,
        "record_args":     vars(args),
    }
    recorder.save_json(meta)
    recorder.save_video(fps=args.fps)
    print(f"[bold]Done — success={success}, steps={len(recorder.steps)}, reward={total_reward}[/bold]")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MineAgent trajectory recorder")

    # task / model
    parser.add_argument("--env-config", "-e",   type=str, default="kill/kill_zombie")
    parser.add_argument("--checkpoints",         type=str,
                        default="/root/autodl-tmp/MineAgent/models/jarvis_vla_qwen2_vl_7b_sft")
    parser.add_argument("--base-url",            type=str, default="http://localhost:8000/v1")
    parser.add_argument("--config-base",         type=str,
                        default="/root/autodl-tmp/MineAgent/config")

    # recording
    parser.add_argument("--output-dir",          type=str,
                        default="/root/autodl-tmp/MineAgent/trajectories")
    parser.add_argument("--max-frames",          type=int,   default=300)
    parser.add_argument("--extra-steps",         type=int,   default=20,
                        help="Steps to record after task success")
    parser.add_argument("--fps",                 type=int,   default=20)

    # agent
    parser.add_argument("--temperature",         type=float, default=0.9)
    parser.add_argument("--history-num",         type=int,   default=2)
    parser.add_argument("--action-chunk-len",    type=int,   default=1)
    parser.add_argument("--instruction-type",    type=str,   default="normal")

    args = parser.parse_args()
    record(args)
