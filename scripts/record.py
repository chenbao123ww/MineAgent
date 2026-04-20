"""
record.py — MineAgent trajectory recorder

Per-step captures:
  - POV frame (PNG)
  - Model raw LLM output string
  - Decoded action dict
  - Full env state (inventory, location, event counters, etc.)
  - Reward

Saves:
  trajectories/{task}-{timestamp}/
      trajectory.json   — full step-by-step data
      frames/00000.png  — per-step POV images  (if --save-frames)
      video.mp4         — assembled video

Usage:
    python scripts/record.py \\
        --env-config kill/kill_zombie \\
        --base-url http://localhost:8000/v1
"""

import argparse
import sys
import time
from pathlib import Path

import hydra
from rich import print

# ── Project root on sys.path (utils/ lives there) ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
from jarvisvla.evaluate.env_helper.craft_agent import CraftWorker
from jarvisvla.evaluate.env_helper.smelt_agent import SmeltWorker

from utils.agent import RecordingAgent, TrajectoryRecorder


# ─────────────────────────────────────────────────────────────────────────────
# Core record function
# ─────────────────────────────────────────────────────────────────────────────

def record(args):
    # ── Load Hydra config ─────────────────────────────────────────────────────
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    cfg_path = Path(f"{args.env_config}.yaml")
    abs_dir  = str(Path(args.config_base).resolve() / cfg_path.parent)
    hydra.initialize_config_dir(config_dir=abs_dir, version_base="1.3")
    cfg = hydra.compose(config_name=cfg_path.stem)

    # ── Output directory ──────────────────────────────────────────────────────
    task_tag = Path(args.env_config).name
    run_tag  = f"{task_tag}-{time.strftime('%Y%m%d_%H%M%S')}"
    recorder = TrajectoryRecorder(
        str(Path(args.output_dir) / run_tag),
        fps=args.fps,
        save_frames=args.save_frames,
    )
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
        CommandsCallback(getattr(cfg, "command", [])),
        InitInventoryCallback(
            cfg.init_inventory,
            distraction_level=cfg.inventory_distraction_level,
        ),
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
        need_furnace        = getattr(cfg, "need_furnace", False)
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
    instructions    = [item["text"] for item in cfg.task_conf]
    instruction_str = instructions[0]

    # ── Step loop ─────────────────────────────────────────────────────────────
    success      = False
    total_reward = 0.0

    for step in range(args.max_frames):
        pov    = info["pov"].copy()
        action = agent.forward([pov], instructions,
                               need_crafting_table=need_crafting_table)
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
            for extra in range(1, args.extra_steps + 1):
                pov    = info["pov"].copy()
                action = agent.forward([pov], instructions,
                                       need_crafting_table=need_crafting_table)
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
        "run":          run_tag,
        "success":      success,
        "total_steps":  len(recorder.steps),
        "total_reward": total_reward,
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

    parser.add_argument("--env-config",   "-e", type=str,
                        default="base/kill/kill_zombie")
    parser.add_argument("--checkpoints",        type=str,
                        default="/root/autodl-tmp/MineAgent/models/jarvis_vla_qwen2_vl_7b_sft")
    parser.add_argument("--base-url",           type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--config-base",        type=str,
                        default="/root/autodl-tmp/MineAgent/config")

    parser.add_argument("--output-dir",         type=str,
                        default="/root/autodl-tmp/MineAgent/trajectories")
    parser.add_argument("--max-frames",         type=int,   default=1000)
    parser.add_argument("--extra-steps",        type=int,   default=20)
    parser.add_argument("--fps",                type=int,   default=20)
    parser.add_argument("--save-frames",        action="store_true")

    parser.add_argument("--temperature",        type=float, default=0.9)
    parser.add_argument("--history-num",        type=int,   default=2)
    parser.add_argument("--action-chunk-len",   type=int,   default=1)
    parser.add_argument("--instruction-type",   type=str,   default="normal")

    args = parser.parse_args()
    record(args)
