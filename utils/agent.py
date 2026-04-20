"""
utils/agent.py — RecordingAgent and TrajectoryRecorder for MineAgent.

RecordingAgent wraps VLLM_AGENT (JarvisVLA) and exposes per-step raw model
output, action tokens, and cache status.

TrajectoryRecorder writes per-step PNG frames, assembles video.mp4 via
imageio_ffmpeg, and persists trajectory.json.
"""

import copy
import json
import time
import sys
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from rich import print

sys.path.insert(0, "/root/autodl-tmp/JarvisVLA")
from jarvisvla.evaluate import agent_wrapper


# ── Serialisation helpers ─────────────────────────────────────────────────────

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


# ── Recording agent ───────────────────────────────────────────────────────────

class RecordingAgent(agent_wrapper.VLLM_AGENT):
    """Wraps VLLM_AGENT to expose per-step raw model output and cache status."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_raw_output: str = ""
        self.last_action_tokens: Any = None
        self.last_from_cache: bool = False

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
        image  = self.processor_wrapper.create_image_input(observations[0])
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


# ── Trajectory recorder ───────────────────────────────────────────────────────

class TrajectoryRecorder:

    def __init__(self, output_dir: str, fps: int = 20, save_frames: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.save_frames = save_frames
        self.steps: List[dict] = []
        self._writer = None
        self._video_path: Optional[str] = None
        if save_frames:
            self.frames_dir = self.output_dir / "frames"
            self.frames_dir.mkdir(exist_ok=True)

    def _init_writer(self, pov: np.ndarray) -> None:
        import imageio_ffmpeg
        h, w = pov.shape[:2]
        self._video_path = str(self.output_dir / "video.mp4")
        self._writer = imageio_ffmpeg.write_frames(
            self._video_path, (w, h),
            fps=self.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=2,
            output_params=["-crf", "18"],
        )
        self._writer.send(None)

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
        if self._writer is None:
            self._init_writer(pov)
        self._writer.send(pov.astype(np.uint8).tobytes())
        if self.save_frames:
            cv2.imwrite(
                str(self.frames_dir / f"{step:05d}.png"),
                cv2.cvtColor(pov.astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
        self.steps.append({
            "step":             step,
            "instruction":      instruction,
            "model_raw_output": raw_output,
            "model_from_cache": from_cache,
            "action_tokens":    _to_python(action_tokens),
            "action":           _to_python(action),
            "reward":           float(reward),
            "env_state":        _extract_env_state(info),
        })

    def save_json(self, meta: dict) -> str:
        path = self.output_dir / "trajectory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "trajectory": self.steps}, f, indent=2, ensure_ascii=False)
        print(f"[green]trajectory.json → {path}[/green]")
        return str(path)

    def save_video(self, fps: int = 20) -> Optional[str]:
        if self._writer is None:
            return None
        self._writer.close()
        print(f"[green]video.mp4       → {self._video_path}[/green]")
        return self._video_path
