"""
utils/api.py — Unified Gemini API client for all MineAgent scripts.

Supports:
  chat(system, user_text, image_b64=None, video_b64=None, ...)
      — text-only, text+image (inline PNG), or text+video (inline MP4 b64)
  chat_with_video(system, user_text, video_path, ...)
      — reads video_path from disk, base64-encodes, calls chat()

All requests go to the poloai.top relay via the Gemini generateContent endpoint.
"""

import base64
import http.client
import json
from pathlib import Path
from typing import Optional


class GeminiClient:
    def __init__(self, host: str = "poloai.top", api_key: str = "",
                 model: str = "gemini-3-flash-preview"):
        self.host    = host
        self.api_key = api_key
        self.model   = model

    def _post(self, payload: str, timeout: int = 300) -> dict:
        conn = http.client.HTTPSConnection(self.host, timeout=timeout)
        conn.request(
            "POST",
            f"/v1beta/models/{self.model}:generateContent",
            payload.encode("utf-8"),
            headers={
                "Accept":        "application/json",
                "Authorization": self.api_key,
                "Content-Type":  "application/json",
            },
        )
        data = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        return data

    def chat(
        self,
        system: str,
        user_text: str,
        image_b64: Optional[str] = None,
        video_b64: Optional[str] = None,
        video_path: Optional[Path] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a text prompt with an optional inline media attachment.
        Exactly one of image_b64 / video_b64 / video_path may be provided.
        video_path is read and base64-encoded here if supplied.
        """
        parts = [{"text": user_text}]

        if video_path is not None and Path(video_path).exists():
            video_b64 = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")

        if video_b64:
            parts.append({"inline_data": {"mime_type": "video/mp4", "data": video_b64}})
        elif image_b64:
            parts.append({"inline_data": {"mime_type": "image/png", "data": image_b64}})

        payload = json.dumps({
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature":     temperature,
                "maxOutputTokens": max_tokens,
            },
        }, ensure_ascii=False)

        data = self._post(payload, timeout=300)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected API response: {data}") from e

    def chat_with_video(
        self,
        system: str,
        user_text: str,
        video_path: Path,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> str:
        """Convenience wrapper: read video from disk and send inline."""
        return self.chat(
            system=system,
            user_text=user_text,
            video_path=video_path,
            temperature=temperature,
            max_tokens=max_tokens,
        )
