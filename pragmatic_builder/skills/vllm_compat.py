"""Compatibility helpers for running the BWIM pipeline on vLLM/Nemotron."""

from __future__ import annotations

import os
import re


def is_vllm_mode() -> bool:
    """Return True if the agent is configured for vLLM (local inference).

    Detection: model name starts with /models/ (vLLM convention) or
    the VLLM_MODE env var is set.
    """
    if os.getenv("VLLM_MODE", "").strip().lower() in {"1", "true", "yes"}:
        return True
    model = (os.getenv("OPENAI_MODEL_PURPLE", "").strip()
             or os.getenv("OPENAI_MODEL", ""))
    return model.startswith("/models/")


def strip_think_tags(content: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    Nemotron 3 Super emits <think> blocks before the actual response.
    These must be stripped before JSON parsing or command extraction.
    """
    if not content:
        return ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def vllm_extra_body() -> dict | None:
    """Return extra_body kwargs for vLLM requests, or None for OpenAI."""
    if not is_vllm_mode():
        return None
    return {"chat_template_kwargs": {"force_nonempty_content": True}}
