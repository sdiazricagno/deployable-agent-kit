from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

from agentkit.config import LLMConfig

_TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse: ...


def _parse_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_text_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    """Recover tool calls that small models emit as text instead of structured calls."""
    calls = []
    for match in _TEXT_TOOL_CALL.finditer(content):
        payload = _parse_args(match.group(1))
        if "name" in payload:
            args = _parse_args(payload.get("arguments") or payload.get("parameters"))
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", str(payload["name"]), args))
    return _TEXT_TOOL_CALL.sub("", content).strip(), calls


class OpenAICompatibleModel:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.name = config.model
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=os.environ.get(config.api_key_env) or "not-needed",
            timeout=config.timeout_s,
        )

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        start = time.perf_counter()
        kwargs: dict[str, Any] = {"tools": tools} if tools else {}
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )
        message = completion.choices[0].message
        calls = [
            ToolCall(
                tc.id or f"call_{uuid.uuid4().hex[:8]}", tc.function.name, _parse_args(tc.function.arguments)
            )
            for tc in message.tool_calls or []
            if tc.type == "function"
        ]
        content = message.content or ""
        if not calls and "<tool_call>" in content:
            content, calls = parse_text_tool_calls(content)
        usage = completion.usage
        return LLMResponse(
            content=content,
            tool_calls=calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - start,
        )


class ScriptedModel:
    """Deterministic model for tests: replays canned responses in order."""

    def __init__(self, responses: list[LLMResponse], name: str = "scripted") -> None:
        self.responses = list(responses)
        self.name = name
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0) if self.responses else LLMResponse(content="I don't know.")
