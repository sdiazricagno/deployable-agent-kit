from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentkit.config import CustomerConfig, ToolSpec

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"),
    "card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<![\w-])\+?\d{1,3}[ .-]?\(?\d{2,4}\)?[ .-]\d{3,4}[ .-]\d{3,4}(?![\w-])"),
}


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return len(digits) >= 13 and total % 10 == 0


@dataclass
class Redaction:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str, types: list[str] | None = None) -> Redaction:
    counts: dict[str, int] = {}
    for kind in types or list(PII_PATTERNS):
        pattern = PII_PATTERNS[kind]

        def _sub(match: re.Match[str], kind: str = kind) -> str:
            if kind == "card" and not luhn_ok(match.group(0)):
                return match.group(0)
            counts[kind] = counts.get(kind, 0) + 1
            return f"[{kind.upper()}]"

        text = pattern.sub(_sub, text)
    return Redaction(text=text, counts=counts)


@dataclass
class ToolDecision:
    status: str
    reason: str = ""
    tool: ToolSpec | None = None
    args: dict[str, object] = field(default_factory=dict)


_CASTS = {"string": str, "integer": int, "number": float, "boolean": bool}


def validate_args(tool: ToolSpec, args: dict[str, object]) -> tuple[dict[str, object], str | None]:
    unknown = set(args) - set(tool.params)
    if unknown:
        return {}, f"unknown arguments: {sorted(unknown)}"
    clean: dict[str, object] = {}
    for name, spec in tool.params.items():
        if name not in args or args[name] in (None, ""):
            if spec.required:
                return {}, f"missing required argument {name!r}"
            continue
        try:
            value = _CASTS[spec.type](args[name])
        except (TypeError, ValueError):
            return {}, f"argument {name!r} must be {spec.type}"
        if spec.enum and value not in spec.enum:
            return {}, f"argument {name!r} must be one of {spec.enum}"
        clean[name] = value
    return clean, None


def check_tool_call(config: CustomerConfig, name: str, args: dict[str, object]) -> ToolDecision:
    """Allow-list, schema and approval gate applied to every tool call the model proposes."""
    tool = config.tool(name)
    if tool is None or name not in config.policies.allowed_tools:
        return ToolDecision("blocked", f"tool {name!r} is not allowed for {config.customer_id}")
    clean, error = validate_args(tool, args)
    if error:
        return ToolDecision("invalid", error, tool)
    if config.needs_approval(tool):
        return ToolDecision("needs_approval", "write action requires human approval", tool, clean)
    return ToolDecision("allowed", "", tool, clean)
