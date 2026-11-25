from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:8081/v1"
    model: str = "qwen2.5-1.5b-instruct"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 400
    timeout_s: float = 180.0
    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0


class RestConnector(BaseModel):
    type: Literal["rest"]
    base_url: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 10.0


class SqlConnector(BaseModel):
    type: Literal["sql"]
    path: str


Connector = Annotated[RestConnector | SqlConnector, Field(discriminator="type")]


class ToolParam(BaseModel):
    type: Literal["string", "integer", "number", "boolean"] = "string"
    description: str = ""
    enum: list[str] | None = None
    required: bool = True
    id_pattern: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    kind: Literal["read", "write"] = "read"
    connector: str
    method: Literal["GET", "POST", "PATCH"] = "GET"
    path: str | None = None
    sql: str | None = None
    params: dict[str, ToolParam] = Field(default_factory=dict)
    response_fields: list[str] | None = None

    @model_validator(mode="after")
    def _has_target(self) -> ToolSpec:
        if (self.path is None) == (self.sql is None):
            raise ValueError(f"tool {self.name!r} needs exactly one of 'path' or 'sql'")
        if self.sql is not None and self.kind == "write":
            raise ValueError(f"tool {self.name!r}: SQL tools are read-only")
        return self


class Policies(BaseModel):
    allowed_tools: list[str]
    require_approval: list[str] | None = None
    redact_pii: bool = True
    pii_types: list[str] = Field(default_factory=lambda: ["email", "iban", "card", "ssn", "phone"])
    max_tool_calls: int = 4
    require_citations: bool = True


class RetrievalConfig(BaseModel):
    docs_dir: str
    top_k: int = 4


class CustomerConfig(BaseModel):
    customer_id: str
    display_name: str
    persona: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig
    connectors: dict[str, Connector]
    tools: list[ToolSpec]
    policies: Policies
    golden_set: str | None = None
    base_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _check_refs(self) -> CustomerConfig:
        names = {t.name for t in self.tools}
        unknown = set(self.policies.allowed_tools) - names
        if unknown:
            raise ValueError(f"allowed_tools references unknown tools: {sorted(unknown)}")
        for tool in self.tools:
            if tool.connector not in self.connectors:
                raise ValueError(f"tool {tool.name!r} uses unknown connector {tool.connector!r}")
            kind = self.connectors[tool.connector].type
            if (kind == "sql") != (tool.sql is not None):
                raise ValueError(f"tool {tool.name!r} does not match connector type {kind!r}")
        return self

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    def allowed(self) -> list[ToolSpec]:
        return [t for t in self.tools if t.name in self.policies.allowed_tools]

    def needs_approval(self, tool: ToolSpec) -> bool:
        if self.policies.require_approval is None:
            return tool.kind == "write"
        return tool.name in self.policies.require_approval or tool.kind == "write"

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (self.base_dir / path).resolve()


def expand_env(value: object) -> object:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> CustomerConfig:
    path = Path(path).resolve()
    raw = expand_env(yaml.safe_load(path.read_text()))
    config = CustomerConfig.model_validate(raw)
    config.base_dir = path.parent
    if base_url := os.environ.get("LLM_BASE_URL"):
        config.llm.base_url = base_url
    if model := os.environ.get("LLM_MODEL"):
        config.llm.model = model
    return config
