from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentkit.agent import DeployedAgent
from agentkit.config import load_config
from agentkit.llm import OpenAICompatibleModel
from agentkit.runtime import prepare
from agentkit.tracing import configure


class Ask(BaseModel):
    question: str


class Approval(BaseModel):
    thread_id: str
    approved: bool


def _response(thread_id: str, state: dict[str, Any]) -> dict[str, Any]:
    if "__interrupt__" in state:
        return {
            "status": "pending_approval",
            "thread_id": thread_id,
            "request": state["__interrupt__"][0].value,
        }
    return {
        "status": "done",
        "thread_id": thread_id,
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "tool_log": [{k: v for k, v in t.items() if k != "result"} for t in state.get("tool_log", [])],
        "usage": state.get("usage", {}),
    }


def create_server(config_path: str | None = None) -> FastAPI:
    configure(jsonl_path=os.environ.get("AGENTKIT_TRACE_FILE", "traces/spans.jsonl"))
    config = load_config(config_path or os.environ["AGENTKIT_CONFIG"])
    prepare(config)
    agent = DeployedAgent(config, OpenAICompatibleModel(config.llm))
    app = FastAPI(title=f"{config.display_name} operations agent")
    pending: set[str] = set()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "customer": config.customer_id, "model": config.llm.model}

    @app.post("/v1/ask")
    def ask(body: Ask) -> dict[str, Any]:
        thread_id, state = agent.start(body.question)
        if "__interrupt__" in state:
            pending.add(thread_id)
        return _response(thread_id, state)

    @app.post("/v1/approve")
    def approve(body: Approval) -> dict[str, Any]:
        if body.thread_id not in pending:
            raise HTTPException(404, "no pending approval for this thread")
        pending.discard(body.thread_id)
        state = agent.resume(body.thread_id, body.approved)
        if "__interrupt__" in state:
            pending.add(body.thread_id)
        return _response(body.thread_id, state)

    return app
