from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agentkit.config import CustomerConfig, ToolSpec
from agentkit.connectors import ToolExecutor
from agentkit.guardrails import check_tool_call, redact
from agentkit.llm import ChatModel, LLMResponse
from agentkit.retrieval import load_index
from agentkit.tracing import span

_CITATION = re.compile(r"\[S(\d+)\]")

RULES = """Rules:
- Answer only from the numbered sources and from tool results. If they do not contain the answer, say you don't know.
- When the request names a record id (client, account, ticket, case), look it up with a tool before answering.
- For changes (updating a ticket or case, adding a note) call the write tool; a human reviewer approves it before it runs.
- In a write call, set only the fields the request asks to change.
- If a tool result says the reviewer rejected the change, tell the user the change was not made.
- If the request needs an action you have no tool for, say you cannot do it and point to the right team per policy.
- Personal data in the request is masked; never try to recover it.
- Be concise: at most four sentences."""

ANSWER_INSTRUCTION = "Cite every source you use inline like [S1]. In write calls, set only the fields the request asks to change."


def tool_hints(config: CustomerConfig, question: str) -> list[str]:
    """Point the model at the tools whose id parameters match identifiers in the request."""
    hints = []
    for tool in config.allowed():
        for name, param in tool.params.items():
            if param.id_pattern:
                for value in dict.fromkeys(re.findall(param.id_pattern, question)):
                    hints.append(f"{value} is a {name}; {tool.name} accepts it.")
    return hints


class AgentState(TypedDict, total=False):
    question: str
    redacted_question: str
    pii_redactions: int
    sources: list[dict[str, str]]
    messages: list[dict[str, Any]]
    pending: list[dict[str, Any]]
    tool_log: list[dict[str, Any]]
    steps: int
    answer: str
    citations: list[str]
    invalid_citations: list[str]
    usage: dict[str, float]


def tool_schema(tool: ToolSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, p in tool.params.items():
        prop: dict[str, Any] = {"type": p.type, "description": p.description}
        if p.enum:
            prop["enum"] = p.enum
        properties[name] = prop
    required = [n for n, p in tool.params.items() if p.required]
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _add_usage(state: AgentState, response: LLMResponse, config: CustomerConfig) -> dict[str, float]:
    usage = dict(
        state.get("usage") or {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0, "cost_usd": 0.0}
    )
    usage["input_tokens"] += response.input_tokens
    usage["output_tokens"] += response.output_tokens
    usage["llm_calls"] += 1
    usage["cost_usd"] += (
        response.input_tokens * config.llm.input_price_per_mtok
        + response.output_tokens * config.llm.output_price_per_mtok
    ) / 1e6
    return usage


class AgentGraph:
    def __init__(self, config: CustomerConfig, model: ChatModel, http: httpx.Client | None = None) -> None:
        self.config = config
        self.model = model
        self.executor = ToolExecutor(config, http)
        self.index = load_index(config.resolve(config.retrieval.docs_dir))
        self.tools = [tool_schema(t) for t in config.allowed()]
        self.graph = self._build()

    def _build(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("ingress", self.ingress)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("agent", self.agent)
        builder.add_node("tools", self.run_tool)
        builder.add_node("egress", self.egress)
        builder.add_edge(START, "ingress")
        builder.add_edge("ingress", "retrieve")
        builder.add_edge("retrieve", "agent")
        builder.add_conditional_edges("agent", lambda s: "tools" if s.get("pending") else "egress")
        builder.add_conditional_edges("tools", lambda s: "tools" if s.get("pending") else "agent")
        builder.add_edge("egress", END)
        return builder.compile(checkpointer=InMemorySaver())

    def ingress(self, state: AgentState) -> AgentState:
        policies = self.config.policies
        with span("agent.ingress") as s:
            if policies.redact_pii:
                red = redact(state["question"], policies.pii_types)
                s.set_attribute("guardrail.pii_redactions", red.total)
                return {
                    "redacted_question": red.text,
                    "pii_redactions": red.total,
                    "steps": 0,
                    "tool_log": [],
                }
            return {"redacted_question": state["question"], "pii_redactions": 0, "steps": 0, "tool_log": []}

    def retrieve(self, state: AgentState) -> AgentState:
        with span("agent.retrieve", **{"retrieval.top_k": self.config.retrieval.top_k}) as s:
            chunks = self.index.search(state["redacted_question"], self.config.retrieval.top_k)
            s.set_attribute("retrieval.chunk_ids", ",".join(c.chunk_id for c in chunks))
        sources = [
            {"ref": f"S{i}", "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "text": c.text}
            for i, c in enumerate(chunks, 1)
        ]
        hints = tool_hints(self.config, state["redacted_question"])
        hint_text = ("\n\nRecord ids in the request:\n" + "\n".join(hints)) if hints else ""
        context = "\n\n".join(f"[{s['ref']}] {s['title']}\n{s['text']}" for s in sources) or "(no sources)"
        messages = [
            {"role": "system", "content": f"{self.config.persona.strip()}\n\n{RULES}"},
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nRequest: {state['redacted_question']}{hint_text}\n\n{ANSWER_INSTRUCTION}",
            },
        ]
        return {"sources": sources, "messages": messages}

    def agent(self, state: AgentState) -> AgentState:
        steps = state.get("steps", 0)
        budget_left = steps < self.config.policies.max_tool_calls
        with span(
            "agent.llm",
            **{
                "gen_ai.system": "openai_compatible",
                "gen_ai.request.model": self.model.name,
                "agent.step": steps,
            },
        ) as s:
            response = self.model.chat(state["messages"], self.tools if budget_left else None)
            s.set_attribute("gen_ai.usage.input_tokens", response.input_tokens)
            s.set_attribute("gen_ai.usage.output_tokens", response.output_tokens)
            s.set_attribute("agent.tool_calls", ",".join(c.name for c in response.tool_calls))
        usage = _add_usage(state, response, self.config)
        if response.tool_calls and budget_left:
            message = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.args)},
                    }
                    for c in response.tool_calls
                ],
            }
            pending = [{"id": c.id, "name": c.name, "args": c.args} for c in response.tool_calls]
            return {"messages": [*state["messages"], message], "pending": pending, "usage": usage}
        return {
            "messages": [*state["messages"], {"role": "assistant", "content": response.content}],
            "answer": response.content,
            "pending": [],
            "usage": usage,
        }

    def run_tool(self, state: AgentState) -> AgentState:
        call, rest = state["pending"][0], state["pending"][1:]
        decision = check_tool_call(self.config, call["name"], call["args"])
        entry: dict[str, Any] = {"name": call["name"], "args": call["args"], "status": decision.status}
        if decision.status == "needs_approval":
            verdict = interrupt({"tool": call["name"], "args": decision.args, "reason": decision.reason})
            approved = bool(verdict.get("approved")) if isinstance(verdict, dict) else bool(verdict)
            entry["approval_requested"] = True
            entry["status"] = "approved" if approved else "rejected"
        with span("agent.tool", **{"tool.name": call["name"], "guardrail.decision": entry["status"]}):
            if entry["status"] in ("allowed", "approved"):
                output: Any = self.executor.run(decision.tool, decision.args)  # type: ignore[arg-type]
            elif entry["status"] == "rejected":
                output = {"error": "rejected by human reviewer; do not retry, tell the user"}
            else:
                output = {"error": decision.reason}
        text = json.dumps(output, default=str)
        if self.config.policies.redact_pii:
            text = redact(text, self.config.policies.pii_types).text
        entry["result"] = text
        tool_message = {"role": "tool", "tool_call_id": call["id"], "content": text}
        return {
            "messages": [*state["messages"], tool_message],
            "pending": rest,
            "tool_log": [*state.get("tool_log", []), entry],
            "steps": state.get("steps", 0) + 1,
        }

    def egress(self, state: AgentState) -> AgentState:
        answer = state.get("answer") or ""
        refs = {s["ref"]: s["chunk_id"] for s in state.get("sources", [])}
        cited = [f"S{n}" for n in dict.fromkeys(_CITATION.findall(answer))]
        valid = [refs[c] for c in cited if c in refs]
        invalid = [c for c in cited if c not in refs]
        with span("agent.egress", **{"citations.valid": len(valid), "citations.invalid": len(invalid)}):
            if self.config.policies.redact_pii:
                answer = redact(answer, self.config.policies.pii_types).text
        return {"answer": answer, "citations": valid, "invalid_citations": invalid}


@dataclass
class RunResult:
    question: str
    answer: str
    citations: list[str]
    invalid_citations: list[str]
    sources: list[dict[str, str]]
    tool_log: list[dict[str, Any]]
    approvals: list[dict[str, Any]]
    pii_redactions: int
    usage: dict[str, float]
    latency_s: float
    error: str | None = None
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class DeployedAgent:
    def __init__(self, config: CustomerConfig, model: ChatModel, http: httpx.Client | None = None) -> None:
        self.config = config
        self.app = AgentGraph(config, model, http)

    def start(self, question: str, thread_id: str | None = None) -> tuple[str, dict[str, Any]]:
        thread_id = thread_id or uuid.uuid4().hex
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
        return thread_id, self.app.graph.invoke({"question": question}, cfg)

    def resume(self, thread_id: str, approved: bool) -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
        return self.app.graph.invoke(Command(resume={"approved": approved}), cfg)

    def run(self, question: str, approve: Any = None) -> RunResult:
        """Run to completion; `approve(request) -> bool` plays the human reviewer for write actions."""
        start = time.perf_counter()
        approvals: list[dict[str, Any]] = []
        with span(
            "agent.run",
            **{"customer.id": self.config.customer_id, "gen_ai.request.model": self.app.model.name},
        ):
            thread_id, state = self.start(question)
            while "__interrupt__" in state:
                request = state["__interrupt__"][0].value
                verdict = bool(approve(request)) if approve else False
                approvals.append({**request, "approved": verdict})
                state = self.resume(thread_id, verdict)
        return RunResult(
            question=question,
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            invalid_citations=state.get("invalid_citations", []),
            sources=state.get("sources", []),
            tool_log=state.get("tool_log", []),
            approvals=approvals,
            pii_redactions=state.get("pii_redactions", 0),
            usage=state.get("usage", {}),
            latency_s=time.perf_counter() - start,
            thread_id=thread_id,
        )
