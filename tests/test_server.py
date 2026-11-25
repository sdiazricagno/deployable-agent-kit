from fastapi.testclient import TestClient

import agentkit.server as server
from agentkit.llm import LLMResponse, ScriptedModel, ToolCall
from agentkit.runtime import inproc_http_factory
from tests.conftest import ROOT


def test_http_approval_flow(monkeypatch):
    call = ToolCall("c1", "update_case", {"case_ref": "CS-7781", "team": "payments-ops"})
    model = ScriptedModel([LLMResponse("", [call]), LLMResponse("Assigned to payments-ops.")])
    monkeypatch.setattr(server, "OpenAICompatibleModel", lambda _cfg: model)
    original = server.DeployedAgent
    monkeypatch.setattr(server, "DeployedAgent", lambda cfg, m: original(cfg, m, inproc_http_factory()()))
    client = TestClient(server.create_server(str(ROOT / "configs/customers/contoso-wealth.yaml")))

    first = client.post("/v1/ask", json={"question": "Route CS-7781 to payments-ops"}).json()
    assert first["status"] == "pending_approval"
    assert first["request"]["tool"] == "update_case"

    done = client.post("/v1/approve", json={"thread_id": first["thread_id"], "approved": True}).json()
    assert done["status"] == "done"
    assert done["tool_log"][0]["status"] == "approved"
    assert (
        client.post("/v1/approve", json={"thread_id": first["thread_id"], "approved": True}).status_code
        == 404
    )
