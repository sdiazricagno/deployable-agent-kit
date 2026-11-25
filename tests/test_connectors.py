from agentkit.connectors import ToolExecutor
from agentkit.runtime import inproc_http_factory


def test_rest_tool_projects_fields(acme):
    executor = ToolExecutor(acme, inproc_http_factory()())
    record = executor.run(acme.tool("get_client"), {"client_id": "C-1002"})
    assert record["kyb_status"] == "pending"
    assert "contact_email" not in record


def test_rest_write_and_404(acme):
    executor = ToolExecutor(acme, inproc_http_factory()())
    updated = executor.run(acme.tool("update_ticket"), {"ticket_id": "T-2042", "queue": "disputes"})
    assert updated["queue"] == "disputes"
    assert executor.run(acme.tool("get_ticket"), {"ticket_id": "T-0000"}) == {"error": "not found"}


def test_sql_tool_is_parameterised_and_read_only(contoso):
    executor = ToolExecutor(contoso, inproc_http_factory()())
    rows = executor.run(contoso.tool("recent_transfers"), {"account_id": "A-501"})
    assert rows[0]["status"] == "held_compliance_review"
    injected = executor.run(contoso.tool("get_account"), {"account_id": "A-501' OR '1'='1"})
    assert injected == []
