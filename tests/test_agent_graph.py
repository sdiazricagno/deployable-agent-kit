from agentkit.agent import DeployedAgent
from agentkit.eval.harness import run_eval
from agentkit.eval.scoring import GoldenCase, grounded_fraction
from agentkit.llm import LLMResponse, ScriptedModel, ToolCall, parse_text_tool_calls
from agentkit.runtime import inproc_http_factory


def _agent(config, responses):
    model = ScriptedModel(responses)
    return DeployedAgent(config, model, inproc_http_factory()()), model


def test_rag_answer_with_citation(acme):
    agent, model = _agent(acme, [LLMResponse("Merchants have 7 calendar days to respond [S1]. See [S9].")])
    result = agent.run("How many days does a merchant have to respond to a chargeback?")
    assert result.citations == ["disputes-runbook#chargeback-response-window"]
    assert result.invalid_citations == ["S9"]
    assert "Sources:" in model.calls[0][1]["content"]


def test_pii_never_reaches_model(acme):
    agent, model = _agent(acme, [LLMResponse("Collect the receipt [S1].")])
    agent.run("Cardholder jane.roe@example.com, card 4111 1111 1111 1111: what evidence?")
    prompt = model.calls[0][1]["content"]
    assert "jane.roe@example.com" not in prompt and "4111" not in prompt
    assert "[EMAIL]" in prompt and "[CARD]" in prompt


def test_write_requires_approval_and_rejection_is_respected(acme):
    call = ToolCall("c1", "update_ticket", {"ticket_id": "T-2041", "status": "resolved"})
    agent, _ = _agent(
        acme, [LLMResponse("", [call]), LLMResponse("The change was rejected by the reviewer.")]
    )
    seen = []
    result = agent.run("Resolve T-2041", approve=lambda req: seen.append(req) or False)
    assert seen and seen[0]["tool"] == "update_ticket"
    assert result.tool_log[0]["status"] == "rejected"
    assert "rejected by human reviewer" in result.tool_log[0]["result"]


def test_blocked_tool_is_not_executed(acme):
    call = ToolCall("c1", "update_client_limit", {"client_id": "C-1001", "monthly_limit_usd": 80000})
    agent, _ = _agent(
        acme, [LLMResponse("", [call]), LLMResponse("Limit changes go to the risk-limits queue.")]
    )
    result = agent.run("Raise the limit for C-1001")
    assert result.tool_log[0]["status"] == "blocked"
    assert result.approvals == []


def test_tool_budget_forces_final_answer(acme):
    loop = [LLMResponse("", [ToolCall(f"c{i}", "get_client", {"client_id": "C-1001"})]) for i in range(10)]
    agent, model = _agent(acme, loop)
    result = agent.run("Tell me about C-1001")
    assert len(result.tool_log) == acme.policies.max_tool_calls


def test_text_tool_call_fallback():
    content, calls = parse_text_tool_calls(
        'ok <tool_call>{"name": "get_case", "arguments": {"case_ref": "CS-7781"}}</tool_call>'
    )
    assert content == "ok"
    assert calls[0].name == "get_case" and calls[0].args == {"case_ref": "CS-7781"}


def test_eval_harness_scores_scripted_run(contoso):
    case = GoldenCase(
        id="route",
        category="action",
        question="Assign case CS-7783 to kyc-ops.",
        expected_tools=[{"name": "update_case", "args": {"case_ref": "CS-7783", "team": "kyc-ops"}}],
        approval="required",
        must_include=["kyc-ops"],
    )
    call = ToolCall("c1", "update_case", {"case_ref": "CS-7783", "team": "kyc-ops"})
    model = ScriptedModel([LLMResponse("", [call]), LLMResponse("Case CS-7783 is now with kyc-ops.")])
    report = run_eval(contoso, model, inproc_http_factory(), cases=[case], log=lambda *_: None)
    row = report["cases"][0]
    assert row["success"] and row["tool_ok"] and row["approval_ok"]
    assert report["summary"]["task_success"] == 1.0


def test_groundedness_proxy():
    evidence = "Merchants have 7 calendar days to respond with evidence"
    assert grounded_fraction("Merchants have 7 calendar days to respond [S1].", evidence) == 1.0
    assert grounded_fraction("Refunds are processed by the moon team quickly.", evidence) == 0.0


def test_reviewer_and_write_precision(acme):
    from agentkit.agent import RunResult
    from agentkit.eval.scoring import reviewer_decision, score_case

    case = GoldenCase(
        id="r",
        category="action",
        question="Route T-2042 to disputes.",
        expected_tools=[{"name": "update_ticket", "args": {"ticket_id": "T-2042", "queue": "disputes"}}],
        approval="required",
    )
    good = {"tool": "update_ticket", "args": {"ticket_id": "T-2042", "queue": "disputes"}}
    assert reviewer_decision(case, good)
    assert not reviewer_decision(
        case, {"tool": "update_ticket", "args": {"ticket_id": "T-2041", "queue": "disputes"}}
    )

    noisy = {
        "name": "update_ticket",
        "args": {**good["args"], "priority": "p1", "note": "x"},
        "status": "approved",
    }
    result = RunResult(
        "q", "Routed to disputes.", [], [], [], [{**noisy, "result": "{}"}], [good], 0, {}, 1.0
    )
    row = score_case(case, result, acme)
    assert row["write_args_exact"] is False and not row["tool_ok"]


def test_tool_hints_from_config(acme, contoso):
    from agentkit.agent import tool_hints

    hints = tool_hints(acme, "Why is C-1003 on hold? See T-2044.")
    assert "C-1003 is a client_id; get_client accepts it." in hints
    assert any(h.startswith("T-2044 is a ticket_id; get_ticket") for h in hints)
    assert any("update_ticket" in h for h in hints)
    assert tool_hints(contoso, "What is the trust policy?") == []
