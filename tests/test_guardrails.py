from agentkit.guardrails import check_tool_call, luhn_ok, redact


def test_redacts_email_card_phone_iban_ssn():
    text = (
        "Mail jane.roe@example.com, card 4111 1111 1111 1111, call +1 415 555 0142, "
        "IBAN GB82 WEST 1234 5698 7654 32, SSN 123-45-6789"
    )
    result = redact(text)
    for raw in ("jane.roe@example.com", "4111 1111 1111 1111", "555 0142", "GB82", "123-45-6789"):
        assert raw not in result.text
    assert result.counts == {"email": 1, "card": 1, "phone": 1, "iban": 1, "ssn": 1}


def test_keeps_business_identifiers():
    text = "Ticket T-2042 for C-1001 on 2026-11-24, amount 75,000 USD, order 88123"
    assert redact(text).text == text


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")


def test_tool_gate(acme):
    assert check_tool_call(acme, "get_client", {"client_id": "C-1001"}).status == "allowed"
    assert check_tool_call(acme, "update_client_limit", {"client_id": "C-1001"}).status == "blocked"
    assert check_tool_call(acme, "drop_tables", {}).status == "blocked"
    assert check_tool_call(acme, "get_client", {}).status == "invalid"
    assert (
        check_tool_call(acme, "update_ticket", {"ticket_id": "T-1", "queue": "nowhere"}).status == "invalid"
    )
    decision = check_tool_call(acme, "update_ticket", {"ticket_id": "T-2042", "queue": "disputes"})
    assert decision.status == "needs_approval"
