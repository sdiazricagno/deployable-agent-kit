import pytest

from agentkit.config import CustomerConfig, load_config
from agentkit.retrieval import chunk_markdown, load_index
from tests.conftest import ROOT


def test_configs_load(acme, contoso):
    assert {t.name for t in acme.allowed()} == {"get_client", "get_ticket", "update_ticket"}
    assert "close_account" not in {t.name for t in contoso.allowed()}
    assert contoso.connectors["warehouse"].type == "sql"


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-hosted-model")
    monkeypatch.setenv("MOCK_API_URL", "http://crm.internal:9000")
    config = load_config(ROOT / "configs/customers/acme-payments.yaml")
    assert config.llm.model == "some-hosted-model"
    assert config.connectors["crm"].base_url == "http://crm.internal:9000/acme/crm"


def test_rejects_unknown_allowed_tool(acme):
    raw = acme.model_dump()
    raw["policies"]["allowed_tools"].append("wire_money")
    with pytest.raises(ValueError, match="unknown tools"):
        CustomerConfig.model_validate(raw)


def test_chunking_and_search(acme):
    chunks = chunk_markdown("doc", "# Title\n\n## First part\nalpha beta\n\n## Second\ngamma")
    assert [c.chunk_id for c in chunks] == ["doc#first-part", "doc#second"]
    index = load_index(acme.resolve(acme.retrieval.docs_dir))
    top = index.search("how many days to respond to a chargeback", k=1)[0]
    assert top.chunk_id == "disputes-runbook#chargeback-response-window"
