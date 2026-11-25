from pathlib import Path

import pytest

from agentkit.config import CustomerConfig, load_config
from agentkit.runtime import prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def acme() -> CustomerConfig:
    return load_config(ROOT / "configs/customers/acme-payments.yaml")


@pytest.fixture
def contoso() -> CustomerConfig:
    config = load_config(ROOT / "configs/customers/contoso-wealth.yaml")
    prepare(config)
    return config
