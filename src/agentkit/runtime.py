from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import httpx

from agentkit.config import CustomerConfig, SqlConnector


def prepare(config: CustomerConfig) -> None:
    """Build any synthetic SQLite warehouse a config points at, from the seed.json next to it."""
    from agentkit.mock_apis.app import build_warehouse, load_seed

    for connector in config.connectors.values():
        if isinstance(connector, SqlConnector):
            path = config.resolve(connector.path)
            seed = path.parent / "seed.json"
            if not path.exists() and seed.exists():
                build_warehouse(load_seed(path.parent.name), path)


def inproc_http_factory() -> Callable[[], httpx.Client]:
    """Fresh in-process mock systems per call, so evals and tests need no running server."""
    from fastapi.testclient import TestClient

    from agentkit.mock_apis.app import create_app

    warnings.filterwarnings("ignore", message="You should not use the 'timeout' argument")
    return lambda: TestClient(create_app())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
