from __future__ import annotations

import sqlite3
from typing import Any
from urllib.parse import quote

import httpx

from agentkit.config import CustomerConfig, RestConnector, SqlConnector, ToolSpec


def _project(data: Any, fields: list[str] | None) -> Any:
    if fields is None:
        return data
    if isinstance(data, list):
        return [_project(d, fields) for d in data]
    if isinstance(data, dict):
        return {k: data[k] for k in fields if k in data}
    return data


class ToolExecutor:
    """Runs allowed tool calls against the customer's systems as described in config."""

    def __init__(self, config: CustomerConfig, http: httpx.Client | None = None) -> None:
        self.config = config
        self.http = http or httpx.Client()

    def run(self, tool: ToolSpec, args: dict[str, object]) -> Any:
        connector = self.config.connectors[tool.connector]
        try:
            if isinstance(connector, SqlConnector):
                result = self._sql(connector, tool, args)
            else:
                result = self._rest(connector, tool, args)
        except (httpx.HTTPError, sqlite3.Error) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return _project(result, tool.response_fields)

    def _rest(self, connector: RestConnector, tool: ToolSpec, args: dict[str, object]) -> Any:
        assert tool.path is not None
        path_keys = {k for k in args if "{" + k + "}" in tool.path}
        path = tool.path.format(**{k: quote(str(args[k]), safe="") for k in path_keys})
        rest = {k: v for k, v in args.items() if k not in path_keys}
        url = connector.base_url.rstrip("/") + path
        if tool.method == "GET":
            response = self.http.get(url, params=rest, headers=connector.headers, timeout=connector.timeout_s)
        else:
            response = self.http.request(
                tool.method, url, json=rest, headers=connector.headers, timeout=connector.timeout_s
            )
        if response.status_code == 404:
            return {"error": "not found"}
        response.raise_for_status()
        return response.json()

    def _sql(self, connector: SqlConnector, tool: ToolSpec, args: dict[str, object]) -> list[dict[str, Any]]:
        assert tool.sql is not None
        uri = f"file:{self.config.resolve(connector.path)}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(tool.sql, args).fetchmany(25)
        return [dict(r) for r in rows]
