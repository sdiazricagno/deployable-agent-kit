from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from agentkit.agent import DeployedAgent, RunResult
from agentkit.config import CustomerConfig
from agentkit.eval.scoring import GoldenCase, load_golden, reviewer_decision, score_case, summarize
from agentkit.llm import ChatModel


def run_eval(
    config: CustomerConfig,
    model: ChatModel,
    http_factory: Any = None,
    cases: list[GoldenCase] | None = None,
    log: Any = None,
) -> dict[str, Any]:
    """Run the golden set; each case gets fresh mock systems so write actions don't leak between cases."""
    if cases is None:
        assert config.golden_set, "config has no golden_set"
        cases = load_golden(str(config.resolve(config.golden_set)))
    log = log or (lambda message: print(message, flush=True))
    rows = []
    for case in cases:
        http: httpx.Client | None = http_factory() if http_factory else None
        agent = DeployedAgent(config, model, http)
        start = time.perf_counter()
        try:
            result = agent.run(case.question, approve=lambda req, c=case: reviewer_decision(c, req))
        except Exception as exc:  # noqa: BLE001
            result = RunResult(
                case.question, "", [], [], [], [], [], 0, {}, time.perf_counter() - start, repr(exc)
            )
        row = score_case(case, result, config)
        rows.append(row)
        log(f"  {case.id:<32} success={row['success']!s:<5} {row['latency_s']:6.1f}s  tools={row['tools']}")
    return {"customer": config.customer_id, "model": model.name, "summary": summarize(rows), "cases": rows}


def save(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['customer']}__{report['model'].replace('/', '_')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    return path
