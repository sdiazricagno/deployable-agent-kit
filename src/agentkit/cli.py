from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentkit.config import load_config
from agentkit.llm import OpenAICompatibleModel
from agentkit.runtime import inproc_http_factory, prepare
from agentkit.tracing import configure


def _ask(args: argparse.Namespace) -> None:
    from agentkit.agent import DeployedAgent

    config = load_config(args.config)
    prepare(config)
    http = inproc_http_factory()() if args.inproc else None
    agent = DeployedAgent(config, OpenAICompatibleModel(config.llm), http)

    def approve(request: dict) -> bool:
        if args.auto_approve:
            return True
        reply = input(f"\nApprove {request['tool']}({json.dumps(request['args'])})? [y/N] ")
        return reply.strip().lower() in ("y", "yes")

    result = agent.run(args.question, approve)
    print(result.answer)
    print(f"\ncitations: {result.citations}")
    for call in result.tool_log:
        print(f"tool: {call['name']} {call['args']} -> {call['status']}")
    print(f"latency: {result.latency_s:.1f}s  usage: {result.usage}")


def _eval(args: argparse.Namespace) -> None:
    from agentkit.eval.harness import run_eval, save

    for path in args.configs:
        config = load_config(path)
        prepare(config)
        model = OpenAICompatibleModel(config.llm)
        print(f"{config.customer_id} with {model.name} at {config.llm.base_url}")
        report = run_eval(config, model, inproc_http_factory() if args.inproc else None)
        out = save(report, Path(args.out))
        print(json.dumps(report["summary"], indent=2))
        print(f"saved {out}")


def _report(args: argparse.Namespace) -> None:
    from agentkit.eval.report import load_reports, markdown_table, plot_categories, plot_metrics

    reports = load_reports(Path(args.results))
    docs = Path(args.docs)
    plot_metrics(reports, docs / "eval_metrics.png")
    plot_categories(reports, docs / "eval_categories_latency.png")
    table = markdown_table(reports)
    (docs / "eval_results.md").write_text(table + "\n")
    print(table)


def _mocks(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("agentkit.mock_apis.app:app", host=args.host, port=args.port)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from agentkit.server import create_server

    uvicorn.run(create_server(args.config), host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentkit")
    parser.add_argument("--trace-file", default=None, help="write OpenTelemetry spans as JSONL")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask the deployed agent one question")
    ask.add_argument("config")
    ask.add_argument("question")
    ask.add_argument("--inproc", action="store_true", help="use in-process mock systems")
    ask.add_argument("--auto-approve", action="store_true")
    ask.set_defaults(func=_ask)

    ev = sub.add_parser("eval", help="run the golden set for one or more customer configs")
    ev.add_argument("configs", nargs="+")
    ev.add_argument("--inproc", action="store_true")
    ev.add_argument("--out", default="docs/results")
    ev.set_defaults(func=_eval)

    rep = sub.add_parser("report", help="build the eval table and charts from saved results")
    rep.add_argument("--results", default="docs/results")
    rep.add_argument("--docs", default="docs")
    rep.set_defaults(func=_report)

    mocks = sub.add_parser("mocks", help="run the mock customer systems")
    mocks.add_argument("--host", default="127.0.0.1")
    mocks.add_argument("--port", type=int, default=8080)
    mocks.set_defaults(func=_mocks)

    serve = sub.add_parser("serve", help="serve the agent over HTTP with approval endpoints")
    serve.add_argument("config")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    configure(jsonl_path=args.trace_file)
    args.func(args)
