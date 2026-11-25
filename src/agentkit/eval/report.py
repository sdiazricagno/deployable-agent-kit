from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = [
    ("task_success", "Task success"),
    ("answer_accuracy", "Answer accuracy"),
    ("tool_call_correctness", "Tool-call correctness"),
    ("approval_gate_correct", "Approval gate correct"),
    ("citation_accuracy", "Citation accuracy"),
    ("groundedness", "Groundedness"),
]


def load_reports(results_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.1f}"


def markdown_table(reports: list[dict[str, Any]]) -> str:
    header = [
        "Model",
        "Customer",
        "Cases",
        *[label for _, label in METRICS],
        "p50 latency (s)",
        "p95 latency (s)",
        "Tokens/task",
        "PII leaks",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in reports:
        s = r["summary"]
        cells = [
            r["model"],
            r["customer"],
            str(s["cases"]),
            *[_fmt(s[k]) for k, _ in METRICS],
            _fmt(s["latency_p50_s"], False),
            _fmt(s["latency_p95_s"], False),
            f"{s['tokens_per_task']:.0f}",
            str(s["pii_leaks"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pooled(reports: list[dict[str, Any]], model: str) -> dict[str, float]:
    from agentkit.eval.scoring import summarize

    rows = [row for r in reports if r["model"] == model for row in r["cases"]]
    return summarize(rows)


def plot_metrics(reports: list[dict[str, Any]], path: Path) -> Path:
    models = list(dict.fromkeys(r["model"] for r in reports))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        summary = _pooled(reports, model)
        values = [summary[k] or 0 for k, _ in METRICS]
        xs = [j + i * width for j in range(len(METRICS))]
        bars = ax.bar(xs, values, width, label=model)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in values], fontsize=8, padding=2)
    ax.set_xticks([j + width * (len(models) - 1) / 2 for j in range(len(METRICS))])
    ax.set_xticklabels([label for _, label in METRICS], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (golden set, both customers)")
    ax.set_title("Agent eval on CPU: golden-set metrics by model")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_categories(reports: list[dict[str, Any]], path: Path) -> Path:
    models = list(dict.fromkeys(r["model"] for r in reports))
    cats = ["policy", "lookup", "action", "guardrail"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / max(len(models), 1)
    all_lat = [row["latency_s"] for r in reports for row in r["cases"]] or [0.0]
    step = max(all_lat) / 12 or 1.0
    bins = [step * k for k in range(14)]
    for i, model in enumerate(models):
        summary = _pooled(reports, model)
        values = [summary["by_category"][c] or 0 for c in cats]
        bars = ax1.bar([j + i * width for j in range(len(cats))], values, width, label=model)
        ax1.bar_label(bars, labels=[f"{v:.0%}" for v in values], fontsize=8, padding=2)
        lat = [row["latency_s"] for r in reports if r["model"] == model for row in r["cases"]]
        ax2.hist(lat, bins=bins, alpha=0.6, label=model)
    ax1.set_xticks([j + width * (len(models) - 1) / 2 for j in range(len(cats))])
    ax1.set_xticklabels(cats)
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Task success by case category")
    ax1.legend(fontsize=8)
    ax2.set_title("End-to-end latency per task (CPU)")
    ax2.set_xlabel("seconds")
    ax2.set_ylabel("tasks")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
