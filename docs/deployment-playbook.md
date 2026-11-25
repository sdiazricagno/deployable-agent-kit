# Deployment playbook

How a new customer deployment of the operations agent runs, from first call to steady state. Each phase has an exit criterion so the rollout can be paused or rolled back at any point.

## 1. Discovery (week 1)

- **Map the workflow, not the org chart.** Sit with two or three analysts and list the ten requests they handle most (policy questions, record lookups, ticket or case routing). These become the first golden-set cases.
- **Inventory the systems of record.** For each request, note where the data lives (REST CRM, ticketing tool, warehouse table), how it is authenticated, and which fields hold personal data.
- **Agree what the agent may do.** Split actions into read and write. Writes start behind human approval; some actions (closing accounts, changing limits) stay off the allow-list entirely.
- **Collect the policy corpus.** Runbooks, SOPs and policy pages in markdown. Remove anything that is not meant to be quoted to staff.
- **Define success with the customer.** Pick 3 to 5 measurable targets, for example: task success on the golden set, median handling time, share of tickets routed correctly on the first try, zero PII leaks.

Exit: a signed-off list of use cases, systems, allowed actions and success metrics.

## 2. Integration (weeks 2 to 3)

- **Write the customer config**, not code: `configs/customers/<customer>.yaml` declares connectors, tools (endpoint or parameterised SQL, parameters, field projection), persona and policies.
- **Project fields at the connector.** `response_fields` and narrow SQL `SELECT`s keep names, emails and phone numbers out of the prompt in the first place; redaction is the second layer.
- **Stand up the mocks first.** Mirror each system's real response shape in `mock_apis/` with synthetic records, so the agent and evals can be built before production credentials arrive.
- **Build the golden set** in `evals/<customer>.yaml` from the discovery list: policy answers with expected source documents, lookups with expected tool calls, write actions with expected arguments and approval behaviour, and guardrail cases (blocked actions, PII in the request, out-of-scope questions).
- **Choose the model endpoint.** Anything OpenAI-compatible works: a small open model on the customer's own hardware through llama.cpp or Ollama when data cannot leave their network, or a hosted API set through `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`.

Exit: the eval runs end to end against the mocks, and the security review has approved the allow-list and data flows.

## 3. Rollout (weeks 3 to 6)

1. **Shadow mode.** The agent answers alongside analysts; nothing it proposes is executed. Compare its routing with what the analyst did.
2. **Assisted mode.** Write actions run only after an analyst approves them through `/v1/approve`. Track the approval rate per tool; a low rate points at a prompt, config or policy gap.
3. **Scoped autonomy.** Only for read-only or low-risk writes with a sustained high approval rate, and only after the customer's risk owner signs off. High-impact writes stay behind approval permanently.

Every phase keeps the traces (`traces/spans.jsonl`, or any OTLP backend such as Langfuse through `OTEL_EXPORTER_OTLP_ENDPOINT`) so each answer can be audited back to its sources and tool calls.

Exit criteria per phase: golden-set task success at or above the agreed target, no PII leaks, no blocked-tool executions, and latency inside the agreed budget.

## 4. Success metrics and operations

| Metric | Source | Typical target to agree |
|---|---|---|
| Task success on golden set | `agentkit eval` | agreed per customer, re-run on every config or model change |
| Tool-call correctness | `agentkit eval` | near 100% for write actions |
| Citation accuracy and groundedness | `agentkit eval` | answers cite the right policy |
| Approval rate per write tool | traces | rising over the rollout |
| PII leaks | `agentkit eval`, trace audits | zero |
| p50 / p95 latency, tokens and cost per task | traces | inside the agreed budget |

Changes to prompts, tools, policies or the model ship like code: open a pull request, CI runs the unit tests, and the golden-set eval is re-run against the live model before the config is promoted.
