# TokenCircuit

<p align="center">
  <img src="docs/assets/logo.png" alt="TokenCircuit Logo" width="200"/>
</p>

<p align="center">
  <b>The Pre-Model Intervention Engine for Autonomous Agents.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/tokencircuit/">
    <img src="https://img.shields.io/pypi/v/tokencircuit" alt="PyPI"/>
  </a>
  <a href="https://pypi.org/project/tokencircuit/">
    <img src="https://img.shields.io/pypi/pyversions/tokencircuit" alt="Python Versions"/>
  </a>
  <a href="https://github.com/Devaretanmay/TokenCircut/actions">
    <img src="https://github.com/Devaretanmay/TokenCircut/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/Devaretanmay/TokenCircut" alt="License"/>
  </a>
  <a href="tests/performance/">
    <img src="https://img.shields.io/badge/overhead-%3C20%C2%B5s%2Fturn-green" alt="<20µs/turn overhead"/>
  </a>
</p>

---

Most guardrails are blunt instruments. They wait for your agent to burn $50 in API
credits, hit a hard `recursion_limit`, and crash—wiping the state and returning an
error to your user.

**TokenCircuit** is a surgical, pre-model intervention layer. Using zero-dependency
semantic shingling, it detects paraphrased loops *before* the next LLM call.
Instead of just killing the run, it uses a **Progressive Intervention Protocol**
(Nudge → Override → Hard Stop) to safely rewrite the transcript and force the
agent to pivot strategies.

> **< 20µs overhead per turn.** Zero network calls. 100% local.

## Key Features

*   **Progressive Intervention Protocol**: Escalates from a *Nudge* (soft coaching
    injection) to an *Override* (surgical transcript compaction) before a *Hard Stop*
    (clean termination with state preserved).
*   **Zero-Dependency Semantic Loop Detection**: Shingle-based fingerprinting catches
    paraphrased loops without embedding models, network calls, or external APIs.
*   **Atomic Transcript Surgery**: Removes orphaned tool-call transactions to prevent
    LLM API validation errors (400 Bad Request) before they happen.
*   **Local Budget Enforcement**: USD-denominated budget tracking per thread. No
    surprise "$4k Tuesday Morning" bills.
*   **Zero-Trust Privacy**: Every detection runs in your process. No telemetry,
    no prompts leave your RAM.

## Quick Start

TokenCircuit integrates with LangGraph through the framework's official extension
points—no monkey-patching required.

```python
from langgraph.prebuilt import create_react_agent
from tokencircuit.adapters.langgraph import tc_pre_model_hook, TokenCircuitToolNode

# 1. Wrap your tools with TokenCircuit's transaction tracking
safe_tool_node = TokenCircuitToolNode(tools)

# 2. Inject the pre-model hook for transcript surgery
agent = create_react_agent(
    model,
    tools=safe_tool_node,
    pre_model_hook=tc_pre_model_hook(),
)
```

The hook intercepts the agent *before* each LLM call, runs the intervention
pipeline (~20µs), and returns ephemeral message mutations. No graph state
is modified—the original checkpoint remains clean.

### Manual graph with named hooks

```python
from tokencircuit.adapters.langgraph import tc_pre_model_hook

builder.add_node(
    "agent", call_model,
    pre_model_hook=tc_pre_model_hook(config=my_config, node_name="agent"),
)
```

## Installation

```bash
pip install tokencircuit                    # Core engine
pip install "tokencircuit[langgraph]"       # + LangGraph adapter
pip install "tokencircuit[crewai]"          # + CrewAI adapter
```

## Performance

Benchmarked on the full intervention pipeline:

| Scenario | Latency |
|---|---|
| `decide()` hot path (PASS) | ~1.4 µs |
| `decide()` with NUDGE | ~3.5 µs |
| `process()` full pipeline | ~20 µs |
| `process()` with tool calls | ~33 µs |

Zero external embedding dependencies. All detection is local shingle-based
fingerprinting.

## Supported Frameworks

*   **LangGraph**: Native `pre_model_hook` integration. `tc_pre_model_hook()`
    factory and `TokenCircuitToolNode` for tool call transaction tracking.
*   **CrewAI**: Execution hook support (`crewai>=0.60`) for proactive
    intervention.
*   **OpenAI**: Standard function calling wrappers for raw LLM usage.

## Fleet Dashboard (Coming Soon)

A single pane of glass to monitor agentic reliability, tokens saved, and
intervention logs across your entire fleet.

---

<p align="center">
  Built for the 2026 Agentic Economy.
</p>
