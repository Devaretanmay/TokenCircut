# TokenCircuit

<p align="center">
  <img src="docs/assets/logo.png" alt="TokenCircuit Logo" width="200"/>
</p>

<p align="center">
  <b>The Agentic Pre-Frontal Cortex.</b>
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
</p>

---

Current agent frameworks rely on blunt "Circuit Breakers" (like `recursion_limit`) that throw raw Python exceptions, destroying execution state and crashing the run when an agent loops. 

**TokenCircuit** is a zero-bloat, local-first SDK that intercepts the agent *before* the next LLM call, actively coaching it out of infinite loops and semantic stagnation. It acts as an external **Pre-Frontal Cortex** for your agents, allowing them to self-correct and complete tasks instead of burning your API budget.

## Key Features

*   🧠 **Progressive Intervention**: Escalates from a *Nudge* (soft prompt injection) to an *Override* (surgical transcript surgery) before a *Hard Stop*.
*   🔪 **Atomic Transcript Surgery**: Surgically removes failing tool-call transactions to prevent LLM API validation errors (400 Bad Request).
*   🔍 **Semantic Loop Detection**: Zero-dependency shingle-based detection via `tiktoken` to catch paraphrased loops without external embedding models.
*   💰 **Budget Enforcement**: Local USD-denominated budget tracking and enforcement to prevent the "Tuesday Morning $4k Bill".
*   🔒 **Zero-Trust Privacy**: Runs 100% locally in your RAM. No telemetry of prompts, outputs, or private data.

## Quick Start

TokenCircuit integrates with your existing LangGraph builder with exactly **one line of code**.

```python
from tokencircuit import instrument_langgraph, InterventionConfig

# Configure safety thresholds and budget
config = InterventionConfig(
    max_budget_usd=0.50, 
    auto_recovery=True
)

# Instrument your graph builder
instrument_langgraph(builder, config=config)

# Compile and run as normal
graph = builder.compile()
```

## Supported Frameworks

*   **LangGraph**: Native `pre_model_hook` integration for deep safety.
*   **CrewAI**: Execution hook support for proactive intervention.
*   **OpenAI**: Standard function calling wrappers for raw LLM usage.

## Installation

```bash
pip install tokencircuit                    # Core engine
pip install "tokencircuit[langgraph]"       # + LangGraph adapter
pip install "tokencircuit[crewai]"          # + CrewAI adapter
```

## Fleet Dashboard (Coming Soon)

A single pane of glass to monitor agentic reliability, tokens saved, and intervention logs across your entire fleet.

---

<p align="center">
  Built for the 2026 Agentic Economy.
</p>
