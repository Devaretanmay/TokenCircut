# TokenCircuit

<p align="center">
  <img src="https://raw.githubusercontent.com/Devaretanmay/TokenCircut/main/docs/logo.png" alt="TokenCircuit Logo" width="200"/>
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

*   **Progressive Intervention**: Escalates from a Nudge (system message) to an Override (transcript surgery) before a Hard Stop.
*   **Atomic Transcript Surgery**: Surgically removes failing tool-call transactions to prevent LLM API validation errors.
*   **Semantic Loop Detection**: Zero-dependency shingle-based detection to catch paraphrased loops.
*   **Budget Enforcement**: Local USD-denominated budget tracking to prevent runaway costs.
*   **Zero-Trust Privacy**: Runs 100% locally in your RAM. No telemetry of prompts or data.

## Quick Start

TokenCircuit integrates with your existing graphs with exactly **one line of code**.

```python
from tokencircuit import instrument_langgraph, InterventionConfig

# Secure your graph with auto-recovery and a budget
config = InterventionConfig(
    max_budget_usd=0.50, 
    auto_recovery=True
)

# Instrument the builder before compiling
safe_graph = instrument_langgraph(builder, config=config).compile()
```

## Supported Frameworks

*   **LangGraph**: Native `pre_model_hook` integration.
*   **CrewAI**: Execution hook support for proactive intervention.
*   **OpenAI**: Standard function calling wrappers.

## Installation

```bash
pip install tokencircuit                    # Core
pip install "tokencircuit[langgraph]"       # + LangGraph support
pip install "tokencircuit[crewai]"          # + CrewAI support
```

## Fleet Dashboard (Coming Soon)

A single pane of glass to monitor agentic reliability across your entire fleet.

---

<p align="center">
  Built for the 2026 Agentic Economy.
</p>
