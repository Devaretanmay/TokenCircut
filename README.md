# TokenCircuit

[![PyPI](https://img.shields.io/pypi/v/tokencircuit)](https://pypi.org/project/tokencircuit/)
[![Python](https://img.shields.io/pypi/pyversions/tokencircuit)](https://pypi.org/project/tokencircuit/)
[![CI](https://github.com/Devaretanmay/TokenCircut/actions/workflows/ci.yml/badge.svg)](https://github.com/Devaretanmay/TokenCircut/actions)
[![License](https://img.shields.io/github/license/Devaretanmay/TokenCircut)](LICENSE)

**The Agentic Pre-Frontal Cortex.**

Current agent frameworks rely on blunt "Circuit Breakers" (like `recursion_limit`) that throw raw Python exceptions, destroying execution state and crashing the run when an agent loops. **TokenCircuit** is a zero-bloat, local-first SDK that intercepts the agent *before* the next LLM call, actively coaching it out of infinite loops and semantic stagnation.

Instead of failing the run, TokenCircuit forces a strategy pivot, turning bounded degradation into successful task completion.

Supported frameworks: **LangGraph**, **CrewAI**, **OpenAI function calling**.

## The Progressive Intervention Protocol

TokenCircuit operates silently in the background, intervening progressively:

| Stage | What Happens |
|-------|-------------|
| **PASS** | No intervention — agent proceeds normally |
| **NUDGE** | Ephemeral system message injected: *"You have tried this 3 times. Change strategy."* |
| **OVERRIDE** | Compacts failed transactions + injects forceful directive. Protects against OpenAI API validation errors. |
| **HARD_STOP** | Graceful fallback termination, preserving state dictionary for debugging. |

```python
from tokencircuit import InterventionConfig, LangGraphPreModelAdapter

adapter = LangGraphPreModelAdapter(
    config=InterventionConfig(
        nudge_threshold=3,
        override_threshold=5,
        hard_stop_threshold=8,
    )
)

# LangGraph pre_model_hook integration
graph.add_node("agent", call_model, pre_model_hook=adapter.hook)
```

## Installation

```bash
pip install tokencircuit                    # Core
pip install "tokencircuit[langgraph]"       # + LangGraph support
pip install "tokencircuit[crewai]"          # + CrewAI support
pip install "tokencircuit[otel]"            # + OpenTelemetry tracing
```

Requires Python ≥ 3.11.

## Quick Start — LangGraph

```python
from typing import Annotated
from langgraph.graph import StateGraph, MessagesState
from tokencircuit import (
    InterventionConfig,
    LangGraphPreModelAdapter,
    InterventionStateSchema,
    tc_state_reducer,
)

# 1. Create the adapter
adapter = LangGraphPreModelAdapter(
    config=InterventionConfig(
        nudge_threshold=3,
        override_threshold=5,
        hard_stop_threshold=8,
        audit_mode=False,          # Set True to monitor without intervening
        max_tokens_per_turn=4000,  # Runaway generation detection
    )
)

# 2. Define state with TokenCircuit channel
class AgentState(MessagesState):
    _tc_intervention: Annotated[InterventionStateSchema, tc_state_reducer]

# 3. Build graph with pre_model_hook
builder = StateGraph(AgentState)
builder.add_node("agent", call_model, pre_model_hook=adapter.hook)
graph = builder.compile()

# 4. Run — TokenCircuit handles the rest
async for step in graph.astream({"messages": [...]}, config):
    print(step)
```

## Quick Start — CrewAI

```python
from tokencircuit import instrument_crewai, InterventionConfig

config = InterventionConfig(
    nudge_threshold=3,
    override_threshold=5,
    hard_stop_threshold=8,
)
safe_crew = instrument_crewai(crew, config=config)
safe_crew.kickoff()  # Raises TokenCircuitError if loop detected
```

## Detection Signals

Six signal types evaluate the agent's behavior:

| Signal | Description |
|--------|-------------|
| `STATE_STAGNATION` | Identical content hash across the sliding window |
| `FUTILE_ACTION` | Same tool signature repeats with no progress |
| `SEMANTIC_STAGNATION` | Paraphrased repetition detected via token n-gram Jaccard similarity |
| `TRANSCRIPT_CORRUPTION` | Malformed tool calls or excessive orphaned results |
| `TOOL_TRANSACTION_ORPHAN` | Tool results without matching calls |
| `RUNAWAY_GENERATION` | Single AI turn exceeds token velocity limit |

## Enterprise Features

### Audit Mode

Monitor interventions without mutating the agent's behavior:

```python
config = InterventionConfig(audit_mode=True)
# Engine computes all signals and logs them, but always returns PASS
```

### Runaway Generation Detection

Catch agents that dump massive garbage output:

```python
config = InterventionConfig(max_tokens_per_turn=4000)
# Triggers immediate HARD_STOP if a single AI turn exceeds 4000 tokens
```

### OpenTelemetry Observability

```bash
pip install "tokencircuit[otel]"
```

TokenCircuit emits spans and events via `opentelemetry-api`:

```
TokenCircuit.Intervention
  ├── thread_id: "thread_abc"
  ├── node_name: "agent"
  ├── audit_mode: false
  ├── intervention.stage: "NUDGE"
  └── SignalDetected: "SEMANTIC_STAGNATION"
```

Visualize in Datadog, Grafana, or any OTel-compatible backend.

## Configuration

```python
from tokencircuit import InterventionConfig

config = InterventionConfig(
    # Escalation thresholds (consecutive stagnation turns)
    nudge_threshold=3,        # Turns before first coaching nudge
    override_threshold=5,     # Turns before forceful directive
    hard_stop_threshold=8,    # Turns before termination

    # Cooldown
    cooldown_turns=2,         # Turns to wait after de-escalation

    # Semantic detection
    window_size=5,            # Sliding window for fingerprint comparison
    similarity_threshold=0.92, # Jaccard similarity threshold
    enable_semantic_detection=True,

    # Transaction validation
    enable_transcript_validation=True,
    max_orphan_tolerance=2,
    auto_repair=True,

    # Enterprise
    audit_mode=False,
    max_tokens_per_turn=4000,
)
```

## Error Handling

```python
from tokencircuit import TokenCircuitError, StateStagnationError, FutileActionError

try:
    async for step in graph.astream(input, config):
        ...
except TokenCircuitError as e:
    print(f"Loop detected: {e}")
    print(f"Signal: {e.signal_type}")
    print(f"Node: {e.node_name}")
    print(f"Iteration: {e.iteration}")
```

## Architecture

```
LangGraph pre_model_hook → LangGraphPreModelAdapter
  └── InterventionEngine.process()
        ├── MessageCanonicalizer (normalize messages)
        ├── TranscriptValidator (enforce 10 invariants)
        ├── SemanticStagnationDetector (n-gram Jaccard)
        ├── Runaway Generation Check (token velocity)
        ├── Signal Aggregation
        └── decide() → PASS | NUDGE | OVERRIDE | HARD_STOP
```

- **Stateless**: Validators rebuild state from the transcript every turn
- **O(N)**: Intelligent caching avoids O(N²) transcript reprocessing
- **< 4ms P99**: Full pipeline latency under 4ms for 50-turn transcripts
- **Thread-safe**: Independent state per thread_id + node_name

## Development

```bash
pip install -e ".[dev,langgraph,openai,otel]"
make check
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
