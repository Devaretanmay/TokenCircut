# TokenCircuit

[![PyPI](https://img.shields.io/pypi/v/tokencircuit)](https://pypi.org/project/tokencircuit/)
[![Python](https://img.shields.io/pypi/pyversions/tokencircuit)](https://pypi.org/project/tokencircuit/)
[![CI](https://github.com/Devaretanmay/TokenCircut/actions/workflows/ci.yml/badge.svg)](https://github.com/Devaretanmay/TokenCircut/actions)
[![License](https://img.shields.io/github/license/Devaretanmay/TokenCircut)](LICENSE)

Detect and interrupt infinite loops in LLM agentic workflows. Supported frameworks: LangGraph, CrewAI, OpenAI.

```python
from tokencircuit import instrument_langgraph

safe = instrument_langgraph(graph)
async for step in safe.astream(input, config):
    ...
```

If the agent loops, `TokenCircuitError` is raised. Checkpointed state preserved for inspection.

## Installation

```bash
pip install tokencircuit
pip install "tokencircuit[langgraph]"   # LangGraph support
pip install "tokencircuit[crewai]"      # CrewAI support
pip install "tokencircuit[openai]"      # OpenAI support
```

Requires Python >= 3.11.

## Quick start

```python
from tokencircuit import (
    TokenCircuitConfig,
    TokenCircuitError,
    instrument_langgraph,
)

config = TokenCircuitConfig(max_repeats=5, window_size=5)
safe = instrument_langgraph(graph, config=config)

try:
    async for step in safe.astream({"messages": [...]}, config):
        ...
except TokenCircuitError as e:
    print(f"Loop detected: {e}")
    state = graph.get_state(config)
```

## Detection signals

Two detectors evaluate a sliding window of action fingerprints:

- **State Stagnation** — state hash and tool signature identical across N consecutive steps. Agent produces same output every time. (Priority signal.)
- **Futile Action** — tool call signature repeats but state hash changes. Agent calls same tool, gets different results — state moves but unproductively.

Detection fires within 5-10 iterations depending on how quickly the LLM settles into the loop pattern.

## Architecture

```
User app → instrument_langgraph(graph) → interceptor
  astream() yields → compute action hash + tool signature
                  → push to ring buffer
                  → CompositeDetector.evaluate()
                  → raise TokenCircuitError if triggered
```

## Why `astream` not `astream_events`

`astream` yields each step synchronously — the graph pauses between steps and waits for the consumer. Raising inside the `async for` loop stops the graph.

`astream_events` runs the graph in a background task. Events are fire-and-forget; the graph continues regardless of what the consumer does.

TokenCircuit only wraps `astream`. Use the unwrapped graph for `astream_events`.

## Configuration

```python
config = TokenCircuitConfig(
    max_repeats=10,
    window_size=10,
    model_name="gpt-4",
    telemetry_enabled=True,
)
```

## Error handling

```python
from tokencircuit import TokenCircuitError, StateStagnationError, FutileActionError

try:
    async for step in safe.astream(input, config):
        ...
except StateStagnationError:
    ...  # Agent stuck producing same output
except FutileActionError:
    ...  # Agent calling same tool, no progress
```

## Development

```bash
pip install -e ".[dev,langgraph,openai]"
make check
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
