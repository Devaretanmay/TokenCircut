# TokenCircuit

TokenCircuit detects and interrupts infinite loops in LLM agentic workflows, preventing runaway costs when agents get stuck in repetitive cycles.

## Quick Start

```python
from tokencircuit import instrument_langgraph

# Wrap your compiled LangGraph
safe_graph = instrument_langgraph(graph)

# Use it like the original graph
async for step in safe_graph.astream(input, config):
    # An TokenCircuitError is raised automatically
    # when a loop is detected (state stagnation or futile action)
    ...
```

## Why `astream` (Not `astream_events`)

TokenCircuit wraps a LangGraph `CompiledStateGraph` using `astream` — the synchronous-per-step streaming API. This is a deliberate choice with an important architectural implication:

- `astream` yields each step's output synchronously. The graph pauses between steps and waits for the consumer to pull the next item. Raising an exception inside the `async for` loop properly stops graph execution.
- `astream_events` runs the graph in a background task. Events are fire-and-forget; the graph continues executing regardless of what the consumer does. Raising an exception inside an `async for event in astream_events(...)` loop only stops the consumer — the graph keeps running silently until it hits the recursion limit.

If you wrap a graph and only expose `astream_events`, your loop detector will appear to work during testing (the consumer stops) but the graph continues burning tokens in the background. TokenCircuit only wraps `astream`. Consumers that need `astream_events` for observability should use the original unwrapped graph for that purpose.

## How It Works

TokenCircuit monitors each node's output as it streams through `astream`. It maintains a ring buffer of recent action fingerprints for each (agent, node) pair. Two detectors evaluate the buffer, with a priority hierarchy:

- **State Stagnation (priority)** — fires when both the action hash and tool signature remain identical across N consecutive steps. This is the more severe condition: the agent is producing exactly the same output every time, with no variation in either action or result.
- **Futile Action** — fires when the tool call signature repeats but the action hash changes. The agent is calling the same tool, but getting different results each time — state is moving, but the loop is still unproductive.

State stagnation takes priority because if the state isn't changing at all, that's strictly more broken than a futile-but-moving loop. In practice, a deterministically identical failure (like a 403 error with the same body) will fire STATE_STAGNATION, not FUTILE_ACTION. This is the correct label — the tool result isn't varying.

Detection typically fires 5–10 iterations into the loop, depending on how quickly the LLM settles into the repeating pattern. The buffer requires N identical entries before raising, so initial attempts with slightly varied arguments before the pattern locks in are tolerated without false positives.

When a loop is detected, `TokenCircuitError` (a `RuntimeError` subclass) is raised, stopping the stream. Checkpointed state is preserved for inspection and optional resumption.

## Installation

```bash
pip install tokencircuit
```

Or with LangGraph support:

```bash
pip install "tokencircuit[langgraph]"
```

## Configuration

```python
from tokencircuit import TokenCircuitConfig
from tokencircuit import instrument_langgraph

config = TokenCircuitConfig(
    max_repeats=10,      # max window size for detection
    window_size=10,      # number of steps to look back
    model_name="gpt-4",  # used for cost estimation
    telemetry_enabled=True,
    agency_id="my-org",
    client_id="my-app",
)

safe = instrument_langgraph(graph, config=config)
```

Remote configuration can be loaded from Supabase by passing an API key:

```python
from tokencircuit import instrument_langgraph
from tokencircuit.config import load_config

config = load_config(api_key="your-supabase-key")
safe = instrument_langgraph(graph, config=config)
```

## Error Handling

```python
from tokencircuit import TokenCircuitError

try:
    async for step in safe.astream(input, config):
        ...
except TokenCircuitError as e:
    print(f"Loop detected: {e}")
    # Checkpointed state is available via graph.get_state(config)
    state = graph.get_state(config)
```

## Telemetry

When configured with `agency_id`, `client_id`, and an API key, TokenCircuit emits telemetry events to the control plane on each detection, including estimated tokens and cost saved.

## Architecture

```
┌─────────────────────────────────────────────┐
│               User Application              │
├─────────────────────────────────────────────┤
│  instrument_langgraph(graph) → interceptor  │
├─────────────────────────────────────────────┤
│  astream() → for each step output:          │
│    1. Compute action hash + tool signature  │
│    2. Push to ring buffer                   │
│    3. CompositeDetector.evaluate()          │
│    4. Raise TokenCircuitError if triggered  │
└─────────────────────────────────────────────┘
```

## Development

```bash
git clone https://github.com/anomalyco/tokencircuit.git
cd tokencircuit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,langgraph]"
pytest
```

## License

MIT
