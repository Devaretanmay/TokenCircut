# TokenCircuit

Detect and interrupt infinite loops in LLM agentic workflows.

```python
from tokencircuit import instrument_langgraph

safe = instrument_langgraph(graph)
async for step in safe.astream(input, config):
    ...
```

If the agent enters a repetitive loop, `TokenCircuitError` is raised and graph execution stops. Checkpointed state is preserved for inspection.

## Why `astream` not `astream_events`

`astream` yields each step synchronously — the graph pauses between steps and waits for the consumer. Raising inside the `async for` loop stops the graph.

`astream_events` runs the graph in a background task. Events are fire-and-forget; the graph continues executing regardless of what the consumer does. Raising inside an event loop only stops the consumer — the graph keeps burning tokens until it hits the recursion limit.

TokenCircuit only wraps `astream`. Use the unwrapped graph for `astream_events` observability.

## Detection Signals

Two detectors evaluate a sliding window of action fingerprints:

- **State Stagnation** — state hash and tool signature are identical across N consecutive steps. The agent is producing the exact same output every time. (Priority signal — more severe condition.)
- **Futile Action** — tool call signature repeats but state hash changes. The agent is calling the same tool but getting different results. State is moving, but unproductively.

When a deterministically identical failure occurs (same 403 error body, same null result), `STATE_STAGNATION` fires — not `FUTILE_ACTION`. This is correct: if the result isn't varying, stagnation is the accurate label.

Detection typically fires within 5–10 iterations depending on how quickly the LLM settles into the loop pattern. Initial varied attempts are tolerated without false positives.

## Installation

```bash
pip install tokencircuit
pip install "tokencircuit[langgraph]"   # LangGraph support
```

## Configuration

```python
from tokencircuit import TokenCircuitConfig, instrument_langgraph

config = TokenCircuitConfig(
    max_repeats=10,
    window_size=10,
    model_name="gpt-4",
    telemetry_enabled=True,
)

safe = instrument_langgraph(graph, config=config)
```

Remote config from Supabase:

```python
from tokencircuit.config import load_config
config = load_config(api_key="your-key")
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
    state = graph.get_state(config)  # preserved after interrupt
```

## Telemetry

When configured with `agency_id`, `client_id`, and an API key, detection events (including estimated tokens/cost saved) are emitted to the control plane.

## Architecture

```
User app → instrument_langgraph(graph) → interceptor
  astream() yields → compute action hash + tool signature
                  → push to ring buffer
                  → CompositeDetector.evaluate()
                  → raise TokenCircuitError if triggered
```

## Development

```bash
pip install -e ".[dev,langgraph]"
pytest
```

## License

MIT
