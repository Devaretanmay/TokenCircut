# LangGraph Interceptor

## Overview

The LangGraph interceptor is the core integration point. It wraps a `CompiledStateGraph` so that every step the agent takes passes through detection before reaching the caller. If a loop is detected, the graph stops mid-execution and the checkpointed state is preserved.

The critical architectural decision was using `astream` instead of `astream_events`. This choice determines whether interruption actually works or silently fails.

## Code Walkthrough

### `astream` — The interception point

```python
async def astream(self, input, config=None, **kwargs):
    stream = self._graph.astream(input, config, **kwargs)
    async for step_output in stream:
```

The outer `astream` delegates to the inner graph's `astream`. Each yielded step is intercepted before being passed to the caller.

```python
        node_name = None
        state = None
        if isinstance(step_output, dict) and self._node_names:
            for n in self._node_names:
                if n in step_output:
                    node_name = n
                    state = step_output[n]
                    break
```

`astream` yields step outputs as `{node_name: node_output}` dicts. We match the output key against the list of known nodes to identify which node just ran and extract its output.

```python
        action_hash = compute_action_hash(state)
        tool_call = self._extract_tool_call(state)
        tool_sig = extract_tool_type_signature(tool_call)
```

Each step produces a fingerprint: an action hash (what happened) and a tool signature (what tool was called). These go into the ring buffer.

```python
        result = self._detector.evaluate(agent_id, node_name, buffer)
        if result is not None:
            msg = self._on_detection(result, agent_id, model_name)
            raise TokenCircuitError(msg)
```

On detection, the interceptor raises `TokenCircuitError` from inside the `async for` loop. With `astream`, this correctly stops the graph. With `astream_events`, it would only stop the consumer — the graph keeps running.

### Why `astream` works and `astream_events` doesn't

The difference is how each API manages graph execution:

```
astream:
┌──────────┐   yield step 1   ┌──────────┐
│  Graph   │ ───────────────> │ Consumer │
│          │   wait for next  │          │
│          │ <─────────────── │          │
│          │   yield step 2   │          │
│          │ ───────────────> │          │
└──────────┘                  └──────────┘

astream_events:
┌──────────┐   background task ───┐
│  Graph   │                       │
│ (runs to │  event queue ───────┐ │
│  finish) │                     │ │
└──────────┘                     ▼ ▼
                          ┌──────────────┐
                          │   Consumer   │
                          │ (can raise,  │
                          │  graph gone) │
                          └──────────────┘
```

With `astream`, the graph yields control to the consumer at each step. The graph's state machine advances one tick, produces output, and waits. If the consumer raises, the generator's caller unwinds and the graph's state machine stops.

With `astream_events`, the graph runs in a background `asyncio.Task`. Events are pushed to an `asyncio.Queue` that the consumer reads from. The background task runs the graph to completion (or recursion limit) regardless of what the consumer does. Raising in the consumer only stops the queue reader — the graph's state machine continues executing, calling LLMs and burning tokens.

### `astream_events` passthrough

```python
    async def astream_events(self, input, config, **kwargs):
        async for event in self._graph.astream_events(input, config, **kwargs):
            yield event
```

The interceptor still exposes `astream_events` as a passthrough to the underlying graph. This is for observability tooling (LangSmith, callbacks) that consume events but don't need control flow. The warning in the README explains: if you need loop detection, use `astream`.

### `__getattr__` delegation

```python
    def __getattr__(self, name):
        return getattr(self._graph, name)
```

Any method not explicitly defined on the interceptor falls through to the underlying graph. This makes the interceptor a transparent proxy — tools that inspect the graph (LangSmith callbacks, `get_graph()`, `get_state()`) work without modification.

This is the Delegation pattern from the Gang of Four. The interceptor pretends to be the graph while adding behavior around `astream`.

## Concepts Explained

### Generator / Async Iterator

**What**: A function that yields multiple values over time, maintaining state between yields.

**Why**: `astream` is an async generator. Each `yield` pauses execution and returns control to the caller. The next `async for` iteration resumes from where it left off. This is what makes interruption possible — the graph is literally stopped between yields, waiting for the next `next()` call.

**When**: Any streaming data pipeline where you need backpressure or per-item control. HTTP SSE streams, file readers, WebSocket message handlers.

**Alternatives**: Callbacks (inverted control, harder to reason about), promises/futures (one-shot), queues (over-engineered for synchronous yield patterns).

### Proxy / Delegation Pattern

**What**: An object that wraps another object and forwards most operations, adding behavior to specific methods.

**Why**: Lets you instrument a complex object without modifying it. The interceptor adds loop detection to `astream` while leaving every other method untouched. Callers that inspect the graph for visualization, debugging, or state access continue to work.

**When**: AOP-style concerns (logging, metrics, caching, rate limiting) applied to an existing object. ORM session wrappers, API client retry wrappers, cache-aside proxies.

**Alternatives**: Monkey-patching (fragile, hard to trace), inheritance (tight coupling, can't wrap instances), decorators (per-function, not per-object).

### Thread ID as Agent Identifier

```python
    agent_id = config.get("configurable", {}).get("thread_id", "default_agent")
```

LangGraph's `thread_id` in the config serves double duty: it identifies the conversation thread for checkpointing, and it identifies the agent for loop detection. Multiple agents using the same graph with different thread IDs get independent buffers.

### Node Discovery

```python
    def _discover_nodes(self):
        g = self._graph.get_graph()
        for node_id, node_data in g.nodes.items():
            name = node_data.name if hasattr(node_data, "name") else node_id
            if name and not name.startswith("__"):
                self._node_names.add(name)
```

The interceptor introspects the compiled graph to learn which nodes exist. This lets it match step outputs to node names without the user having to declare them. The `__` prefix filter skips LangGraph's internal nodes (entry point, condition edges).

## Learning Resources

- [LangGraph streaming guide](https://langchain-ai.github.io/langgraph/how-tos/streaming/) — Official docs on astream vs astream_events
- [Python async generators (PEP 525)](https://www.python.org/dev/peps/pep-0525/) — The language spec
- [Proxy pattern (refactoring.guru)](https://refactoring.guru/design-patterns/proxy) — The design pattern behind transparent wrapping
- [Gang of Four: Proxy pattern](https://en.wikipedia.org/wiki/Proxy_pattern) — Original academic reference
- [asyncio queues vs generators](https://vorpus.org/blog/some-thoughts-on-asynchronous-api-design/) — Great blog post on API design trade-offs

## Related Code

- `src/tokencircuit/interceptors/langgraph.py`
- `src/tokencircuit/__init__.py`
- `tests/integration/test_langgraph.py`
- `tests/stress_test_real_world.py`
