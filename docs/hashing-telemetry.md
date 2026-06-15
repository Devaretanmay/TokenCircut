# State Hashing & Telemetry

## Overview

Two supporting systems make the detection pipeline operational. The hashing layer converts agent state into repeatable fingerprints that the detectors compare. The telemetry layer emits cost-saving metrics to a control plane when loops are detected.

Neither is visible to the end user, but both determine whether detection is accurate and whether you can measure its business impact.

## Code Walkthrough

### `hash_utils.py` — The fingerprint system

The hashing layer evolved through two iterations:

**Iteration 1 — `compute_state_hash`**

```python
def compute_state_hash(state: dict[str, Any]) -> str:
    filtered = {k: v for k, v in state.items()
                if not any(x in k.lower() for x in EXCLUDED_KEYS)}
    serialized = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
```

This hashes the **entire state dict**. The excluded keys (`timestamp`, `trace_id`, `_meta`, `_tc_`) strip metadata that would make every hash unique even when the agent is looping.

But this approach has a fatal flaw for agent monitoring: state accumulates. Each iteration adds messages to `state["messages"]`, so the hash changes every time even when the agent is doing the same thing. A detector comparing hashes across iterations would never see a match.

**Iteration 2 — `compute_action_hash`**

```python
def compute_action_hash(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
```

Instead of hashing the full state, we extract just the **last action pair**: the most recent AI message (with tool calls) and the most recent tool response. This gives us a fingerprint of what the agent just did, invariant to accumulated history.

```python
    fingerprint = {}
    if tool_call:
        tc_d = _to_dict(tool_call) if not isinstance(tool_call, dict) else tool_call
        fingerprint["tool_name"] = tc_d.get("name", "unknown")
        fingerprint["tool_args"] = _serializable(tc_d.get("args", {}))
    if tool_content is not None:
        stable = tool_content[-200:]
        stable = "".join(ch for ch in stable if not ch.isdigit())
        fingerprint["tool_result"] = stable
```

The result snippet is truncated to 200 characters and has digits stripped. This prevents false negatives from non-semantic variation like `"timeout after 3001ms"` vs `"timeout after 2998ms"` — both reduce to `"timeout after ms"` and produce the same hash.

**Serialization helpers**

```python
def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj

def _serializable(obj: Any) -> Any:
```

LangChain messages (`AIMessage`, `ToolMessage`) are Pydantic models with `model_dump()`. The `_to_dict` helper normalizes them to plain dicts. The `_serializable` helper recursively converts any nested objects to JSON-safe types, falling back to `str()` for anything it can't handle (datetimes, UUIDs, enums, numpy arrays).

### `telemetry.py` — The cost tracker

**Cost estimation**

```python
MODEL_PRICING = {
    "gpt-4o":         {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
}

def compute_cost_estimate(model_name, iterations_saved):
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["gpt-4o"])
    input_cost = (pricing["input"] / 1_000_000) * avg_tokens * 0.6 * iterations_saved
    output_cost = (pricing["output"] / 1_000_000) * avg_tokens * 0.4 * iterations_saved
```

Pricing from the model provider's API page (as of 2025). The 60/40 input/output split is a reasonable default for agentic workloads, which tend to be output-heavy due to tool call generations.

**Async event emission**

```python
def emit_event_async(event, api_key):
    thread = threading.Thread(
        target=_emit_sync, args=(event, api_key), daemon=True
    )
    thread.start()
```

Telemetry must not block the agent's execution. A daemon thread fires the HTTP POST to the control plane without waiting. If the network request hangs or fails, the thread terminates silently when the process exits.

## Concepts Explained

### Content-Addressed Fingerprinting

**What**: A hash constructed from the *semantic content* of an action rather than the raw bytes.

**Why**: Two identical agent actions may produce slightly different byte representations (different timestamps, different request IDs, different serialization order). A content-addressed fingerprint strips this noise and produces the same hash for semantically identical actions.

**When**: Any system that needs to detect repetition in noisy data. Deduplication, change detection, plagiarism checkers, cache keys.

**Alternatives**: Exact byte comparison (brittle), semantic embedding comparison (expensive, slow), rule-based pattern matching (brittle, hard to maintain).

### Model Pricing Strategy

**What**: Hardcoded per-model pricing tables updated periodically from provider APIs.

**Why**: Token costs vary by provider and model by orders of magnitude. A gpt-4o-mini call is ~16x cheaper than gpt-4o. Hardcoding the pricing lets the cost estimate be computed offline without API calls.

**When**: Any system that needs to estimate costs without actually billing. Cost projections, savings reports, budget alerts.

**Trade-off**: Pricing changes over time. The table must be updated when providers change prices. An alternative would be to fetch pricing from an API, but that adds latency and a dependency.

## Learning Resources

- [SHA-256 explained (Computerphile)](https://www.youtube.com/watch?v=DMtFhACPnTY) — Visual explanation of cryptographic hashing
- [JSON serialization in Python](https://docs.python.org/3/library/json.html) — Official docs for `json.dumps` and `default=str`
- [LangChain message serialization](https://python.langchain.com/docs/how_to/#messages) — How AIMessage/ToolMessage serialize
- [OpenAI pricing page](https://openai.com/pricing) — Current model costs
- [Anthropic pricing page](https://docs.anthropic.com/en/docs/about-claude/pricing) — Current model costs
- [Python threading daemon threads](https://docs.python.org/3/library/threading.html#thread-objects) — How daemon threads work and when to use them

## Related Code

- `src/tokencircuit/otel/hash_utils.py`
- `src/tokencircuit/telemetry.py`
- `tests/test_hash_utils.py`
- `tests/test_telemetry.py`
