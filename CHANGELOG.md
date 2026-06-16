# Changelog

## 0.2.0 (2026-06-15)

### Core Architectural Overhaul (V7)
- **100% Stateless Engine**: Deprecated the V6 `RingBuffer` and stateful detectors. The engine now operates entirely as a pure function `InterventionEngine.process()`, hydrating all necessary context from a unified `_tc_intervention` state schema or `StateGraph` history.
- **Unified Adapter Protocol**: Standardized the `LangGraphPreModelAdapter` and `CrewAIInterventionAdapter` to ensure consistent pre-model hooking. Deprecated `instrument_langgraph` private API usage.
- **O(1) Memory Tracking**: Replaced O(N) array scans and infinite cache keys with strict `LRU` eviction caches. Memory overhead is capped explicitly via `InterventionConfig.max_threads`.
- **Immutable Tool Transaction Ledger**: Implemented `ToolTransactionLedger` with a strict lifecycle model (PENDING -> COMMITTED or ORPHANED) to eliminate dangling requests and handle incremental hydration gracefully.
- **Semantic Stagnation Detection Engine**: Moved beyond exact-hash matching. V7 introduces a tiktoken-powered Jaccard similarity detector to catch structurally and semantically paraphrased infinite loops. Includes `O(N^2)` shingle filtering elimination and caching optimizations.
- **Config Convergence**: Merged `TokenCircuitConfig` entirely into `InterventionConfig`. Legacy fields are aliased and fire deprecation warnings to maintain backward compatibility.
- **Multi-Level Escalation Ladder**: Replaced binary signals with a progressive coaching model:
    - `PASS`: No intervention.
    - `NUDGE`: Soft suggestion using `nudge_template`.
    - `OVERRIDE`: Forceful pivot directive.
    - `HARD_STOP`: Guaranteed process termination.
- **Prometheus Metrics**: Integrated OpenTelemetry and Prometheus for explicit reporting of total interventions, tokens saved, and stagnation scores.
- **Optional Dependencies**: Moved `tiktoken` and `prometheus-client` to optional dependency layers.

## 0.1.0 (2025-06-15)

- Initial release
- LangGraph interceptor (`instrument_langgraph`)
- CrewAI interceptor (`instrument_crewai`)
- OpenAI client wrapper (`TokenCircuitClient`)
- Two detection signals: StateStagnation and FutileAction
- Sliding window ring buffer
- Configurable thresholds and window size
- Remote config via Supabase
- Telemetry with cost estimation
- Control plane dashboard (Next.js)
