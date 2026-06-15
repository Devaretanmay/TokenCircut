# Changelog

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
