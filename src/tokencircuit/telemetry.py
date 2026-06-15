import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("tokencircuit")

MODEL_COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-4-sonnet": {"input": 0.003, "output": 0.015},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "unknown": {"input": 0.001, "output": 0.002},
}

SUPABASE_INGEST_URL = (
    "https://tokencircuit.supabase.co/functions/v1/ingest"
)


@dataclass
class TelemetryEvent:
    agency_id: str
    client_id: str
    agent_framework: str
    signal_type: str
    node_name: str
    iterations_at_detection: int
    model_name: str
    estimated_tokens_saved: int
    estimated_cost_saved_usd: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def compute_cost_estimate(
    model_name: str,
    iterations_saved: int,
    avg_tokens_per_call: int = 1024,
) -> tuple[int, float]:
    tokens_saved = iterations_saved * avg_tokens_per_call
    rates = MODEL_COST_PER_1K_TOKENS.get(
        model_name, MODEL_COST_PER_1K_TOKENS["unknown"]
    )
    avg_rate = (rates["input"] + rates["output"]) / 2
    cost_saved = (tokens_saved / 1000) * avg_rate
    return tokens_saved, round(cost_saved, 6)


def emit_event_async(
    event: TelemetryEvent,
    api_key: Optional[str] = None,
    ingest_url: Optional[str] = None,
) -> None:
    url = ingest_url or SUPABASE_INGEST_URL
    if not api_key:
        return

    def _send() -> None:
        try:
            import httpx

            httpx.post(
                url,
                json={
                    "agency_id": event.agency_id,
                    "client_id": event.client_id,
                    "agent_framework": event.agent_framework,
                    "signal_type": event.signal_type,
                    "node_name": event.node_name,
                    "iterations_at_detection": event.iterations_at_detection,
                    "model_name": event.model_name,
                    "estimated_tokens_saved": event.estimated_tokens_saved,
                    "estimated_cost_saved_usd": event.estimated_cost_saved_usd,
                    "timestamp": event.timestamp,
                },
                headers={"apikey": api_key},
                timeout=5.0,
            )
        except Exception:
            pass

    thread = threading.Thread(target=_send, daemon=True)
    thread.start()
