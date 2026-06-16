"""Telemetry events and cost estimation for TokenCircuit."""

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("tokencircuit")


try:
    from prometheus_client import Counter, Histogram
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """Singleton for Prometheus metrics collection."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsCollector, cls).__new__(cls)
                cls._instance._init_metrics()
            return cls._instance

    def _init_metrics(self) -> None:
        self.enabled = _PROMETHEUS_AVAILABLE
        if self.enabled:
            self.interventions_total = Counter(
                "tokencircuit_interventions_total",
                "Total interventions triggered",
                ["stage", "model"]
            )
            self.tokens_saved_total = Counter(
                "tokencircuit_tokens_saved_total",
                "Total tokens saved by interventions",
                ["model"]
            )
            self.stagnation_score = Histogram(
                "tokencircuit_stagnation_score",
                "Semantic stagnation Jaccard similarity score",
                buckets=[0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
            )

    def record_intervention(self, stage: str, model: str, tokens_saved: int = 0) -> None:
        if self.enabled:
            self.interventions_total.labels(stage=stage, model=model).inc()
            if tokens_saved > 0:
                self.tokens_saved_total.labels(model=model).inc(tokens_saved)

    def record_stagnation_score(self, score: float) -> None:
        if self.enabled:
            self.stagnation_score.observe(score)


__all__ = [
    "TelemetryEvent",
    "compute_cost_estimate",
    "emit_event_async",
    "get_tracer",
]

def get_tracer(name: str = "tokencircuit"):
    """
    Get an OpenTelemetry tracer if the API is installed.
    Returns None if opentelemetry-api is not available.
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return None

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


_telemetry_queue: queue.Queue[tuple[TelemetryEvent, str, str]] = queue.Queue(maxsize=1000)
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()


def _telemetry_worker() -> None:
    import httpx

    # Reuse client across requests
    with httpx.Client(timeout=5.0) as client:
        while True:
            try:
                event, api_key, url = _telemetry_queue.get()

                client.post(
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
                )
            except Exception:
                logger.debug("TokenCircuit: telemetry emit failed")
            finally:
                _telemetry_queue.task_done()


def emit_event_async(
    event: TelemetryEvent,
    api_key: Optional[str] = None,
    ingest_url: Optional[str] = None,
) -> None:
    global _worker_thread
    url = ingest_url or SUPABASE_INGEST_URL
    if not api_key:
        return

    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_telemetry_worker, daemon=True)
            _worker_thread.start()

    try:
        _telemetry_queue.put_nowait((event, api_key, url))
    except queue.Full:
        logger.warning("TokenCircuit: telemetry queue full, dropping event")
