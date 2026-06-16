"""OpenTelemetry tracing and Prometheus metrics for TokenCircuit."""

import logging
import threading

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


def get_tracer(name: str = "tokencircuit"):
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return None


__all__ = [
    "get_tracer",
    "MetricsCollector",
]
