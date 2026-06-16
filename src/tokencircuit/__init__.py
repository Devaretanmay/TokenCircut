"""TokenCircuit — Pre-Model Intervention Engine."""

import importlib.metadata

from .adapters.langgraph import LangGraphPreModelAdapter
from .canonicalizer import MessageCanonicalizer
from .engine import InterventionConfig, InterventionEngine, TokenCircuitError
from .instrumentation import instrument_crewai, instrument_langgraph
from .ledger import ToolTransactionLedger
from .semantic_detector import SemanticStagnationDetector
from .state_schema import (
    InterventionStateSchema,
    default_intervention_state,
    tc_state_reducer,
)

try:
    __version__ = importlib.metadata.version("tokencircuit")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = [
    "__version__",
    "CanonicalRole",
    "InterventionStage",
    "SignalType",
    "TransactionOutcome",
    "TransactionStatus",
    "InterventionStateSchema",
    "default_intervention_state",
    "tc_state_reducer",
    "MessageCanonicalizer",
    "ToolTransactionLedger",
    "TranscriptValidator",
    "SemanticStagnationDetector",
    "InterventionEngine",
    "InterventionConfig",
    "TokenCircuitError",
    "instrument_langgraph",
    "instrument_crewai",
    "LangGraphPreModelAdapter",
]
