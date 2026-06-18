"""TokenCircuit — Pre-Model Intervention Engine."""

from __future__ import annotations

import importlib.metadata

try:
    from .adapters.langgraph import tc_pre_model_hook, tc_wrap_tool_call
except ImportError:
    tc_pre_model_hook = None  # type: ignore[assignment]
    tc_wrap_tool_call = None  # type: ignore[assignment]

from .canonicalizer import MessageCanonicalizer
from .engine import InterventionConfig, InterventionEngine, TokenCircuitError
from .ledger import ToolTransactionLedger
from .semantic_detector import SemanticStagnationDetector
from .state_schema import (
    InterventionStateSchema,
    default_intervention_state,
    tc_state_reducer,
)
from .types import (
    CanonicalRole,
    InterventionStage,
    SignalType,
    TransactionOutcome,
    TransactionStatus,
)
from .validator import TranscriptValidator

try:
    __version__ = importlib.metadata.version("tokencircuit")
except importlib.metadata.PackageNotFoundError:
    __version__ = "8.0.0"

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
    "tc_pre_model_hook",
    "tc_wrap_tool_call",
]
