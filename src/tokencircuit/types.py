"""Core type definitions for TokenCircuit."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class InterventionStage(enum.IntEnum):
    """Progressive intervention stages."""

    PASS = 0
    NUDGE = 1
    OVERRIDE = 2
    HARD_STOP = 3


class SignalType(enum.Enum):
    """Detection signal taxonomy."""

    STATE_STAGNATION = "STATE_STAGNATION"
    FUTILE_ACTION = "FUTILE_ACTION"
    SEMANTIC_STAGNATION = "SEMANTIC_STAGNATION"
    TRANSCRIPT_CORRUPTION = "TRANSCRIPT_CORRUPTION"
    TOOL_TRANSACTION_ORPHAN = "TOOL_TRANSACTION_ORPHAN"
    RUNAWAY_GENERATION = "RUNAWAY_GENERATION"


class TransactionStatus(enum.Enum):
    """Lifecycle state of a tool call transaction."""

    PENDING = "pending"
    COMMITTED = "committed"
    ORPHANED = "orphaned"
    DUPLICATE = "duplicate"


class TransactionOutcome(enum.Enum):
    """Classification of a committed tool result's content."""

    SUCCESS = "success"
    EMPTY = "empty"
    TRANSIENT_ERROR = "transient_error"
    PERMANENT_ERROR = "permanent_error"
    UNKNOWN = "unknown"


class CanonicalRole(enum.Enum):
    """Normalized message roles across LLM providers."""

    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Immutable record of a single tool invocation."""

    call_id: str
    tool_name: str
    source_message_index: int
    turn_number: int


@dataclass(frozen=True, slots=True)
class ToolResultRecord:
    """Immutable record of a tool execution result."""

    call_id: str
    result_content_prefix: str = ""
    result_length: int = 0
    source_message_index: int = -1
    turn_number: int = 0
    outcome: TransactionOutcome = TransactionOutcome.UNKNOWN


@dataclass(frozen=True, slots=True)
class ToolTransaction:
    """A complete tool call transaction: call + optional result."""

    call: ToolCallRecord
    result: ToolResultRecord | None = None
    status: TransactionStatus = TransactionStatus.PENDING
    outcome: TransactionOutcome = TransactionOutcome.UNKNOWN
    committed_at_turn: int | None = None


@dataclass(frozen=True, slots=True)
class SemanticFingerprint:
    """Captures the semantic essence of a turn for stagnation comparison."""

    turn_number: int
    content_hash: str
    tool_signature: str
    structural_pattern: str
    bigram_set: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    trigram_set: frozenset[tuple[str, str, str]] = field(default_factory=frozenset)


@dataclass(slots=True)
class InterventionContext:
    """Complete context passed to InterventionEngine.decide()."""

    thread_id: str
    node_name: str
    turn_number: int
    active_signals: list[SignalType] = field(default_factory=list)
    semantic_similarity_score: float = 0.0
    orphaned_transaction_ids: list[str] = field(default_factory=list)
    dropped_this_turn: list[str] = field(default_factory=list)
    consecutive_empty_results: int = 0
    consecutive_errors: int = 0
    current_stage: InterventionStage = InterventionStage.PASS
    consecutive_stagnation_count: int = 0
    total_interventions: int = 0
    cooldown_remaining: int = 0
    strategies_attempted: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InterventionDecision:
    """Output of InterventionEngine.decide()."""

    stage: InterventionStage
    signals: list[SignalType] = field(default_factory=list)
    llm_input_messages: list[dict[str, Any]] | None = None
    coaching_message: str | None = None
    state_patch: dict[str, Any] = field(default_factory=dict)
    should_terminate: bool = False
    termination_reason: str | None = None
    estimated_tokens_saved: int = 0


@dataclass(slots=True)
class CanonicalMessage:
    """Normalized message representation."""

    role: CanonicalRole
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    source_index: int = -1
    name: str | None = None
