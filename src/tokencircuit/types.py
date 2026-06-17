"""Core type definitions for TokenCircuit V7."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Enumerations
# =============================================================================


class InterventionStage(enum.IntEnum):
    """
    Progressive intervention stages.
    IntEnum allows direct comparison: NUDGE > PASS, etc.
    """

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
    """
    Classification of a committed tool result's content.
    Feeds the InterventionEngine's cost/benefit analysis.
    """

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


# =============================================================================
# Pydantic Models — Immutable Records
# =============================================================================


class ToolCallRecord(BaseModel):
    """Immutable record of a single tool invocation."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    tool_name: str
    source_message_index: int
    turn_number: int


class ToolResultRecord(BaseModel):
    """Immutable record of a tool execution result."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    result_content_prefix: str = Field(
        default="",
        description="First 200 chars of result for outcome classification",
    )
    result_length: int
    source_message_index: int
    turn_number: int
    outcome: TransactionOutcome = TransactionOutcome.UNKNOWN


class ToolTransaction(BaseModel):
    """A complete tool call transaction: call + optional result."""

    model_config = ConfigDict(frozen=True)

    call: ToolCallRecord
    result: Optional[ToolResultRecord] = None
    status: TransactionStatus = TransactionStatus.PENDING
    outcome: TransactionOutcome = TransactionOutcome.UNKNOWN
    committed_at_turn: Optional[int] = None


class SemanticFingerprint(BaseModel):
    """Captures the semantic essence of a turn for stagnation comparison."""

    model_config = ConfigDict(frozen=True)

    turn_number: int
    content_hash: str
    tool_signature: str
    structural_pattern: str
    bigram_set: frozenset[tuple[int, int]] = Field(default_factory=frozenset)
    trigram_set: frozenset[tuple[int, int, int]] = Field(default_factory=frozenset)


class InterventionContext(BaseModel):
    """Complete context passed to InterventionEngine.decide()."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    thread_id: str
    node_name: str
    turn_number: int

    active_signals: list[SignalType] = Field(default_factory=list)
    semantic_similarity_score: float = 0.0

    orphaned_transaction_ids: list[str] = Field(default_factory=list)
    dropped_this_turn: list[str] = Field(default_factory=list)
    consecutive_empty_results: int = 0
    consecutive_errors: int = 0

    current_stage: InterventionStage = InterventionStage.PASS
    consecutive_stagnation_count: int = 0
    total_interventions: int = 0
    cooldown_remaining: int = 0
    strategies_attempted: list[str] = Field(default_factory=list)


class InterventionDecision(BaseModel):
    """Output of InterventionEngine.decide()."""

    model_config = ConfigDict(frozen=True)

    stage: InterventionStage
    signals: list[SignalType] = Field(default_factory=list)

    llm_input_messages: Optional[list[dict[str, Any]]] = None
    coaching_message: Optional[str] = None

    state_patch: dict[str, Any] = Field(default_factory=dict)

    should_terminate: bool = False
    termination_reason: Optional[str] = None

    estimated_tokens_saved: int = 0


# =============================================================================
# Canonical Message (lightweight, not Pydantic for performance in hot path)
# =============================================================================


@dataclass(slots=True)
class CanonicalMessage:
    """
    Normalized message representation. Using dataclass with slots for performance
    in the hot path of every pre_model_hook invocation.
    """
    role: CanonicalRole
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    source_index: int = -1
    name: Optional[str] = None

    def __repr__(self) -> str:
        parts = [f"role={self.role.value}"]
        if self.content:
            parts.append(f"content={self.content[:40]!r}")
        if self.tool_calls:
            parts.append(f"tool_calls={len(self.tool_calls)}")
        if self.tool_call_id:
            parts.append(f"tool_call_id={self.tool_call_id!r}")
        return f"CanonicalMessage({', '.join(parts)})"


