"""
TokenCircuit V7.0 — Architecture & Interface Contracts
=======================================================

Principal Architect: [Finalized]
Status: APPROVED — Ready for Implementation
Version: 7.0.0-rc1

OVERVIEW
--------
V7 transforms TokenCircuit from a post-execution astream observer into a
pre-model intervention engine. Instead of detecting loops after the LLM
responds (and then killing the graph with an exception), V7 intercepts
BEFORE the LLM call via LangGraph's `pre_model_hook`, coaching the agent
out of loops using ephemeral `llm_input_messages` that never corrupt
checkpointed state.

KEY ARCHITECTURAL SHIFTS FROM V6
---------------------------------
┌─────────────────────────────────────────────────────────────────────────┐
│  V6 (Post-Execution Observer)         │  V7 (Pre-Model Intervention)    │
│───────────────────────────────────────│──────────────────────────────── │
│  Wraps astream, observes outputs      │  Hooks pre_model_hook, acts     │
│  Only action: raise TokenCircuitError │  Actions: PASS/NUDGE/OVERRIDE/  │
│                                       │           HARD_STOP             │
│  Hash-based detection only            │  Semantic + hash + transaction  │
│  No message modification              │  Ephemeral llm_input_messages   │
│  Destroys graph on detection          │  Coaches, then escalates        │
│  No state injection                   │  _tc_intervention state schema  │
│  Tool calls not validated             │  TranscriptValidator + Ledger   │
└─────────────────────────────────────────────────────────────────────────┘

DATA FLOW DIAGRAM
------------------

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    LangGraph Execution Loop                         │
    │                                                                     │
    │  ┌──────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
    │  │  Graph   │───▶│  pre_model_hook  │───▶│  LLM Call (ChatModel)│  │
    │  │  State   │    │  (V7 INTERCEPT)  │    │                      │  │
    │  └──────────┘    └────────┬─────────┘    └──────────────────────┘  │
    │                           │                                         │
    └───────────────────────────┼─────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────────────────────────────┐
                    │         TokenCircuit V7 Pipeline               │
                    │                                                │
                    │  ┌─────────────────────────────────────────┐  │
                    │  │  1. MessageCanonicalizer                 │  │
                    │  │     Raw messages → Canonical form        │  │
                    │  │     (normalize roles, strip metadata)    │  │
                    │  └──────────────────┬──────────────────────┘  │
                    │                     │                          │
                    │  ┌──────────────────▼──────────────────────┐  │
                    │  │  2. TranscriptValidator                  │  │
                    │  │     Verify tool_call ↔ tool_result       │  │
                    │  │     pairing via ToolTransactionLedger    │  │
                    │  │     Drop orphaned/duplicate results      │  │
                    │  └──────────────────┬──────────────────────┘  │
                    │                     │                          │
                    │  ┌──────────────────▼──────────────────────┐  │
                    │  │  3. SemanticStagnationDetector           │  │
                    │  │     Embedding similarity + hash-based    │  │
                    │  │     sliding window analysis              │  │
                    │  └──────────────────┬──────────────────────┘  │
                    │                     │                          │
                    │  ┌──────────────────▼──────────────────────┐  │
                    │  │  4. InterventionEngine                   │  │
                    │  │     Stage: PASS→NUDGE→OVERRIDE→HARD_STOP│  │
                    │  │     Builds InterventionDecision          │  │
                    │  └──────────────────┬──────────────────────┘  │
                    │                     │                          │
                    │  ┌──────────────────▼──────────────────────┐  │
                    │  │  5. Decision Executor                    │  │
                    │  │     PASS: return messages unchanged      │  │
                    │  │     NUDGE: append coaching system msg    │  │
                    │  │     OVERRIDE: replace with directive     │  │
                    │  │     HARD_STOP: raise / return Command    │  │
                    │  └─────────────────────────────────────────┘  │
                    │                                                │
                    │  Output: llm_input_messages (ephemeral)        │
                    │  (Never written to checkpoint)                 │
                    └────────────────────────────────────────────────┘


MODULE LAYOUT (V7 — PROMOTED TO ROOT)
--------------------------------------

    src/tokencircuit/
    ├── __init__.py             ← Public API (exports all V7 types & adapter)
    ├── engine.py               ← InterventionEngine — central orchestrator
    ├── types.py                ← Enums, Pydantic models, CanonicalMessage
    ├── state_schema.py         ← _tc_intervention TypedDict & reducer
    ├── canonicalizer.py        ← MessageCanonicalizer
    ├── validator.py            ← TranscriptValidator (10+1 invariants)
    ├── ledger.py               ← ToolTransactionLedger
    ├── semantic_detector.py    ← SemanticStagnationDetector (n-gram Jaccard)
    ├── config.py               ← Remote config loading (Supabase)
    ├── telemetry.py            ← OpenTelemetry integration & Prometheus metrics
    ├── exceptions.py           ← TokenCircuitError hierarchy
    ├── adapters/
    │   ├── langgraph.py        ← LangGraphPreModelAdapter (pre_model_hook)
    │   ├── crewai.py           ← CrewAIInterventionAdapter (step_callback)
    │   └── wrapper.py          ← ModelNodeWrapper (fallback for custom graphs)
    └── otel/
        └── hash_utils.py       ← State & action fingerprinting utilities
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# SECTION 1: ENUMERATIONS
# =============================================================================


class InterventionStage(enum.Enum):
    """
    Progressive intervention stages. Each turn, the engine evaluates whether
    to escalate based on stagnation persistence across the cooldown window.

    State Machine:
        PASS ──(stagnation detected)──▶ NUDGE
        NUDGE ──(persists N turns)────▶ OVERRIDE
        OVERRIDE ──(persists M turns)─▶ HARD_STOP
        Any ──(stagnation clears)─────▶ PASS (with cooldown)

    PASS:      No intervention. Messages forwarded unchanged.
    NUDGE:     Append ephemeral coaching message to llm_input_messages.
               LLM sees it but it is never checkpointed.
    OVERRIDE:  Replace message list with a directive forcing strategy change.
               Original messages preserved in state for recovery.
    HARD_STOP: Terminate the agent loop. Return LangGraph Command or raise.
    """

    PASS = "pass"
    NUDGE = "nudge"
    OVERRIDE = "override"
    HARD_STOP = "hard_stop"


class SignalType(enum.Enum):
    """Detection signal taxonomy — extends V6 signals."""

    STATE_STAGNATION = "STATE_STAGNATION"
    FUTILE_ACTION = "FUTILE_ACTION"
    SEMANTIC_STAGNATION = "SEMANTIC_STAGNATION"
    TRANSCRIPT_CORRUPTION = "TRANSCRIPT_CORRUPTION"
    TOOL_TRANSACTION_ORPHAN = "TOOL_TRANSACTION_ORPHAN"


class TransactionStatus(enum.Enum):
    """Lifecycle of a tool call transaction in the ledger."""

    PENDING = "pending"       # tool_call issued, no result yet
    COMMITTED = "committed"   # tool_result received and paired
    ORPHANED = "orphaned"     # tool_call with no matching result (dropped)
    DUPLICATE = "duplicate"   # multiple results for same call_id


class CanonicalRole(enum.Enum):
    """Normalized message roles across providers."""

    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"
    FUNCTION = "function"  # legacy OpenAI function calling


# =============================================================================
# SECTION 2: PYDANTIC MODELS — Core Data Contracts
# =============================================================================


class ToolCallRecord(BaseModel):
    """Immutable record of a single tool invocation."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(description="Unique ID assigned by the LLM (e.g., 'call_abc123')")
    tool_name: str = Field(description="Name of the tool/function invoked")
    arguments_hash: str = Field(description="SHA-256 of canonical JSON arguments")
    arguments_type_signature: str = Field(
        description="PII-safe type signature, e.g., 'search(str,int)'"
    )
    source_message_index: int = Field(
        description="Index in the canonical message list where this call originated"
    )
    turn_number: int = Field(description="Logical turn counter within the thread")


class ToolResultRecord(BaseModel):
    """Immutable record of a tool execution result."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(description="Must match a ToolCallRecord.call_id")
    result_hash: str = Field(description="SHA-256 of normalized result content")
    result_length: int = Field(description="Character length of raw result")
    source_message_index: int = Field(
        description="Index in canonical message list where this result appears"
    )
    turn_number: int = Field(description="Logical turn counter within the thread")


class ToolTransaction(BaseModel):
    """
    A complete tool call transaction — the atomic unit of the ledger.
    A transaction is COMMITTED only when both call and result are present
    and their call_ids match.
    """

    model_config = ConfigDict(frozen=True)

    call: ToolCallRecord
    result: Optional[ToolResultRecord] = None
    status: TransactionStatus = TransactionStatus.PENDING
    committed_at_turn: Optional[int] = Field(
        default=None,
        description="Turn number when transaction was committed (call+result paired)"
    )


class SemanticFingerprint(BaseModel):
    """
    Captures the semantic essence of a turn for stagnation comparison.
    Uses both embedding vectors and structural hashes.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    turn_number: int
    content_hash: str = Field(description="SHA-256 of normalized AI response content")
    tool_signature: str = Field(description="Type-level tool signature or 'NO_TOOL_CALL'")
    intent_embedding: Optional[list[float]] = Field(
        default=None,
        description="Dense embedding of AI intent (optional, requires embedder)"
    )
    structural_pattern: str = Field(
        description=(
            "Encoded structural pattern of the turn: "
            "e.g., 'REASON→TOOL_CALL→OBSERVE' or 'REASON→RESPOND'"
        )
    )


class InterventionContext(BaseModel):
    """
    Complete context passed to the InterventionEngine for decision-making.
    Assembled by the adapter from graph state + pipeline outputs.

    This is the INPUT contract for InterventionEngine.decide().
    """

    model_config = ConfigDict(frozen=False)

    # --- Identity ---
    thread_id: str = Field(description="LangGraph thread/session identifier")
    node_name: str = Field(description="Current node being executed")
    turn_number: int = Field(description="Monotonic turn counter for this thread")

    # --- Message State ---
    canonical_messages: list[dict[str, Any]] = Field(
        description="Canonicalized message list (output of MessageCanonicalizer)"
    )
    raw_message_count: int = Field(description="Count before canonicalization")
    validated_message_count: int = Field(
        description="Count after TranscriptValidator pruning"
    )

    # --- Detection Signals ---
    active_signals: list[SignalType] = Field(
        default_factory=list,
        description="All signals currently firing this turn"
    )
    semantic_fingerprints_window: list[SemanticFingerprint] = Field(
        default_factory=list,
        description="Last N semantic fingerprints for sliding window analysis"
    )
    semantic_similarity_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity of current turn vs window centroid [0.0, 1.0]"
    )

    # --- Transaction State ---
    pending_transactions: list[ToolTransaction] = Field(
        default_factory=list,
        description="Tool calls awaiting results"
    )
    orphaned_transaction_ids: list[str] = Field(
        default_factory=list,
        description="call_ids that were dropped (no matching result found)"
    )
    dropped_this_turn: list[str] = Field(
        default_factory=list,
        description="call_ids dropped by validator THIS turn"
    )

    # --- Intervention History ---
    current_stage: InterventionStage = Field(
        default=InterventionStage.PASS,
        description="Current escalation stage from _tc_intervention state"
    )
    consecutive_interventions: int = Field(
        default=0,
        description="How many consecutive turns have triggered non-PASS decisions"
    )
    last_intervention_turn: Optional[int] = Field(
        default=None,
        description="Turn number of the last non-PASS intervention"
    )
    cooldown_remaining: int = Field(
        default=0,
        description="Turns remaining before re-escalation is allowed"
    )


class InterventionDecision(BaseModel):
    """
    Output of InterventionEngine.decide() — describes what action to take.
    The adapter translates this into llm_input_messages or a Command.

    This is the OUTPUT contract for InterventionEngine.decide().
    """

    model_config = ConfigDict(frozen=True)

    # --- Decision ---
    stage: InterventionStage = Field(description="Decided intervention stage")
    signals: list[SignalType] = Field(
        description="Signals that contributed to this decision"
    )

    # --- Message Modifications ---
    llm_input_messages: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "If set, these messages replace the default input to the LLM. "
            "Ephemeral — never written to checkpointed state. "
            "For PASS: None (use canonical messages as-is). "
            "For NUDGE: canonical + appended coaching message. "
            "For OVERRIDE: replacement directive message list. "
            "For HARD_STOP: None (graph will be terminated)."
        )
    )
    coaching_message: Optional[str] = Field(
        default=None,
        description="Human-readable coaching text injected for NUDGE/OVERRIDE"
    )

    # --- State Mutations ---
    state_patch: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Patch to apply to _tc_intervention state channel. "
            "Adapter merges this into graph state update."
        )
    )

    # --- Termination ---
    should_terminate: bool = Field(
        default=False,
        description="If True, adapter should halt execution (HARD_STOP)"
    )
    termination_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason for HARD_STOP"
    )

    # --- Telemetry ---
    estimated_tokens_saved: int = Field(
        default=0,
        description="Estimated tokens that will be saved by this intervention"
    )
    estimated_cost_saved_usd: float = Field(
        default=0.0,
        description="Estimated USD cost saved"
    )


class InterventionConfig(BaseModel):
    """
    V7 configuration model — extends TokenCircuitConfig with intervention params.
    Loaded from remote config or local overrides.
    """

    model_config = ConfigDict(frozen=False)

    # --- V6 Compatibility ---
    max_repeats: int = Field(default=5, ge=1, description="Legacy: max iterations before V6 kill")
    window_size: int = Field(default=5, ge=2, description="Sliding window size for detection")
    agency_id: Optional[str] = Field(default=None, description="Telemetry: agency identifier")
    client_id: Optional[str] = Field(default=None, description="Telemetry: client identifier")
    model_name: str = Field(default="unknown", description="LLM model name for cost estimation")
    telemetry_enabled: bool = Field(default=True, description="Toggle telemetry emission")

    # --- V7 Intervention Thresholds ---
    nudge_threshold: int = Field(
        default=3, ge=1,
        description="Consecutive stagnant turns before NUDGE stage"
    )
    override_threshold: int = Field(
        default=5, ge=2,
        description="Consecutive stagnant turns before OVERRIDE stage"
    )
    hard_stop_threshold: int = Field(
        default=8, ge=3,
        description="Consecutive stagnant turns before HARD_STOP"
    )
    cooldown_turns: int = Field(
        default=2, ge=0,
        description="Turns to wait after de-escalation before allowing re-escalation"
    )

    # --- Semantic Detection ---
    semantic_similarity_threshold: float = Field(
        default=0.92, ge=0.0, le=1.0,
        description="Cosine similarity threshold to flag semantic stagnation"
    )
    enable_semantic_detection: bool = Field(
        default=True,
        description="Toggle embedding-based semantic stagnation detection"
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Model ID for embeddings. None = use structural hashing only."
    )

    # --- Transaction Validation ---
    enable_transcript_validation: bool = Field(
        default=True,
        description="Toggle TranscriptValidator (tool call pairing enforcement)"
    )
    max_orphan_tolerance: int = Field(
        default=2, ge=0,
        description="Max orphaned transactions before TRANSCRIPT_CORRUPTION signal"
    )
    drop_orphaned_results: bool = Field(
        default=True,
        description="Auto-drop tool_results with no matching tool_call"
    )

    # --- Coaching Templates ---
    nudge_template: str = Field(
        default=(
            "You appear to be repeating the same approach. "
            "Consider: {suggestion}. "
            "Previous attempts: {attempt_summary}."
        ),
        description="Template for NUDGE coaching message"
    )
    override_template: str = Field(
        default=(
            "SYSTEM OVERRIDE: Your previous {n_attempts} attempts used the same strategy "
            "and produced identical results. You MUST try a different approach. "
            "Specifically: {directive}."
        ),
        description="Template for OVERRIDE directive message"
    )


# =============================================================================
# SECTION 3: TypedDict — LangGraph State Schema (_tc_intervention)
# =============================================================================


class ToolTransactionEntry(TypedDict):
    """Single entry in the serialized transaction ledger within graph state."""

    call_id: str
    tool_name: str
    status: str  # TransactionStatus.value
    turn_issued: int
    turn_committed: Optional[int]


class InterventionStateSchema(TypedDict, total=False):
    """
    The `_tc_intervention` state channel injected into LangGraph graph state.

    This is a REDUCER-MANAGED channel. The adapter provides a custom reducer
    that merges state_patch from InterventionDecision into this schema.

    All fields are optional (total=False) to support incremental updates.

    Usage in graph definition:
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            _tc_intervention: Annotated[InterventionStateSchema, tc_reducer]
    """

    # --- Stage Tracking ---
    current_stage: str  # InterventionStage.value
    stage_entered_at_turn: int  # Turn when current stage was entered
    previous_stage: str  # InterventionStage.value

    # --- Counters ---
    turn_counter: int  # Monotonic turn counter
    consecutive_stagnation_count: int  # Consecutive turns with any stagnation signal
    total_interventions: int  # Lifetime count of non-PASS decisions
    nudge_count: int  # Total NUDGEs issued
    override_count: int  # Total OVERRIDEs issued

    # --- Cooldown ---
    cooldown_remaining: int  # Turns remaining in cooldown after de-escalation
    last_escalation_turn: int  # Turn number of last stage increase
    last_deescalation_turn: int  # Turn number of last stage decrease

    # --- Semantic Window ---
    fingerprint_hashes: list[str]  # Last N content_hashes for quick comparison
    last_similarity_score: float  # Most recent semantic similarity score

    # --- Transaction Ledger (serialized) ---
    pending_transaction_ids: list[str]  # call_ids awaiting results
    orphaned_transaction_ids: list[str]  # call_ids permanently dropped
    committed_transaction_count: int  # Total committed transactions
    dropped_this_session: list[str]  # All call_ids dropped in this session

    # --- Coaching History ---
    last_coaching_message: str  # Most recent coaching text sent
    coaching_history: list[str]  # All coaching messages (for diversity)
    strategies_attempted: list[str]  # Logged strategy changes for override


# =============================================================================
# SECTION 4: PROTOCOL DEFINITIONS — Extension Points
# =============================================================================


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers used by SemanticStagnationDetector.
    Implementations can use OpenAI, local models, or structural hashing.
    """

    async def embed(self, text: str) -> list[float]:
        """Return dense embedding vector for the given text."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        ...


@runtime_checkable
class CoachingStrategy(Protocol):
    """
    Protocol for coaching message generation strategies.
    Allows pluggable coaching behavior (template-based, LLM-generated, etc).
    """

    def generate_nudge(
        self,
        context: InterventionContext,
        config: InterventionConfig,
    ) -> str:
        """Generate a coaching message for NUDGE stage."""
        ...

    def generate_override_directive(
        self,
        context: InterventionContext,
        config: InterventionConfig,
    ) -> str:
        """Generate a forceful directive for OVERRIDE stage."""
        ...


@runtime_checkable
class StateReducer(Protocol):
    """
    Protocol for the _tc_intervention state channel reducer.
    Merges incoming patches with existing state.
    """

    def __call__(
        self,
        existing: InterventionStateSchema,
        update: InterventionStateSchema,
    ) -> InterventionStateSchema:
        """Merge update into existing state. Must be idempotent."""
        ...


# =============================================================================
# SECTION 5: CLASS INTERFACES — MessageCanonicalizer
# =============================================================================


class CanonicalMessage(BaseModel):
    """A normalized message representation agnostic to LangChain message types."""

    model_config = ConfigDict(frozen=True)

    role: CanonicalRole
    content: str = Field(description="Text content (empty string if tool-only)")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Normalized tool calls [{call_id, name, arguments}]"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="For TOOL role: the call_id this result responds to"
    )
    source_index: int = Field(description="Original index in raw message list")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved non-content metadata (model, finish_reason, etc)"
    )


class MessageCanonicalizer:
    """
    Normalizes heterogeneous message formats into a canonical representation.

    Responsibilities:
    - Convert LangChain BaseMessage subclasses to CanonicalMessage
    - Convert OpenAI-format dicts to CanonicalMessage
    - Normalize roles (assistant→ai, user→human, function→tool)
    - Strip internal metadata prefixed with '_tc_'
    - Preserve message ordering and source indices
    - Deduplicate consecutive identical messages (idempotent retries)

    Thread Safety: Stateless — safe for concurrent use.
    """

    def __init__(self, *, strip_prefixes: tuple[str, ...] = ("_tc_", "_meta_")) -> None:
        """
        Args:
            strip_prefixes: Metadata key prefixes to remove during canonicalization.
        """
        ...

    def canonicalize(
        self,
        messages: Sequence[Any],
    ) -> list[CanonicalMessage]:
        """
        Convert a sequence of messages (LangChain objects or dicts) to canonical form.

        Args:
            messages: Raw message list from graph state["messages"].

        Returns:
            Ordered list of CanonicalMessage instances.

        Raises:
            Never raises. Malformed messages are logged and included as-is with
            role=AI and content=repr(message).
        """
        ...

    def to_llm_format(
        self,
        canonical: list[CanonicalMessage],
        *,
        provider: Literal["openai", "anthropic", "langchain"] = "langchain",
    ) -> list[dict[str, Any]]:
        """
        Convert canonical messages back to provider-specific format for LLM input.

        Args:
            canonical: List of CanonicalMessage instances.
            provider: Target format (openai dict, anthropic dict, or langchain objects).

        Returns:
            Message list suitable for the target LLM provider.
        """
        ...

    def deduplicate(
        self,
        messages: list[CanonicalMessage],
    ) -> list[CanonicalMessage]:
        """
        Remove consecutive duplicate messages (same role + content + tool_calls hash).
        Common pattern: idempotent retries that re-append the same AI message.

        Returns:
            Deduplicated list preserving first occurrence.
        """
        ...


# =============================================================================
# SECTION 6: CLASS INTERFACES — ToolTransactionLedger
# =============================================================================


class ToolTransactionLedger:
    """
    Tracks the lifecycle of tool call → tool result transactions.

    Invariants:
    - Every tool_call generates a PENDING transaction.
    - A tool_result COMMITS the transaction (pairing by call_id).
    - A tool_result with no matching PENDING call_id is flagged as ORPHANED.
    - A second tool_result for an already COMMITTED call_id is DUPLICATE.
    - PENDING transactions exceeding a turn threshold are marked ORPHANED.

    The ledger is the source of truth for TranscriptValidator decisions about
    which messages to keep, drop, or reorder.

    Thread Safety: NOT thread-safe. One ledger per agent thread.
    """

    def __init__(
        self,
        *,
        orphan_timeout_turns: int = 3,
        max_pending: int = 10,
    ) -> None:
        """
        Args:
            orphan_timeout_turns: Turns after which a PENDING transaction is orphaned.
            max_pending: Maximum concurrent pending transactions before warning.
        """
        ...

    def register_call(
        self,
        call_id: str,
        tool_name: str,
        arguments_hash: str,
        arguments_type_signature: str,
        source_message_index: int,
        turn_number: int,
    ) -> ToolTransaction:
        """
        Register a new tool call. Creates a PENDING transaction.

        Args:
            call_id: Unique call identifier from the LLM response.
            tool_name: Name of the tool being called.
            arguments_hash: SHA-256 of serialized arguments.
            arguments_type_signature: PII-safe type signature.
            source_message_index: Index of the AI message containing this call.
            turn_number: Current turn counter.

        Returns:
            The newly created ToolTransaction in PENDING status.

        Raises:
            ValueError: If call_id is already registered (duplicate call emission).
        """
        ...

    def register_result(
        self,
        call_id: str,
        result_hash: str,
        result_length: int,
        source_message_index: int,
        turn_number: int,
    ) -> tuple[ToolTransaction, TransactionStatus]:
        """
        Register a tool result. Attempts to COMMIT the matching transaction.

        Args:
            call_id: Must match a previously registered ToolCallRecord.call_id.
            result_hash: SHA-256 of normalized result content.
            result_length: Character length of raw result.
            source_message_index: Index of the ToolMessage in canonical list.
            turn_number: Current turn counter.

        Returns:
            Tuple of (updated ToolTransaction, resulting TransactionStatus).
            Status will be COMMITTED, ORPHANED (no matching call), or DUPLICATE.
        """
        ...

    def advance_turn(self, turn_number: int) -> list[ToolTransaction]:
        """
        Called at the start of each turn. Ages out stale PENDING transactions.

        Args:
            turn_number: Current turn counter.

        Returns:
            List of transactions that were transitioned to ORPHANED this turn.
        """
        ...

    def get_pending(self) -> list[ToolTransaction]:
        """Return all currently PENDING transactions."""
        ...

    def get_orphaned(self) -> list[ToolTransaction]:
        """Return all ORPHANED transactions."""
        ...

    def get_committed_since(self, turn: int) -> list[ToolTransaction]:
        """Return transactions committed at or after the given turn."""
        ...

    def get_transaction(self, call_id: str) -> Optional[ToolTransaction]:
        """Look up a transaction by call_id. Returns None if not found."""
        ...

    @property
    def total_committed(self) -> int:
        """Total number of successfully committed transactions."""
        ...

    @property
    def total_orphaned(self) -> int:
        """Total number of orphaned transactions."""
        ...

    def serialize(self) -> list[ToolTransactionEntry]:
        """
        Serialize ledger state for inclusion in _tc_intervention graph state.
        Only includes PENDING and recently ORPHANED (last 10 turns).
        """
        ...

    @classmethod
    def deserialize(
        cls,
        entries: list[ToolTransactionEntry],
        *,
        orphan_timeout_turns: int = 3,
    ) -> "ToolTransactionLedger":
        """Reconstruct ledger from serialized graph state."""
        ...


# =============================================================================
# SECTION 7: CLASS INTERFACES — TranscriptValidator
# =============================================================================


class ValidationResult(BaseModel):
    """Output of TranscriptValidator.validate()."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(description="Whether the transcript passes validation")
    validated_messages: list[CanonicalMessage] = Field(
        description="Messages after validation (may have entries removed)"
    )
    dropped_indices: list[int] = Field(
        default_factory=list,
        description="Source indices of messages that were dropped"
    )
    dropped_call_ids: list[str] = Field(
        default_factory=list,
        description="call_ids of dropped orphaned tool results"
    )
    signals: list[SignalType] = Field(
        default_factory=list,
        description="Validation-level signals (TRANSCRIPT_CORRUPTION, TOOL_TRANSACTION_ORPHAN)"
    )
    repair_actions: list[str] = Field(
        default_factory=list,
        description="Human-readable log of repairs performed"
    )


class TranscriptValidator:
    """
    Validates and repairs the message transcript before it reaches the LLM.

    Core Principle: Tool calls are IMMUTABLE TRANSACTIONS.
    - An AI message with tool_calls is a COMMITMENT to execute those tools.
    - A ToolMessage is the RESULT of that commitment.
    - These pairs must be matched, ordered, and complete.
    - Violating this causes API validation errors (OpenAI, Anthropic).

    Validation Rules:
    1. Every ToolMessage must reference a tool_call that exists in a prior AI message.
    2. Every tool_call in an AI message should have a corresponding ToolMessage.
    3. ToolMessages must appear AFTER the AI message containing their tool_call.
    4. No duplicate ToolMessages for the same call_id.
    5. If validation fails and repair is enabled, drop orphaned messages.

    Thread Safety: Stateless given a ledger. Safe for concurrent use with
                   separate ledger instances.
    """

    def __init__(
        self,
        *,
        ledger: ToolTransactionLedger,
        auto_repair: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """
        Args:
            ledger: The ToolTransactionLedger to use for tracking.
            auto_repair: If True, automatically drop invalid messages.
            strict_mode: If True, emit TRANSCRIPT_CORRUPTION on any violation.
        """
        ...

    def validate(
        self,
        messages: list[CanonicalMessage],
        turn_number: int,
    ) -> ValidationResult:
        """
        Validate the canonical message transcript.

        Processes messages sequentially:
        1. Register all tool_calls found in AI messages with the ledger.
        2. Attempt to commit all ToolMessages against the ledger.
        3. Identify orphans (results with no matching call).
        4. If auto_repair: drop orphaned ToolMessages.
        5. If orphan count exceeds tolerance: emit TRANSCRIPT_CORRUPTION signal.

        Args:
            messages: Canonicalized message list (from MessageCanonicalizer).
            turn_number: Current turn counter for the ledger.

        Returns:
            ValidationResult with validated messages and any signals.
        """
        ...

    def validate_single_message(
        self,
        message: CanonicalMessage,
        turn_number: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a single message in isolation against ledger state.

        Args:
            message: Single canonical message to validate.
            turn_number: Current turn.

        Returns:
            Tuple of (is_valid, reason_if_invalid).
        """
        ...

    def get_repair_summary(self) -> dict[str, int]:
        """
        Return cumulative repair statistics.

        Returns:
            Dict with keys: 'total_validated', 'total_dropped', 'total_repaired',
            'orphans_removed', 'duplicates_removed'.
        """
        ...


# =============================================================================
# SECTION 8: CLASS INTERFACES — SemanticStagnationDetector
# =============================================================================


class StagnationAnalysis(BaseModel):
    """Output of SemanticStagnationDetector.analyze()."""

    model_config = ConfigDict(frozen=True)

    is_stagnating: bool = Field(description="Whether semantic stagnation is detected")
    similarity_score: float = Field(
        description="Similarity of current turn to window centroid [0.0, 1.0]"
    )
    pattern_diversity: float = Field(
        description="Diversity of structural patterns in window [0.0, 1.0]"
    )
    signals: list[SignalType] = Field(
        default_factory=list,
        description="Emitted signals (SEMANTIC_STAGNATION, STATE_STAGNATION, FUTILE_ACTION)"
    )
    fingerprint: SemanticFingerprint = Field(
        description="Fingerprint computed for the current turn"
    )
    window_summary: str = Field(
        default="",
        description="Human-readable summary of the stagnation pattern"
    )


class SemanticStagnationDetector:
    """
    Detects semantic-level stagnation beyond simple hash equality.

    Detection Modes:
    1. STRUCTURAL: Compares structural_pattern strings across window.
       No external dependencies. Always active.
    2. HASH-BASED: V6 compatibility — exact content_hash matching.
       Always active.
    3. EMBEDDING: Cosine similarity of intent embeddings.
       Requires EmbeddingProvider. Optional.

    A turn is considered stagnating if ANY of:
    - content_hash matches >threshold of window entries (V6 mode)
    - structural_pattern repeats >threshold of window entries
    - cosine similarity to window centroid > semantic_similarity_threshold

    Subsumes V6's StateStagnationDetector and FutileActionDetector,
    providing a unified interface with richer signals.

    Thread Safety: NOT thread-safe. One detector per agent thread.
    """

    def __init__(
        self,
        *,
        config: InterventionConfig,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        """
        Args:
            config: V7 intervention configuration.
            embedding_provider: Optional provider for dense embeddings.
                                If None, only structural + hash detection is used.
        """
        ...

    async def analyze(
        self,
        messages: list[CanonicalMessage],
        turn_number: int,
    ) -> StagnationAnalysis:
        """
        Analyze the current turn for semantic stagnation.

        Steps:
        1. Compute SemanticFingerprint for the current turn.
        2. Compare against the sliding window of previous fingerprints.
        3. Calculate similarity score (embedding or structural).
        4. Determine pattern diversity.
        5. Emit appropriate signals based on thresholds.

        Args:
            messages: Current canonical message list.
            turn_number: Current turn counter.

        Returns:
            StagnationAnalysis with all computed metrics and signals.
        """
        ...

    def record_fingerprint(self, fingerprint: SemanticFingerprint) -> None:
        """
        Add a fingerprint to the sliding window. Called after analyze().

        Args:
            fingerprint: The fingerprint from the current turn's StagnationAnalysis.
        """
        ...

    def get_window(self) -> list[SemanticFingerprint]:
        """Return the current sliding window of fingerprints."""
        ...

    def reset(self) -> None:
        """Clear the sliding window. Used on de-escalation or graph restart."""
        ...

    @property
    def window_size(self) -> int:
        """Current number of fingerprints in the window."""
        ...

    def compute_fingerprint(
        self,
        messages: list[CanonicalMessage],
        turn_number: int,
        embedding: Optional[list[float]] = None,
    ) -> SemanticFingerprint:
        """
        Compute a SemanticFingerprint for the given messages without analyzing.

        Args:
            messages: Canonical messages for this turn.
            turn_number: Turn counter.
            embedding: Pre-computed embedding vector (optional).

        Returns:
            SemanticFingerprint instance.
        """
        ...


# =============================================================================
# SECTION 9: CLASS INTERFACES — InterventionEngine (Orchestrator)
# =============================================================================


class InterventionEngine:
    """
    Central orchestrator for the V7 pre-model intervention pipeline.

    Responsibilities:
    - Coordinate MessageCanonicalizer, TranscriptValidator, and SemanticStagnationDetector
    - Apply the stage progression state machine (PASS→NUDGE→OVERRIDE→HARD_STOP)
    - Produce InterventionDecision with ephemeral llm_input_messages
    - Manage cooldowns and de-escalation
    - Generate coaching messages via CoachingStrategy

    Lifecycle:
    - One InterventionEngine per graph instance (shared across nodes).
    - Internal state is per-thread (keyed by thread_id + node_name).
    - Stateless between graph invocations — all persistent state lives in
      _tc_intervention graph state channel.

    Thread Safety: Thread-safe via per-key internal state isolation.
    """

    def __init__(
        self,
        *,
        config: InterventionConfig,
        coaching_strategy: Optional[CoachingStrategy] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        """
        Args:
            config: V7 intervention configuration.
            coaching_strategy: Strategy for generating coaching messages.
                               Defaults to TemplateCoachingStrategy.
            embedding_provider: Optional embedding provider for semantic detection.
        """
        ...

    async def process(
        self,
        messages: Sequence[Any],
        state: dict[str, Any],
        *,
        thread_id: str,
        node_name: str,
    ) -> InterventionDecision:
        """
        Main entry point. Runs the full V7 pipeline and returns a decision.

        Pipeline Steps:
        1. Extract _tc_intervention from state (or initialize defaults).
        2. Increment turn counter.
        3. Canonicalize messages via MessageCanonicalizer.
        4. Validate transcript via TranscriptValidator.
        5. Analyze for stagnation via SemanticStagnationDetector.
        6. Build InterventionContext from all outputs.
        7. Call decide() to produce InterventionDecision.
        8. Package state_patch for _tc_intervention update.

        Args:
            messages: Raw message list from graph state["messages"].
            state: Full graph state dict (contains _tc_intervention if present).
            thread_id: LangGraph thread identifier.
            node_name: Current graph node name.

        Returns:
            InterventionDecision describing the action to take.
        """
        ...

    def decide(self, context: InterventionContext) -> InterventionDecision:
        """
        Pure decision function. Given full context, produce intervention decision.

        State Machine Logic:
        - If cooldown_remaining > 0: return PASS (cooling down).
        - If no active signals: de-escalate toward PASS.
        - If signals present:
          - consecutive < nudge_threshold: PASS
          - nudge_threshold <= consecutive < override_threshold: NUDGE
          - override_threshold <= consecutive < hard_stop_threshold: OVERRIDE
          - consecutive >= hard_stop_threshold: HARD_STOP

        Args:
            context: Complete InterventionContext.

        Returns:
            InterventionDecision with stage, messages, and state patch.
        """
        ...

    def get_engine_state(self, thread_id: str, node_name: str) -> dict[str, Any]:
        """
        Return internal engine state for a given thread+node (for debugging).

        Args:
            thread_id: Thread identifier.
            node_name: Node name.

        Returns:
            Dict with internal counters, window state, ledger summary.
        """
        ...

    def reset(self, thread_id: str, node_name: str) -> None:
        """
        Reset all internal state for a thread+node. Used for testing
        or manual intervention recovery.
        """
        ...

    def reset_all(self) -> None:
        """Reset all internal state across all threads. Nuclear option."""
        ...


# =============================================================================
# SECTION 10: CLASS INTERFACES — LangGraphPreModelAdapter
# =============================================================================


class LangGraphPreModelAdapter:
    """
    Adapter that connects InterventionEngine to LangGraph's pre_model_hook.

    This is the PRIMARY integration path for V7. It hooks into the
    pre_model_hook callback that LangGraph invokes before each LLM call
    within a model-calling node (e.g., call_model, agent).

    Contract with LangGraph:
    - pre_model_hook receives: (state: GraphState) -> dict with 'llm_input_messages'
    - If llm_input_messages is returned, LangGraph uses THOSE messages for the LLM call
      instead of state["messages"]. These are EPHEMERAL — never checkpointed.
    - If None/empty is returned, LangGraph uses state["messages"] as normal.
    - Hook can also return a Command to terminate the graph.

    Integration Pattern:
        from tokencircuit.v7.adapters.langgraph import LangGraphPreModelAdapter

        adapter = LangGraphPreModelAdapter(config=my_config)

        graph = StateGraph(AgentState)
        graph.add_node("agent", adapter.wrap_model_node(my_model_node))
        # OR for pre_model_hook support:
        graph.add_node("agent", call_model, pre_model_hook=adapter.hook)

    Thread Safety: Thread-safe. Uses InterventionEngine's per-key isolation.
    """

    def __init__(
        self,
        *,
        config: Optional[InterventionConfig] = None,
        engine: Optional[InterventionEngine] = None,
        coaching_strategy: Optional[CoachingStrategy] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: V7 configuration. If None, loads from remote/defaults.
            engine: Pre-built InterventionEngine. If None, creates one from config.
            coaching_strategy: Custom coaching strategy (passed to engine).
            embedding_provider: Custom embedding provider (passed to engine).
            api_key: API key for remote config loading.
        """
        ...

    async def hook(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        The pre_model_hook callback. This is what gets passed to LangGraph.

        Contract:
        - Input: Full graph state dict.
        - Output: Dict that MAY contain 'llm_input_messages' key.
          - If PASS: return {} (empty dict, no override)
          - If NUDGE: return {"llm_input_messages": [...original + coaching...]}
          - If OVERRIDE: return {"llm_input_messages": [...directive...]}
          - If HARD_STOP: return Command(goto=END) or raise TokenCircuitError

        State Extraction:
        - Messages from: state["messages"]
        - Thread ID from: state.get("_tc_intervention", {}).get("thread_id")
                          OR config["configurable"]["thread_id"]
        - Node name from: hook invocation context (injected by adapter registration)

        Side Effects:
        - Updates _tc_intervention state channel via returned state patch.
        - Emits telemetry for non-PASS decisions.

        Args:
            state: Full graph state as provided by LangGraph to pre_model_hook.

        Returns:
            Dict with optional 'llm_input_messages' and state updates.
        """
        ...

    def create_hook(
        self,
        *,
        node_name: str,
    ) -> Callable[[dict[str, Any]], Any]:
        """
        Factory that creates a pre_model_hook bound to a specific node name.

        Use this when you need different hook instances per node:
            graph.add_node("agent", call_model,
                           pre_model_hook=adapter.create_hook(node_name="agent"))
            graph.add_node("reviewer", call_model,
                           pre_model_hook=adapter.create_hook(node_name="reviewer"))

        Args:
            node_name: The node name to bind to this hook instance.

        Returns:
            An async callable matching the pre_model_hook signature.
        """
        ...

    def get_state_schema_annotation(self) -> tuple[type, Callable]:
        """
        Returns the TypedDict and reducer for the _tc_intervention state channel.

        Usage:
            from typing import Annotated
            TCSchema, tc_reducer = adapter.get_state_schema_annotation()

            class AgentState(TypedDict):
                messages: Annotated[list, add_messages]
                _tc_intervention: Annotated[TCSchema, tc_reducer]

        Returns:
            Tuple of (InterventionStateSchema, tc_state_reducer function).
        """
        ...

    @staticmethod
    def tc_state_reducer(
        existing: InterventionStateSchema,
        update: InterventionStateSchema,
    ) -> InterventionStateSchema:
        """
        Reducer for the _tc_intervention state channel.

        Merge semantics:
        - Scalar fields: update overwrites existing.
        - List fields (orphaned_transaction_ids, coaching_history):
          APPEND (deduplicated).
        - Counter fields (turn_counter, total_interventions): MAX of existing/update.

        Args:
            existing: Current state value.
            update: Incoming patch from InterventionDecision.state_patch.

        Returns:
            Merged InterventionStateSchema.
        """
        ...

    @property
    def engine(self) -> InterventionEngine:
        """Access the underlying InterventionEngine for inspection/testing."""
        ...


# =============================================================================
# SECTION 11: CLASS INTERFACES — ModelNodeWrapper (Fallback Adapter)
# =============================================================================


class ModelNodeWrapper:
    """
    Fallback adapter for graphs that don't support pre_model_hook.

    Wraps the entire model-calling node function, intercepting the messages
    before they reach the LLM and applying the same intervention logic.

    Use Case:
    - Custom graph nodes that call LLMs directly (not using LangGraph's
      built-in call_model).
    - Older LangGraph versions without pre_model_hook support.
    - Non-LangGraph frameworks that follow similar patterns.

    Integration Pattern:
        from tokencircuit.v7.adapters.wrapper import ModelNodeWrapper

        wrapper = ModelNodeWrapper(config=my_config)

        # Wrap your model node function:
        @wrapper.wrap
        async def call_model(state: AgentState) -> dict:
            messages = state["messages"]  # <-- wrapper intercepts HERE
            response = await llm.ainvoke(messages)
            return {"messages": [response]}

        # Or use as a decorator factory:
        @wrapper.wrap_with(node_name="agent")
        async def call_model(state: AgentState) -> dict:
            ...

    Thread Safety: Thread-safe. Delegates to InterventionEngine.
    """

    def __init__(
        self,
        *,
        config: Optional[InterventionConfig] = None,
        engine: Optional[InterventionEngine] = None,
        coaching_strategy: Optional[CoachingStrategy] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: V7 configuration.
            engine: Pre-built engine (shared with LangGraphPreModelAdapter if desired).
            coaching_strategy: Custom coaching strategy.
            embedding_provider: Custom embedding provider.
            api_key: API key for remote config.
        """
        ...

    def wrap(
        self,
        func: Callable[..., Any],
        *,
        node_name: Optional[str] = None,
    ) -> Callable[..., Any]:
        """
        Wrap a model node function with intervention logic.

        The wrapper:
        1. Extracts messages from the state argument.
        2. Runs InterventionEngine.process().
        3. If PASS: calls original func unchanged.
        4. If NUDGE/OVERRIDE: modifies messages in state before calling func.
        5. If HARD_STOP: raises TokenCircuitError without calling func.

        Args:
            func: The original model node function (sync or async).
            node_name: Override node name (defaults to func.__name__).

        Returns:
            Wrapped function with same signature as original.
        """
        ...

    def wrap_with(
        self,
        *,
        node_name: str,
        message_key: str = "messages",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator factory for wrapping with explicit configuration.

        Args:
            node_name: Name of the node (for state tracking).
            message_key: Key in state dict where messages are stored.

        Returns:
            Decorator that wraps the target function.

        Usage:
            @wrapper.wrap_with(node_name="agent", message_key="messages")
            async def call_model(state):
                ...
        """
        ...

    @property
    def engine(self) -> InterventionEngine:
        """Access the underlying InterventionEngine."""
        ...


# =============================================================================
# SECTION 12: PUBLIC API — Module-level convenience functions
# =============================================================================


def create_intervention_engine(
    *,
    config: Optional[InterventionConfig] = None,
    api_key: Optional[str] = None,
    coaching_strategy: Optional[CoachingStrategy] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> InterventionEngine:
    """
    Factory function to create a configured InterventionEngine.

    Args:
        config: V7 configuration. If None, loads from remote/defaults.
        api_key: API key for remote config loading.
        coaching_strategy: Custom coaching strategy.
        embedding_provider: Custom embedding provider.

    Returns:
        Configured InterventionEngine instance.
    """
    ...


# NOTE: create_pre_model_hook was dropped during implementation.
# The actual API uses LangGraphPreModelAdapter directly:
#
#   adapter = LangGraphPreModelAdapter(config=config)
#   graph.add_node("agent", call_model, pre_model_hook=adapter.hook)
#
# This keeps the adapter instance accessible for inspection and
# state management while providing the same one-liner ergonomics.


# NOTE: instrument_langgraph_v7 was dropped during implementation.
# The V7 equivalent is using LangGraphPreModelAdapter directly.
# `instrument_langgraph()` in __init__.py is preserved as a deprecated
# backward-compat wrapper for users migrating from V6.


# =============================================================================
# SECTION 13: STATE INITIALIZATION HELPERS
# =============================================================================


def default_intervention_state() -> InterventionStateSchema:
    """
    Return the default initial value for the _tc_intervention state channel.

    Used when a graph starts fresh with no prior intervention history.

    Returns:
        InterventionStateSchema with all fields at their zero/default values.
    """
    ...


def merge_intervention_state(
    existing: InterventionStateSchema,
    patch: dict[str, Any],
) -> InterventionStateSchema:
    """
    Apply a partial state patch to an existing intervention state.

    Follows the same merge semantics as tc_state_reducer:
    - Scalars: overwrite
    - Lists: append (deduplicated)
    - Counters: max(existing, update)

    Args:
        existing: Current _tc_intervention state.
        patch: Partial update from InterventionDecision.state_patch.

    Returns:
        New InterventionStateSchema with patch applied.
    """
    ...


# =============================================================================
# SECTION 14: BACKWARD COMPATIBILITY — V6 Bridge
# =============================================================================


# NOTE: V6CompatibilityBridge was a design proposal that was not implemented.
# The V6 `instrument_langgraph()` function is still available in
# tokencircuit.__init__.py as a deprecated wrapper. It uses ModelNodeWrapper
# internally with the V7 InterventionEngine.
#
# For new projects, use LangGraphPreModelAdapter directly.


# =============================================================================
# SECTION 15: COMPLETE INTEGRATION EXAMPLE (Type-Checked, No Implementation)
# =============================================================================

"""
INTEGRATION EXAMPLE — Full V7 Setup with LangGraph
====================================================

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from tokencircuit import (
    InterventionConfig,
    InterventionStateSchema,
    LangGraphPreModelAdapter,
    default_intervention_state,
)


# 1. Define state WITH _tc_intervention channel
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    _tc_intervention: Annotated[
        InterventionStateSchema,
        LangGraphPreModelAdapter.tc_state_reducer
    ]


# 2. Configure V7
config = InterventionConfig(
    nudge_threshold=3,
    override_threshold=5,
    hard_stop_threshold=8,
    semantic_similarity_threshold=0.92,
    window_size=5,
)


# 3. Create adapter
adapter = instrument_langgraph_v7(graph=None, config=config)


# 4. Define model node (standard LangGraph pattern)
llm = ChatOpenAI(model="gpt-4o")

async def call_model(state: AgentState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


# 5. Build graph with pre_model_hook
graph_builder = StateGraph(AgentState)
graph_builder.add_node(
    "agent",
    call_model,
    pre_model_hook=adapter.create_hook(node_name="agent"),
)
# ... add edges, tools node, etc.
graph = graph_builder.compile(checkpointer=my_checkpointer)


# 6. State initializes with default intervention state
initial_state = {
    "messages": [],
    "_tc_intervention": default_intervention_state(),
}


# ALTERNATIVE: ModelNodeWrapper for custom graphs without pre_model_hook
from tokencircuit.v7.adapters.wrapper import ModelNodeWrapper

wrapper = ModelNodeWrapper(config=config)

@wrapper.wrap_with(node_name="agent")
async def custom_model_call(state: AgentState) -> dict:
    # Wrapper intercepts state["messages"] BEFORE this body executes
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}
"""


# =============================================================================
# SECTION 16: INVARIANTS AND ARCHITECTURAL CONSTRAINTS
# =============================================================================

"""
ARCHITECTURAL INVARIANTS
=========================

1. EPHEMERAL MESSAGE GUARANTEE
   - llm_input_messages returned by pre_model_hook are NEVER written to
     the checkpointed graph state.
   - The checkpoint always contains the TRUE conversation history.
   - Coaching messages exist only for the duration of one LLM call.

2. TOOL TRANSACTION IMMUTABILITY
   - Once a tool_call is emitted by the LLM, it is a COMMITTED transaction.
   - TokenCircuit will NEVER modify, reorder, or remove committed tool_calls.
   - TokenCircuit MAY drop orphaned tool_results (no matching call).
   - TokenCircuit MAY drop duplicate tool_results (already committed).

3. STATE ISOLATION
   - _tc_intervention is the ONLY state channel TokenCircuit reads/writes.
   - TokenCircuit NEVER mutates state["messages"] directly.
   - All message modifications are ephemeral (via llm_input_messages).

4. MONOTONIC ESCALATION
   - Stage can only increase by one level per turn (no skipping stages).
   - De-escalation is immediate (any clear turn → PASS) but subject to cooldown.
   - Cooldown prevents re-escalation thrashing.

5. FAIL-SAFE DEFAULT
   - If ANY component raises an unexpected exception, the pipeline returns PASS.
   - TokenCircuit must NEVER prevent a working agent from making progress.
   - All errors are logged but never propagated (except HARD_STOP).

6. BACKWARD COMPATIBILITY
   - V6 instrument_langgraph() continues to work (deprecated wrapper).
   - V7 modules live at src/tokencircuit/ root (import via tokencircuit.X).
   - TokenCircuitConfig is deprecated; use InterventionConfig instead.

7. DETERMINISTIC DECISIONS
   - Given the same InterventionContext, decide() ALWAYS returns the same
     InterventionDecision. No randomness, no external state.
   - This enables testing, replay, and audit.

8. CHECKPOINT RECOVERY
   - If a graph resumes from checkpoint, _tc_intervention state is restored.
   - InterventionEngine reconstructs its internal state from _tc_intervention.
   - No loss of intervention context across checkpoint/resume cycles.
"""


# =============================================================================
# SECTION 17: ERROR TAXONOMY (V7 Extensions)
# =============================================================================


class InterventionError(Exception):
    """Base exception for V7 intervention subsystem."""

    def __init__(
        self,
        message: str,
        *,
        decision: Optional[InterventionDecision] = None,
        context: Optional[InterventionContext] = None,
    ) -> None:
        ...


class HardStopError(InterventionError):
    """
    Raised when HARD_STOP stage is reached.
    Contains the full decision and context for debugging.
    Inherits from InterventionError, NOT TokenCircuitError (V6).
    """

    def __init__(
        self,
        message: str,
        *,
        decision: InterventionDecision,
        context: InterventionContext,
        total_tokens_saved: int = 0,
        total_cost_saved_usd: float = 0.0,
    ) -> None:
        ...


class TranscriptCorruptionError(InterventionError):
    """
    Raised in strict_mode when transcript validation fails irrecoverably.
    NOT raised in normal mode (auto-repair handles it silently).
    """

    def __init__(
        self,
        message: str,
        *,
        dropped_call_ids: list[str],
        repair_actions: list[str],
    ) -> None:
        ...
