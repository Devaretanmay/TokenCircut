"""
InterventionEngine — central orchestrator for V7 pre-model intervention.

Fuses exact-loop signals, semantic stagnation scores, and transcript health
to decide between PASS, NUDGE, OVERRIDE, or HARD_STOP.

State Machine:
    PASS ──(signals detected, count ≥ nudge_threshold)──▶ NUDGE
    NUDGE ──(persists ≥ override_threshold)──────────────▶ OVERRIDE
    OVERRIDE ──(persists ≥ hard_stop_threshold)──────────▶ HARD_STOP
    Any ──(no signals for 1 turn)────────────────────────▶ PASS (with cooldown)

Cooldown:
    After de-escalation, cooldown_turns must pass before re-escalation.
    During cooldown, the engine returns PASS regardless of signals.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .budget import BudgetEnforcer
from .canonicalizer import MessageCanonicalizer
from .ledger import ToolTransactionLedger
from .semantic_detector import SemanticStagnationDetector, StagnationAnalysis
from .state_schema import InterventionStateSchema, default_intervention_state
from .types import (
    CanonicalMessage,
    CanonicalRole,
    InterventionContext,
    InterventionDecision,
    InterventionStage,
    SignalType,
    TransactionOutcome,
)
from .validator import TranscriptValidator

logger = logging.getLogger("tokencircuit.engine")


def _get_tracer(name: str = "tokencircuit"):
    try:
        from opentelemetry import trace  # pyright: ignore[reportMissingImports]
        return trace.get_tracer(name)
    except ImportError:
        return None


class TokenCircuitError(RuntimeError):
    pass


@dataclass
class InterventionConfig:
    """Configuration for the V7 InterventionEngine."""

    # V6 compatibility
    window_size: int = 5

    # Enterprise Features
    audit_mode: bool = False
    max_tokens_per_turn: int = 4000

    # Stage thresholds (consecutive stagnation turns to escalate)
    nudge_threshold: int = 3
    override_threshold: int = 5
    hard_stop_threshold: int = 8

    # Cooldown
    cooldown_turns: int = 2

    # Capacity
    max_threads: int = 1000

    # Semantic detection
    similarity_threshold: float = 0.92
    enable_semantic_detection: bool = True

    # Transaction validation
    enable_transcript_validation: bool = True
    max_orphan_tolerance: int = 2
    auto_recovery: bool = True

    # Budget
    max_budget_usd: float = 0.0
    token_pricing: dict[str, float] = field(
        default_factory=lambda: {
            "gpt-4o": 5.0,
            "gpt-4o-mini": 0.15,
            "claude-3-5-sonnet": 3.0,
        }
    )

    # Coaching
    nudge_template: str = (
        "I notice you've been repeating a similar approach for {n_turns} turns. "
        "{outcome_summary}. "
        "Consider a different strategy: {suggestion}"
    )
    override_template: str = (
        "SYSTEM DIRECTIVE: Your last {n_turns} attempts used the same strategy "
        "and did not make progress. You MUST abandon the current approach. "
        "Errors seen: {error_summary}. "
        "Required action: {directive}"
    )

    def __post_init__(self) -> None:
        if self.nudge_threshold < 1:
            raise ValueError("nudge_threshold must be >= 1")
        if self.override_threshold <= self.nudge_threshold:
            raise ValueError("override_threshold must be > nudge_threshold")
        if self.hard_stop_threshold <= self.override_threshold:
            raise ValueError("hard_stop_threshold must be > override_threshold")


@dataclass
class _PerThreadState:
    """Internal per-thread-per-node state for the engine."""

    ledger: ToolTransactionLedger = field(default_factory=ToolTransactionLedger)
    detector: Optional[SemanticStagnationDetector] = None
    canonicalizer: MessageCanonicalizer = field(default_factory=MessageCanonicalizer)
    last_analysis: Optional[StagnationAnalysis] = None


class InterventionEngine:
    """
    Central orchestrator for the V7 pre-model intervention pipeline.

    Coordinates MessageCanonicalizer, TranscriptValidator, and
    SemanticStagnationDetector to produce InterventionDecisions.

    One engine per graph instance. Internal state is per thread_id+node_name.
    Persistent state lives in _tc_intervention graph state channel.
    """

    def __init__(self, *, config: Optional[InterventionConfig] = None) -> None:
        self._config = config or InterventionConfig()
        self._thread_states: OrderedDict[str, _PerThreadState] = OrderedDict()
        self._budget_enforcer = BudgetEnforcer(
            max_budget_usd=self._config.max_budget_usd,
            token_pricing=self._config.token_pricing,
        )

    def _get_state(self, thread_id: str, node_name: str) -> _PerThreadState:
        """Get or create per-thread-per-node internal state."""
        key = f"{thread_id}:{node_name}"
        if key not in self._thread_states:
            # Enforce max_threads LRU eviction
            if len(self._thread_states) >= self._config.max_threads:
                self._thread_states.popitem(last=False)

            detector = None
            if self._config.enable_semantic_detection:
                detector = SemanticStagnationDetector(
                    window_size=self._config.window_size,
                    similarity_threshold=self._config.similarity_threshold,
                )
            self._thread_states[key] = _PerThreadState(
                ledger=ToolTransactionLedger(),
                detector=detector,
                canonicalizer=MessageCanonicalizer(),
            )
        else:
            # Mark as recently used
            self._thread_states.move_to_end(key)

        return self._thread_states[key]

    def process(
        self,
        messages: Sequence[Any],
        state: dict[str, Any],
        *,
        thread_id: str,
        node_name: str,
    ) -> InterventionDecision:
        tracer = _get_tracer()
        ctx_manager = (
            tracer.start_as_current_span(
                "TokenCircuit.Intervention",
                attributes={
                    "thread_id": thread_id,
                    "node_name": node_name,
                    "audit_mode": self._config.audit_mode,
                }
            ) if tracer else nullcontext()
        )
        with ctx_manager as span:
            decision = self._process_impl(
                messages, state, thread_id=thread_id, node_name=node_name
            )

            if decision.stage > InterventionStage.PASS and span:
                span.set_attribute("intervention.stage", decision.stage.name)
                for sig in decision.signals:
                    span.add_event("SignalDetected", {"signal.type": sig.value})

            return decision

    def _process_impl(
        self,
        messages: Sequence[Any],
        state: dict[str, Any],
        *,
        thread_id: str,
        node_name: str,
    ) -> InterventionDecision:
        ts = self._get_state(thread_id, node_name)
        tc_state: InterventionStateSchema = state.get(
            "_tc_intervention", default_intervention_state()
        )
        turn_number = tc_state.get("turn_counter", 0) + 1

        canonical = ts.canonicalizer.canonicalize(list(messages))

        validation_signals, dropped_this_turn = [], []
        if self._config.enable_transcript_validation:
            validator = TranscriptValidator(
                ledger=ts.ledger,
                auto_recovery=self._config.auto_recovery,
                max_orphan_tolerance=self._config.max_orphan_tolerance,
            )
            res = validator.validate(canonical, turn_number)
            canonical, validation_signals, dropped_this_turn = (
                res.validated_messages,
                res.signals,
                res.dropped_call_ids,
            )

        stagnation_signals, similarity_score = [], 0.0
        if ts.detector:
            if not ts.detector._window:
                ts.detector.hydrate_from_history(canonical)
            analysis = ts.detector.analyze(canonical, turn_number)
            stagnation_signals, similarity_score = (
                analysis.signals,
                analysis.similarity_score,
            )
            ts.detector.record_fingerprint(analysis.fingerprint)
            ts.last_analysis = analysis

        runaway_signals = self._detect_runaway(canonical)
        all_signals = list(
            set(validation_signals + stagnation_signals + runaway_signals)
        )

        prior_stagnation = tc_state.get("consecutive_stagnation_count", 0)
        consecutive_stagnation = prior_stagnation + 1 if all_signals else 0
        current_stage = InterventionStage(
            _stage_str_to_int(tc_state.get("current_stage", "pass"))
        )
        cooldown_remaining = max(0, tc_state.get("cooldown_remaining", 0) - 1)

        context = InterventionContext(
            thread_id=thread_id,
            node_name=node_name,
            turn_number=turn_number,
            active_signals=all_signals,
            semantic_similarity_score=similarity_score,
            orphaned_transaction_ids=[t.call.call_id for t in ts.ledger.get_orphaned()],
            dropped_this_turn=dropped_this_turn,
            consecutive_empty_results=ts.ledger.get_consecutive_outcomes(
                TransactionOutcome.EMPTY
            ),
            consecutive_errors=ts.ledger.get_consecutive_outcomes(
                TransactionOutcome.TRANSIENT_ERROR
            )
            + ts.ledger.get_consecutive_outcomes(TransactionOutcome.PERMANENT_ERROR),
            current_stage=current_stage,
            consecutive_stagnation_count=consecutive_stagnation,
            total_interventions=tc_state.get("total_interventions", 0),
            cooldown_remaining=cooldown_remaining,
            strategies_attempted=tc_state.get("strategies_attempted", []),
        )

        return self.decide(context, canonical)

    def _detect_runaway(self, canonical: list[CanonicalMessage]) -> list[SignalType]:
        if self._config.max_tokens_per_turn > 0 and canonical:
            last_msg = canonical[-1]
            if last_msg.role == CanonicalRole.AI and last_msg.content:
                if len(last_msg.content) // 4 > self._config.max_tokens_per_turn:
                    return [SignalType.RUNAWAY_GENERATION]
        return []

    def decide(
        self,
        context: InterventionContext,
        canonical_messages: Optional[list[CanonicalMessage]] = None,
    ) -> InterventionDecision:
        if context.cooldown_remaining > 0:
            return InterventionDecision(
                stage=InterventionStage.PASS,
                signals=context.active_signals,
                state_patch=self._build_state_patch(
                    context, InterventionStage.PASS, None
                ),
            )

        if not context.active_signals:
            patch = self._build_state_patch(context, InterventionStage.PASS, None)
            if context.current_stage > InterventionStage.PASS:
                patch.update(
                    {
                        "cooldown_remaining": self._config.cooldown_turns,
                        "last_deescalation_turn": context.turn_number,
                    }
                )
            return InterventionDecision(
                stage=InterventionStage.PASS, signals=[], state_patch=patch
            )

        target_stage = self._get_target_stage(context)
        max_allowed = InterventionStage(
            min(context.current_stage + 1, InterventionStage.HARD_STOP)
        )
        is_runaway = SignalType.RUNAWAY_GENERATION in context.active_signals
        effective_stage = (
            target_stage
            if target_stage == InterventionStage.HARD_STOP and is_runaway
            else InterventionStage(min(target_stage, max_allowed))
        )

        if effective_stage == InterventionStage.PASS:
            return InterventionDecision(
                stage=InterventionStage.PASS,
                signals=context.active_signals,
                state_patch=self._build_state_patch(
                    context, InterventionStage.PASS, None
                ),
            )

        coaching = self._generate_coaching(effective_stage, context)
        llm_messages = self._build_intervention_messages(
            effective_stage, canonical_messages, coaching, context
        )

        return InterventionDecision(
            stage=effective_stage,
            signals=context.active_signals,
            llm_input_messages=llm_messages,
            coaching_message=coaching,
            should_terminate=effective_stage == InterventionStage.HARD_STOP,
            termination_reason=coaching
            if effective_stage == InterventionStage.HARD_STOP
            else None,
            state_patch=self._build_state_patch(context, effective_stage, coaching),
            estimated_tokens_saved=self._estimate_savings(context),
        )

    def _get_target_stage(self, context: InterventionContext) -> InterventionStage:
        is_runaway = SignalType.RUNAWAY_GENERATION in context.active_signals
        is_hard_stop = (
            context.consecutive_stagnation_count >= self._config.hard_stop_threshold
        )
        if is_runaway or is_hard_stop:
            return InterventionStage.HARD_STOP
        if context.consecutive_stagnation_count >= self._config.override_threshold:
            return InterventionStage.OVERRIDE
        if context.consecutive_stagnation_count >= self._config.nudge_threshold:
            return InterventionStage.NUDGE
        return InterventionStage.PASS

    def _generate_coaching(
        self, stage: InterventionStage, context: InterventionContext
    ) -> str:
        if stage == InterventionStage.NUDGE:
            return self._config.nudge_template.format(
                n_turns=context.consecutive_stagnation_count,
                outcome_summary=self._summarize_outcomes(context),
                suggestion=self._suggest_alternative(context),
            )
        if stage == InterventionStage.OVERRIDE:
            return self._config.override_template.format(
                n_turns=context.consecutive_stagnation_count,
                error_summary=self._summarize_errors(context),
                directive=self._generate_directive(context),
            )
        return (
            f"TokenCircuit HARD_STOP: Stagnation for "
            f"{context.consecutive_stagnation_count} turns in "
            f"'{context.node_name}'. Signals: "
            f"{[s.value for s in context.active_signals]}."
        )

    def _build_intervention_messages(self, stage, canonical, coaching, context):
        if canonical is None:
            return [{"role": "system", "content": coaching}]
        if stage == InterventionStage.NUDGE:
            res = MessageCanonicalizer.to_openai_format(canonical)
            res.append({"role": "system", "content": coaching})
            return res
        if stage == InterventionStage.OVERRIDE:
            return self._build_override_messages(canonical, coaching, context)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Message construction (ephemeral)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_override_messages(
        self,
        canonical: list[CanonicalMessage],
        coaching: str,
        context: InterventionContext,
    ) -> list[dict[str, Any]]:
        """
        Build llm_input_messages for OVERRIDE:
        Compact failed transactions + inject force-pivot directive.

        Strategy:
        - Keep system messages.
        - Keep the first human message (original task).
        - Drop repeated failed tool call/result pairs (compact).
        - Summarize what was tried.
        - Append the override directive.
        """
        if canonical is None:
            return [{"role": "system", "content": coaching}]

        compacted: list[CanonicalMessage] = []
        seen_tool_signatures: set[str] = set()
        dropped_call_ids: set[str] = set()
        failed_summary_parts: list[str] = []

        for msg in canonical:
            # ALWAYS keep system messages
            if msg.role == CanonicalRole.SYSTEM:
                compacted.append(msg)
                continue

            # Keep first human message
            if msg.role == CanonicalRole.HUMAN and not any(
                m.role == CanonicalRole.HUMAN for m in compacted
            ):
                compacted.append(msg)
                continue

            # For AI messages with tool_calls: deduplicate by tool signature
            if msg.role == CanonicalRole.AI and msg.tool_calls:
                sig = self._tool_calls_signature(msg.tool_calls)
                if sig in seen_tool_signatures:
                    # Summarize and skip this call
                    names = [tc.get("name", "?") for tc in msg.tool_calls]
                    failed_summary_parts.append(f"Repeated call to {', '.join(names)}")
                    for tc in msg.tool_calls:
                        cid = tc.get("id")
                        if cid:
                            dropped_call_ids.add(cid)
                    continue
                seen_tool_signatures.add(sig)
                compacted.append(msg)
                continue

            # For Tool results: drop if the call was dropped
            if msg.role == CanonicalRole.TOOL and msg.tool_call_id:
                if msg.tool_call_id in dropped_call_ids:
                    continue

            # Keep all other messages up to a reasonable limit
            if len(compacted) < 20:
                compacted.append(msg)

        # Convert to dicts
        result = MessageCanonicalizer.to_openai_format(compacted)

        # Add summary of compacted failures
        if failed_summary_parts:
            summary = (
                f"[TokenCircuit: Compacted {len(failed_summary_parts)} repeated failed "
                f"transactions: {'; '.join(failed_summary_parts[:5])}]"
            )
            result.append({"role": "system", "content": summary})

        # Append the override directive
        result.append({"role": "system", "content": coaching})

        return result

    @staticmethod
    def _tool_calls_signature(tool_calls: list[dict[str, Any]]) -> str:
        """Create a deduplication signature for a set of tool calls."""
        parts: list[str] = []
        for tc in sorted(tool_calls, key=lambda x: x.get("name", "")):
            name = tc.get("name", "")
            args = tc.get("args", {})
            if isinstance(args, dict):
                arg_types = ",".join(sorted(type(v).__name__ for v in args.values()))
            else:
                arg_types = "?"
            parts.append(f"{name}({arg_types})")
        return "|".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # State patch construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_state_patch(
        self,
        context: InterventionContext,
        stage: InterventionStage,
        coaching: Optional[str],
    ) -> dict[str, Any]:
        """Build the _tc_intervention state patch."""
        patch: dict[str, Any] = {
            "turn_counter": context.turn_number,
            "current_stage": stage.name.lower(),
            "previous_stage": context.current_stage.name.lower(),
            "consecutive_stagnation_count": (
                context.consecutive_stagnation_count if context.active_signals else 0
            ),
            "cooldown_remaining": context.cooldown_remaining,
        }

        if stage > InterventionStage.PASS:
            patch["total_interventions"] = context.total_interventions + 1
            patch["last_escalation_turn"] = context.turn_number

        if stage == InterventionStage.NUDGE:
            patch["nudge_count"] = context.total_interventions + 1
        elif stage == InterventionStage.OVERRIDE:
            patch["override_count"] = context.total_interventions + 1

        if coaching:
            patch["last_coaching_message"] = coaching[:500]
            patch["coaching_history"] = [coaching[:200]]

        if stage > context.current_stage:
            patch["stage_entered_at_turn"] = context.turn_number

        if context.dropped_this_turn:
            patch["dropped_this_session"] = context.dropped_this_turn
            patch["orphaned_transaction_ids"] = context.orphaned_transaction_ids

        # Semantic window state
        patch["last_similarity_score"] = context.semantic_similarity_score

        return patch

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _summarize_outcomes(self, context: InterventionContext) -> str:
        """Summarize recent tool outcomes."""
        parts: list[str] = []
        if context.consecutive_empty_results > 0:
            parts.append(f"{context.consecutive_empty_results} empty results")
        if context.consecutive_errors > 0:
            parts.append(f"{context.consecutive_errors} errors")
        return ", ".join(parts) if parts else "repeated identical results"

    def _summarize_errors(self, context: InterventionContext) -> str:
        """Summarize error patterns for override message."""
        parts: list[str] = []
        if context.consecutive_errors > 0:
            parts.append(f"{context.consecutive_errors} consecutive errors")
        if context.consecutive_empty_results > 0:
            parts.append(f"{context.consecutive_empty_results} empty responses")
        if context.orphaned_transaction_ids:
            parts.append(f"{len(context.orphaned_transaction_ids)} orphaned calls")
        return "; ".join(parts) if parts else "no progress detected"

    def _suggest_alternative(self, context: InterventionContext) -> str:
        """Suggest an alternative strategy based on observed patterns."""
        if context.consecutive_empty_results > 0:
            return "Try different search terms or a different tool entirely"
        if context.consecutive_errors > 0:
            return (
                "The current approach is failing. Try a fundamentally different method"
            )
        if SignalType.SEMANTIC_STAGNATION in context.active_signals:
            return (
                "You're rephrasing the same approach. Change your strategy completely"
            )
        return "Try a different tool or approach to solve this problem"

    def _generate_directive(self, context: InterventionContext) -> str:
        """Generate a specific directive for override."""
        attempted = context.strategies_attempted
        if attempted:
            return (
                f"Do NOT repeat: {', '.join(attempted[-3:])}. "
                "Use a completely novel approach."
            )
        return "Stop using the current tool. Choose a different strategy entirely."

    def _estimate_savings(self, context: InterventionContext) -> int:
        """Estimate tokens saved by intervention."""
        # Rough heuristic: each prevented loop iteration saves ~1500 tokens
        remaining_turns = max(
            0, self._config.hard_stop_threshold - context.consecutive_stagnation_count
        )
        return remaining_turns * 1500

    # ─────────────────────────────────────────────────────────────────────────
    # Public state access
    # ─────────────────────────────────────────────────────────────────────────

    def get_engine_state(self, thread_id: str, node_name: str) -> dict[str, Any]:
        """Return internal engine state for debugging."""
        key = f"{thread_id}:{node_name}"
        ts = self._thread_states.get(key)
        if ts is None:
            return {"exists": False}
        return {
            "exists": True,
            "ledger_committed": ts.ledger.total_committed,
            "ledger_orphaned": ts.ledger.total_orphaned,
            "ledger_pending": len(ts.ledger.get_pending()),
            "detector_window_size": ts.detector.window_size if ts.detector else 0,
            "last_similarity": (
                ts.last_analysis.similarity_score if ts.last_analysis else 0.0
            ),
        }

    def reset(self, thread_id: str, node_name: str) -> None:
        """Reset internal state for a thread+node."""
        key = f"{thread_id}:{node_name}"
        self._thread_states.pop(key, None)

    def reset_all(self) -> None:
        """Reset all internal state."""
        self._thread_states.clear()
        self._budget_enforcer.reset()

    def record_usage(self, model: str, tokens: int) -> float:
        """Record usage and enforce budget."""
        return self._budget_enforcer.record_usage(model, tokens)

    @property
    def current_spend(self) -> float:
        return self._budget_enforcer.current_spend

    @property
    def config(self) -> InterventionConfig:
        return self._config


def _stage_str_to_int(stage_str: str) -> int:
    """Convert stage string to IntEnum value."""
    mapping = {
        "pass": 0,
        "nudge": 1,
        "override": 2,
        "hard_stop": 3,
    }
    return mapping.get(stage_str.lower(), 0)
