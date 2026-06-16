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
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .canonicalizer import MessageCanonicalizer
from .ledger import ToolTransactionLedger
from .semantic_detector import SemanticStagnationDetector, StagnationAnalysis
from .state_schema import InterventionStateSchema, default_intervention_state
from .telemetry import MetricsCollector, get_tracer
from .types import (
    CanonicalMessage,
    CanonicalRole,
    InterventionContext,
    InterventionDecision,
    InterventionStage,
    SignalType,
    TransactionOutcome,
)
from .validator import TranscriptValidator, ValidationResult

logger = logging.getLogger("tokencircuit.engine")


class _NullContextManager:
    """No-op context manager used when OpenTelemetry is not available."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


@dataclass
class InterventionConfig:
    """Configuration for the V7 InterventionEngine."""

    # V6 compatibility
    window_size: int = 5
    model_name: str = "unknown"
    telemetry_enabled: bool = True

    # Enterprise Features
    audit_mode: bool = False
    max_tokens_per_turn: int = 4000
    agency_id: Optional[str] = None
    client_id: Optional[str] = None

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
    auto_repair: bool = True

    # Coaching
    nudge_template: str = (
        "I notice you've been repeating a similar approach for {n_turns} turns. "
        "The tool '{tool_name}' has returned {outcome_summary}. "
        "Consider a different strategy: {suggestion}"
    )
    override_template: str = (
        "SYSTEM DIRECTIVE: Your last {n_turns} attempts used the same strategy "
        "and did not make progress. You MUST abandon the current approach. "
        "Failed tool: '{tool_name}'. Errors seen: {error_summary}. "
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
        """Main entry point. Runs the full V7 pipeline synchronously."""
        tracer = get_tracer()
        ctx_manager = (
            tracer.start_as_current_span(
                "TokenCircuit.Intervention",
                attributes={
                    "thread_id": thread_id,
                    "node_name": node_name,
                    "audit_mode": self._config.audit_mode,
                }
            ) if tracer else _NullContextManager()
        )
        with ctx_manager as span:
            try:
                decision = self._process_impl(messages, state, thread_id=thread_id, node_name=node_name)

                metrics = MetricsCollector()
                if decision.stage > InterventionStage.PASS:
                    if span:
                        span.set_attribute("intervention.stage", decision.stage.name)
                        for sig in decision.signals:
                            span.add_event("SignalDetected", {"signal.type": sig.value})
                    metrics.record_intervention(
                        stage=decision.stage.name,
                        model=self._config.model_name,
                        tokens_saved=decision.estimated_tokens_saved
                    )

                ts = self._thread_states.get(f"{thread_id}:{node_name}")
                if ts and ts.last_analysis:
                    metrics.record_stagnation_score(ts.last_analysis.similarity_score)

                return decision
            except Exception as exc:
                if span:
                    try:
                        span.record_exception(exc)
                    except Exception:
                        pass
                logger.error(
                    "InterventionEngine.process() failed, returning PASS: %s", exc, exc_info=True
                )
                return InterventionDecision(stage=InterventionStage.PASS, signals=[], state_patch={})

    def _process_impl(
        self,
        messages: Sequence[Any],
        state: dict[str, Any],
        *,
        thread_id: str,
        node_name: str,
    ) -> InterventionDecision:
        """Internal implementation (can raise)."""
        ts = self._get_state(thread_id, node_name)

        # ─── Step 1: Extract intervention state ───
        tc_state: InterventionStateSchema = state.get(
            "_tc_intervention", default_intervention_state()
        )
        turn_number = tc_state.get("turn_counter", 0) + 1

        # ─── Step 2: Canonicalize messages ───
        canonical = ts.canonicalizer.canonicalize(list(messages))
        raw_count = len(messages)

        # ─── Step 3: Validate transcript ───
        validation_result: Optional[ValidationResult] = None
        validation_signals: list[SignalType] = []
        dropped_this_turn: list[str] = []

        if self._config.enable_transcript_validation:
            validator = TranscriptValidator(
                ledger=ts.ledger,
                auto_repair=self._config.auto_repair,
                max_orphan_tolerance=self._config.max_orphan_tolerance,
            )
            validation_result = validator.validate(canonical, turn_number)
            canonical = validation_result.validated_messages
            validation_signals = validation_result.signals
            dropped_this_turn = validation_result.dropped_call_ids

        validated_count = len(canonical)

        # ─── Step 4: Semantic stagnation detection ───
        analysis: Optional[StagnationAnalysis] = None
        stagnation_signals: list[SignalType] = []
        similarity_score = 0.0
        pattern_diversity = 1.0

        if ts.detector is not None:
            # OPTIMIZATION: Only hydrate from history if our in-memory window is empty.
            # This happens after process restarts or on the very first turn.
            if len(ts.detector._window) == 0:
                ts.detector.hydrate_from_history(canonical)

            analysis = ts.detector.analyze(canonical, turn_number)
            stagnation_signals = analysis.signals
            similarity_score = analysis.similarity_score
            pattern_diversity = analysis.pattern_diversity

            # Record fingerprint for next turn's comparison
            ts.detector.record_fingerprint(analysis.fingerprint)
            ts.last_analysis = analysis

        # ─── Step 4.5: Runaway Generation Detection ───
        runaway_signals: list[SignalType] = []
        if self._config.max_tokens_per_turn > 0 and canonical:
            # Check the last message if it's from the AI
            last_msg = canonical[-1]
            if last_msg.role == CanonicalRole.AI and last_msg.content:
                # Simple estimation: 1 token ≈ 4 characters
                estimated_tokens = len(last_msg.content) // 4
                if estimated_tokens > self._config.max_tokens_per_turn:
                    logger.warning(
                        "TokenCircuit: RUNAWAY GENERATION detected. "
                        "Estimated %d tokens exceeds limit of %d.",
                        estimated_tokens,
                        self._config.max_tokens_per_turn,
                    )
                    runaway_signals.append(SignalType.RUNAWAY_GENERATION)

        # ─── Step 5: Merge all signals ───
        all_signals = list(set(validation_signals + stagnation_signals + runaway_signals))

        # ─── Step 6: Compute transaction health metrics ───
        consecutive_empty = ts.ledger.get_consecutive_outcomes(TransactionOutcome.EMPTY)
        consecutive_errors = ts.ledger.get_consecutive_outcomes(
            TransactionOutcome.TRANSIENT_ERROR
        ) + ts.ledger.get_consecutive_outcomes(TransactionOutcome.PERMANENT_ERROR)

        # ─── Step 7: Build InterventionContext ───
        # Track consecutive stagnation from prior state
        prior_stagnation = tc_state.get("consecutive_stagnation_count", 0)
        if all_signals:
            consecutive_stagnation = prior_stagnation + 1
        else:
            consecutive_stagnation = 0

        current_stage = InterventionStage(
            _stage_str_to_int(tc_state.get("current_stage", "pass"))
        )
        cooldown_remaining = max(0, tc_state.get("cooldown_remaining", 0) - 1)

        context = InterventionContext(
            thread_id=thread_id,
            node_name=node_name,
            turn_number=turn_number,
            canonical_messages=[
                {"role": m.role.value, "content": m.content[:100]}
                for m in canonical[-5:]  # last 5 for context, truncated
            ],
            raw_message_count=raw_count,
            validated_message_count=validated_count,
            active_signals=all_signals,
            semantic_similarity_score=similarity_score,
            pattern_diversity=pattern_diversity,
            pending_transactions=len(ts.ledger.get_pending()),
            orphaned_transaction_ids=[t.call.call_id for t in ts.ledger.get_orphaned()],
            dropped_this_turn=dropped_this_turn,
            consecutive_empty_results=consecutive_empty,
            consecutive_errors=consecutive_errors,
            current_stage=current_stage,
            consecutive_stagnation_count=consecutive_stagnation,
            total_interventions=tc_state.get("total_interventions", 0),
            cooldown_remaining=cooldown_remaining,
            strategies_attempted=tc_state.get("strategies_attempted", []),
        )

        # ─── Step 8: Decide ───
        decision = self.decide(context, canonical)

        return decision

    def decide(
        self,
        context: InterventionContext,
        canonical_messages: Optional[list[CanonicalMessage]] = None,
    ) -> InterventionDecision:
        """
        Pure decision function. Given context, produce intervention decision.

        State Machine:
        - If cooldown > 0: PASS (cooling down, no matter what).
        - If no signals: de-escalate → PASS.
        - If signals:
          count < nudge_threshold → PASS (building evidence)
          nudge_threshold ≤ count < override_threshold → NUDGE
          override_threshold ≤ count < hard_stop_threshold → OVERRIDE
          count ≥ hard_stop_threshold → HARD_STOP
        """
        cfg = self._config

        # ─── Cooldown gate ───
        if context.cooldown_remaining > 0:
            return InterventionDecision(
                stage=InterventionStage.PASS,
                signals=context.active_signals,
                state_patch=self._build_state_patch(
                    context, InterventionStage.PASS, coaching=None
                ),
            )

        # ─── No signals → de-escalate ───
        if not context.active_signals:
            patch = self._build_state_patch(context, InterventionStage.PASS, coaching=None)
            # Apply cooldown if we're de-escalating from non-PASS
            if context.current_stage > InterventionStage.PASS:
                patch["cooldown_remaining"] = cfg.cooldown_turns
                patch["last_deescalation_turn"] = context.turn_number
            return InterventionDecision(
                stage=InterventionStage.PASS,
                signals=[],
                state_patch=patch,
            )

        # ─── Determine target stage based on consecutive count or explicit signals ───
        count = context.consecutive_stagnation_count

        if SignalType.RUNAWAY_GENERATION in context.active_signals:
            target_stage = InterventionStage.HARD_STOP
        elif count >= cfg.hard_stop_threshold:
            target_stage = InterventionStage.HARD_STOP
        elif count >= cfg.override_threshold:
            target_stage = InterventionStage.OVERRIDE
        elif count >= cfg.nudge_threshold:
            target_stage = InterventionStage.NUDGE
        else:
            target_stage = InterventionStage.PASS

        # ─── Monotonic escalation: can only go up by one level per turn ───
        # Exception: Runaway Generation skips the ladder and hard stops immediately.
        if target_stage == InterventionStage.HARD_STOP and SignalType.RUNAWAY_GENERATION in context.active_signals:
            effective_stage = InterventionStage.HARD_STOP
        else:
            max_allowed = InterventionStage(min(context.current_stage + 1, InterventionStage.HARD_STOP))
            effective_stage = InterventionStage(min(target_stage, max_allowed))

        # ─── Build decision based on effective stage ───
        if effective_stage == InterventionStage.PASS:
            return InterventionDecision(
                stage=InterventionStage.PASS,
                signals=context.active_signals,
                state_patch=self._build_state_patch(context, InterventionStage.PASS, coaching=None),
            )

        elif effective_stage == InterventionStage.NUDGE:
            coaching = self._generate_nudge(context)
            llm_messages = self._build_nudge_messages(canonical_messages, coaching)
            return InterventionDecision(
                stage=InterventionStage.NUDGE,
                signals=context.active_signals,
                llm_input_messages=llm_messages,
                coaching_message=coaching,
                state_patch=self._build_state_patch(context, InterventionStage.NUDGE, coaching),
                estimated_tokens_saved=self._estimate_savings(context),
            )

        elif effective_stage == InterventionStage.OVERRIDE:
            coaching = self._generate_override(context)
            llm_messages = self._build_override_messages(canonical_messages, coaching, context)
            return InterventionDecision(
                stage=InterventionStage.OVERRIDE,
                signals=context.active_signals,
                llm_input_messages=llm_messages,
                coaching_message=coaching,
                state_patch=self._build_state_patch(context, InterventionStage.OVERRIDE, coaching),
                estimated_tokens_saved=self._estimate_savings(context),
            )

        else:  # HARD_STOP
            reason = self._generate_hard_stop_reason(context)
            return InterventionDecision(
                stage=InterventionStage.HARD_STOP,
                signals=context.active_signals,
                should_terminate=True,
                termination_reason=reason,
                coaching_message=reason,
                state_patch=self._build_state_patch(context, InterventionStage.HARD_STOP, reason),
                estimated_tokens_saved=self._estimate_savings(context),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Coaching message generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_nudge(self, context: InterventionContext) -> str:
        """Generate a coaching message for NUDGE stage."""
        outcome_summary = self._summarize_outcomes(context)
        suggestion = self._suggest_alternative(context)

        return self._config.nudge_template.format(
            n_turns=context.consecutive_stagnation_count,
            tool_name="unknown",
            outcome_summary=outcome_summary,
            suggestion=suggestion,
        )

    def _generate_override(self, context: InterventionContext) -> str:
        """Generate a forceful directive for OVERRIDE stage."""
        error_summary = self._summarize_errors(context)
        directive = self._generate_directive(context)

        return self._config.override_template.format(
            n_turns=context.consecutive_stagnation_count,
            tool_name="unknown",
            error_summary=error_summary,
            directive=directive,
        )

    def _generate_hard_stop_reason(self, context: InterventionContext) -> str:
        """Generate termination reason."""
        return (
            f"TokenCircuit HARD_STOP: Agent has been stagnating for "
            f"{context.consecutive_stagnation_count} consecutive turns in node "
            f"'{context.node_name}'. Signals: "
            f"{[s.value for s in context.active_signals]}. "
            f"Total interventions attempted: {context.total_interventions}."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Message construction (ephemeral)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_nudge_messages(
        self,
        canonical: Optional[list[CanonicalMessage]],
        coaching: str,
    ) -> list[dict[str, Any]]:
        """
        Build llm_input_messages for NUDGE: original messages + appended system coaching.
        """
        if canonical is None:
            return [{"role": "system", "content": coaching}]

        # Convert canonical back to dicts, then append coaching
        result: list[dict[str, Any]] = []
        canonicalizer = MessageCanonicalizer()
        result = canonicalizer.to_openai_format(canonical)

        # Append coaching as a system message at the end
        result.append({"role": "system", "content": coaching})
        return result

    def _build_override_messages(
        self,
        canonical: Optional[list[CanonicalMessage]],
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
        canonicalizer = MessageCanonicalizer()
        result = canonicalizer.to_openai_format(compacted)

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
            patch["orphaned_transaction_ids"] = context.dropped_this_turn

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
            return "The current approach is failing. Try a fundamentally different method"
        if SignalType.SEMANTIC_STAGNATION in context.active_signals:
            return "You're rephrasing the same approach. Change your strategy completely"
        return "Try a different tool or approach to solve this problem"

    def _generate_directive(self, context: InterventionContext) -> str:
        """Generate a specific directive for override."""
        attempted = context.strategies_attempted
        if attempted:
            return f"Do NOT repeat: {', '.join(attempted[-3:])}. Use a completely novel approach."
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
