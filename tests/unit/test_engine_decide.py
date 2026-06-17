"""
Comprehensive unit tests for InterventionEngine.decide() and InterventionConfig.

Tests the pure decision logic in isolation, without requiring any of the
heavy pipeline components (SemanticStagnationDetector, TranscriptValidator, etc.).
"""
from __future__ import annotations

import pytest

from tokencircuit.engine import (
    InterventionConfig,
    InterventionEngine,
    _stage_str_to_int,
)
from tokencircuit.types import (
    InterventionContext,
    InterventionStage,
    SignalType,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def default_config() -> InterventionConfig:
    """Default InterventionConfig with standard thresholds (3/5/8)."""
    return InterventionConfig()


@pytest.fixture
def engine(default_config: InterventionConfig) -> InterventionEngine:
    """Engine with default config, ready for .decide() calls."""
    return InterventionEngine(config=default_config)


@pytest.fixture
def low_threshold_config() -> InterventionConfig:
    """Config with low thresholds (1/2/3) for easier boundary testing."""
    return InterventionConfig(
        nudge_threshold=1,
        override_threshold=2,
        hard_stop_threshold=3,
    )


@pytest.fixture
def low_engine(low_threshold_config: InterventionConfig) -> InterventionEngine:
    """Engine with low thresholds for quicker escalation tests."""
    return InterventionEngine(config=low_threshold_config)


def _make_context(**overrides) -> InterventionContext:
    """
    Factory helper that builds an InterventionContext with sensible defaults.
    Any keyword argument overrides the corresponding field.
    """
    defaults = dict(
        thread_id="test-thread",
        node_name="agent",
        turn_number=1,
        active_signals=[],
        semantic_similarity_score=0.0,
        orphaned_transaction_ids=[],
        dropped_this_turn=[],
        consecutive_empty_results=0,
        consecutive_errors=0,
        current_stage=InterventionStage.PASS,
        consecutive_stagnation_count=0,
        total_interventions=0,
        cooldown_remaining=0,
        strategies_attempted=[],
    )
    defaults.update(overrides)
    return InterventionContext(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. InterventionConfig validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterventionConfigValidation:
    """Tests for InterventionConfig.__post_init__ validation."""

    def test_config_default_is_valid(self):
        """Default config should instantiate without errors."""
        cfg = InterventionConfig()
        assert cfg.nudge_threshold == 3
        assert cfg.override_threshold == 5
        assert cfg.hard_stop_threshold == 8

    def test_config_nudge_threshold_zero_raises(self):
        """nudge_threshold < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="nudge_threshold must be >= 1"):
            InterventionConfig(nudge_threshold=0)

    def test_config_nudge_threshold_negative_raises(self):
        """Negative nudge_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="nudge_threshold must be >= 1"):
            InterventionConfig(nudge_threshold=-5)

    def test_config_override_must_exceed_nudge(self):
        """override_threshold must be strictly > nudge_threshold."""
        with pytest.raises(ValueError, match="override_threshold must be > nudge_threshold"):  # noqa: E501
            InterventionConfig(nudge_threshold=3, override_threshold=3)

    def test_config_override_below_nudge_raises(self):
        """override_threshold below nudge_threshold should raise."""
        with pytest.raises(ValueError, match="override_threshold must be > nudge_threshold"):  # noqa: E501
            InterventionConfig(nudge_threshold=5, override_threshold=3)

    def test_config_hard_stop_must_exceed_override(self):
        """hard_stop_threshold must be strictly > override_threshold."""
        with pytest.raises(ValueError, match="hard_stop_threshold must be > override_threshold"):  # noqa: E501
            InterventionConfig(
                nudge_threshold=1, override_threshold=2, hard_stop_threshold=2
            )

    def test_config_hard_stop_below_override_raises(self):
        """hard_stop_threshold below override_threshold should raise."""
        with pytest.raises(ValueError, match="hard_stop_threshold must be > override_threshold"):  # noqa: E501
            InterventionConfig(
                nudge_threshold=1, override_threshold=3, hard_stop_threshold=2
            )

    def test_config_minimum_valid_thresholds(self):
        """The smallest valid threshold set is 1/2/3."""
        cfg = InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3
        )
        assert cfg.nudge_threshold == 1
        assert cfg.override_threshold == 2
        assert cfg.hard_stop_threshold == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cooldown gate
# ═══════════════════════════════════════════════════════════════════════════════


class TestCooldownGate:
    """When cooldown_remaining > 0 the engine always returns PASS."""

    def test_cooldown_forces_pass_with_signals(self, engine: InterventionEngine):
        """Even with active signals and high stagnation, cooldown → PASS."""
        ctx = _make_context(
            cooldown_remaining=2,
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=10,
            current_stage=InterventionStage.NUDGE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.PASS

    def test_cooldown_preserves_signals_in_decision(self, engine: InterventionEngine):
        """During cooldown the decision still reports the detected signals."""
        signals = [SignalType.FUTILE_ACTION, SignalType.SEMANTIC_STAGNATION]
        ctx = _make_context(cooldown_remaining=1, active_signals=signals)
        decision = engine.decide(ctx)
        assert set(decision.signals) == set(signals)

    def test_cooldown_zero_allows_escalation(self, low_engine: InterventionEngine):
        """When cooldown just expired (== 0), normal escalation applies."""
        ctx = _make_context(
            cooldown_remaining=0,
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE


# ═══════════════════════════════════════════════════════════════════════════════
# 3. No signals → de-escalation to PASS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeescalation:
    """With no signals the engine de-escalates back to PASS."""

    def test_no_signals_returns_pass(self, engine: InterventionEngine):
        """Empty active_signals → PASS."""
        ctx = _make_context(active_signals=[])
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.PASS

    def test_deescalation_from_nudge_sets_cooldown(self, engine: InterventionEngine):
        """De-escalating from NUDGE should set cooldown_remaining in state_patch."""
        ctx = _make_context(
            active_signals=[],
            current_stage=InterventionStage.NUDGE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.PASS
        expected_cooldown = engine.config.cooldown_turns
        assert decision.state_patch["cooldown_remaining"] == expected_cooldown

    def test_deescalation_from_override_sets_cooldown(self, engine: InterventionEngine):
        """De-escalating from OVERRIDE should also set cooldown."""
        ctx = _make_context(
            active_signals=[],
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = engine.decide(ctx)
        assert decision.state_patch["cooldown_remaining"] == engine.config.cooldown_turns  # noqa: E501

    def test_deescalation_from_hard_stop_sets_cooldown(self, engine: InterventionEngine):  # noqa: E501
        """De-escalating from HARD_STOP should set cooldown."""
        ctx = _make_context(
            active_signals=[],
            current_stage=InterventionStage.HARD_STOP,
        )
        decision = engine.decide(ctx)
        assert decision.state_patch["cooldown_remaining"] == engine.config.cooldown_turns  # noqa: E501

    def test_deescalation_from_pass_no_cooldown(self, engine: InterventionEngine):
        """When already PASS and no signals, cooldown should remain 0."""
        ctx = _make_context(
            active_signals=[],
            current_stage=InterventionStage.PASS,
            cooldown_remaining=0,
        )
        decision = engine.decide(ctx)
        # cooldown_remaining in patch stays at context value (0), not overwritten
        assert decision.state_patch.get("cooldown_remaining", 0) == 0

    def test_deescalation_sets_last_deescalation_turn(self, engine: InterventionEngine):
        """De-escalation from non-PASS records last_deescalation_turn."""
        ctx = _make_context(
            active_signals=[],
            current_stage=InterventionStage.NUDGE,
            turn_number=7,
        )
        decision = engine.decide(ctx)
        assert decision.state_patch["last_deescalation_turn"] == 7

    def test_deescalation_signals_list_empty(self, engine: InterventionEngine):
        """When de-escalating (no signals), decision.signals should be empty."""
        ctx = _make_context(
            active_signals=[],
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = engine.decide(ctx)
        assert decision.signals == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. State patch includes required keys
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatePatch:
    """Verify that state_patch always includes the core bookkeeping keys."""

    def test_patch_has_turn_counter(self, engine: InterventionEngine):
        """state_patch must include turn_counter."""
        ctx = _make_context(turn_number=42)
        decision = engine.decide(ctx)
        assert decision.state_patch["turn_counter"] == 42

    def test_patch_has_current_stage(self, engine: InterventionEngine):
        """state_patch must include current_stage as lowercase string."""
        ctx = _make_context()
        decision = engine.decide(ctx)
        assert decision.state_patch["current_stage"] == "pass"

    def test_patch_has_consecutive_stagnation_count_zero_on_no_signals(
        self, engine: InterventionEngine
    ):
        """consecutive_stagnation_count resets to 0 when no signals."""
        ctx = _make_context(active_signals=[], consecutive_stagnation_count=5)
        decision = engine.decide(ctx)
        assert decision.state_patch["consecutive_stagnation_count"] == 0

    def test_patch_preserves_stagnation_count_with_signals(
        self, engine: InterventionEngine
    ):
        """consecutive_stagnation_count is carried through when signals present."""
        ctx = _make_context(
            active_signals=[SignalType.FUTILE_ACTION],
            consecutive_stagnation_count=4,
        )
        decision = engine.decide(ctx)
        assert decision.state_patch["consecutive_stagnation_count"] == 4

    def test_patch_increments_total_interventions_on_escalation(
        self, low_engine: InterventionEngine
    ):
        """total_interventions should increment when stage > PASS."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
            total_interventions=3,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE
        assert decision.state_patch["total_interventions"] == 4

    def test_patch_records_previous_stage(self, engine: InterventionEngine):
        """state_patch should record the previous stage."""
        ctx = _make_context(current_stage=InterventionStage.NUDGE, active_signals=[])
        decision = engine.decide(ctx)
        assert decision.state_patch["previous_stage"] == "nudge"

    def test_patch_records_stage_entered_at_turn_on_escalation(
        self, low_engine: InterventionEngine
    ):
        """When escalating to a higher stage, stage_entered_at_turn is recorded."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
            turn_number=10,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE
        assert decision.state_patch["stage_entered_at_turn"] == 10


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Signals below nudge_threshold → PASS (building evidence)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBelowNudgeThreshold:
    """When signals exist but count < nudge_threshold, result is PASS."""

    def test_count_1_with_default_threshold_3_is_pass(self, engine: InterventionEngine):
        """stagnation_count=1 < nudge_threshold=3 → PASS."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.PASS

    def test_count_2_with_default_threshold_3_is_pass(self, engine: InterventionEngine):
        """stagnation_count=2 < nudge_threshold=3 → PASS."""
        ctx = _make_context(
            active_signals=[SignalType.FUTILE_ACTION],
            consecutive_stagnation_count=2,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.PASS

    def test_pass_still_reports_active_signals(self, engine: InterventionEngine):
        """Even below threshold, the decision carries signals."""
        signals = [SignalType.STATE_STAGNATION]
        ctx = _make_context(active_signals=signals, consecutive_stagnation_count=1)
        decision = engine.decide(ctx)
        assert decision.signals == signals


# ═══════════════════════════════════════════════════════════════════════════════
# 6. count >= nudge_threshold → NUDGE
# ═══════════════════════════════════════════════════════════════════════════════


class TestNudgeThreshold:
    """Reaching nudge_threshold triggers NUDGE (subject to monotonic constraint)."""

    def test_exact_nudge_threshold_triggers_nudge(self, engine: InterventionEngine):
        """count == nudge_threshold (3) → NUDGE (from PASS)."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=3,
            current_stage=InterventionStage.PASS,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE

    def test_above_nudge_still_nudge_if_below_override(
        self, engine: InterventionEngine
    ):
        """count=4 → target is NUDGE (4 ≥ 3 but < 5)."""
        ctx = _make_context(
            active_signals=[SignalType.SEMANTIC_STAGNATION],
            consecutive_stagnation_count=4,
            current_stage=InterventionStage.NUDGE,
        )
        decision = engine.decide(ctx)
        # Already at NUDGE, target is NUDGE → NUDGE
        assert decision.stage == InterventionStage.NUDGE


# ═══════════════════════════════════════════════════════════════════════════════
# 7. count >= override_threshold → OVERRIDE
# ═══════════════════════════════════════════════════════════════════════════════


class TestOverrideThreshold:
    """Reaching override_threshold triggers OVERRIDE (if monotonic allows)."""

    def test_exact_override_threshold_from_nudge(self, engine: InterventionEngine):
        """count=5 from NUDGE → OVERRIDE (5 ≥ override_threshold=5)."""
        ctx = _make_context(
            active_signals=[SignalType.FUTILE_ACTION],
            consecutive_stagnation_count=5,
            current_stage=InterventionStage.NUDGE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.OVERRIDE

    def test_above_override_below_hard_stop(self, engine: InterventionEngine):
        """count=6 from OVERRIDE → still OVERRIDE (6 < 8)."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=6,
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.OVERRIDE


# ═══════════════════════════════════════════════════════════════════════════════
# 8. count >= hard_stop_threshold → HARD_STOP
# ═══════════════════════════════════════════════════════════════════════════════


class TestHardStopThreshold:
    """Reaching hard_stop_threshold triggers HARD_STOP with termination."""

    def test_exact_hard_stop_threshold(self, engine: InterventionEngine):
        """count=8 from OVERRIDE → HARD_STOP."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=8,
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP
        assert decision.should_terminate is True

    def test_hard_stop_sets_termination_reason(self, engine: InterventionEngine):
        """HARD_STOP decision must populate termination_reason."""
        ctx = _make_context(
            active_signals=[SignalType.FUTILE_ACTION],
            consecutive_stagnation_count=8,
            current_stage=InterventionStage.OVERRIDE,
            node_name="my_agent",
        )
        decision = engine.decide(ctx)
        assert decision.termination_reason is not None
        assert "HARD_STOP" in decision.termination_reason
        assert "my_agent" in decision.termination_reason

    def test_hard_stop_above_threshold(self, engine: InterventionEngine):
        """count=12 well above threshold still HARD_STOP."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=12,
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Monotonic escalation: can only go up one level per turn
# ═══════════════════════════════════════════════════════════════════════════════


class TestMonotonicEscalation:
    """Engine enforces that escalation cannot skip stages (except RUNAWAY)."""

    def test_pass_cannot_jump_to_override(self, engine: InterventionEngine):
        """From PASS, even with count >= override_threshold, caps at NUDGE."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=5,  # meets override_threshold
            current_stage=InterventionStage.PASS,
        )
        decision = engine.decide(ctx)
        # Monotonic: PASS + 1 = NUDGE
        assert decision.stage == InterventionStage.NUDGE

    def test_pass_cannot_jump_to_hard_stop(self, engine: InterventionEngine):
        """From PASS with count=8, still caps at NUDGE."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=8,
            current_stage=InterventionStage.PASS,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE

    def test_nudge_cannot_jump_to_hard_stop(self, engine: InterventionEngine):
        """From NUDGE with count=8, caps at OVERRIDE."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=8,
            current_stage=InterventionStage.NUDGE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.OVERRIDE

    def test_override_can_reach_hard_stop(self, engine: InterventionEngine):
        """From OVERRIDE with count=8, goes to HARD_STOP (only one step up)."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=8,
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP

    def test_already_at_stage_stays(self, engine: InterventionEngine):
        """If already at NUDGE and target is NUDGE, stays at NUDGE."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=3,
            current_stage=InterventionStage.NUDGE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE

    def test_hard_stop_stays_at_hard_stop(self, engine: InterventionEngine):
        """Already at HARD_STOP → capped at HARD_STOP."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=10,
            current_stage=InterventionStage.HARD_STOP,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP


# ═══════════════════════════════════════════════════════════════════════════════
# 10. RUNAWAY_GENERATION bypasses monotonic escalation
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunawayGenerationBypass:
    """RUNAWAY_GENERATION signal skips the monotonic ladder → instant HARD_STOP."""

    def test_runaway_from_pass_is_hard_stop(self, engine: InterventionEngine):
        """RUNAWAY_GENERATION from PASS jumps directly to HARD_STOP."""
        ctx = _make_context(
            active_signals=[SignalType.RUNAWAY_GENERATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP
        assert decision.should_terminate is True

    def test_runaway_from_nudge_is_hard_stop(self, engine: InterventionEngine):
        """RUNAWAY_GENERATION from NUDGE still goes to HARD_STOP."""
        ctx = _make_context(
            active_signals=[SignalType.RUNAWAY_GENERATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.NUDGE,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP

    def test_runaway_with_other_signals(self, engine: InterventionEngine):
        """RUNAWAY_GENERATION mixed with other signals still goes to HARD_STOP."""
        ctx = _make_context(
            active_signals=[
                SignalType.RUNAWAY_GENERATION,
                SignalType.STATE_STAGNATION,
            ],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP

    def test_runaway_hard_stop_has_termination_reason(
        self, engine: InterventionEngine
    ):
        """RUNAWAY-triggered HARD_STOP also sets termination_reason."""
        ctx = _make_context(
            active_signals=[SignalType.RUNAWAY_GENERATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = engine.decide(ctx)
        assert decision.termination_reason is not None
        assert len(decision.termination_reason) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Coaching messages
# ═══════════════════════════════════════════════════════════════════════════════


class TestCoachingMessages:
    """Verify that NUDGE and OVERRIDE generate coaching_message text."""

    def test_nudge_has_coaching_message(self, low_engine: InterventionEngine):
        """NUDGE decision must populate coaching_message."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE
        assert decision.coaching_message is not None
        assert len(decision.coaching_message) > 0

    def test_nudge_uses_nudge_template(self, low_engine: InterventionEngine):
        """NUDGE coaching should contain the n_turns from the template."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = low_engine.decide(ctx)
        # The default template includes "for {n_turns} turns"
        msg = decision.coaching_message
        assert "1 turns" in msg or "1 turn" in msg

    def test_override_has_coaching_message(self, low_engine: InterventionEngine):
        """OVERRIDE decision must populate coaching_message."""
        ctx = _make_context(
            active_signals=[SignalType.FUTILE_ACTION],
            consecutive_stagnation_count=2,
            current_stage=InterventionStage.NUDGE,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.OVERRIDE
        assert decision.coaching_message is not None
        assert len(decision.coaching_message) > 0

    def test_override_uses_override_template(self, low_engine: InterventionEngine):
        """OVERRIDE coaching should contain SYSTEM DIRECTIVE from template."""
        ctx = _make_context(
            active_signals=[SignalType.FUTILE_ACTION],
            consecutive_stagnation_count=2,
            current_stage=InterventionStage.NUDGE,
        )
        decision = low_engine.decide(ctx)
        assert "SYSTEM DIRECTIVE" in decision.coaching_message

    def test_hard_stop_coaching_equals_termination_reason(
        self, low_engine: InterventionEngine
    ):
        """HARD_STOP sets coaching_message equal to the termination reason."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=3,
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP
        assert decision.coaching_message == decision.termination_reason

    def test_pass_has_no_coaching_message(self, engine: InterventionEngine):
        """PASS decision should NOT have a coaching message."""
        ctx = _make_context(active_signals=[])
        decision = engine.decide(ctx)
        assert decision.coaching_message is None


# ═══════════════════════════════════════════════════════════════════════════════
# 12. estimated_tokens_saved calculation
# ═══════════════════════════════════════════════════════════════════════════════


class TestEstimatedTokensSaved:
    """Verify the heuristic tokens-saved estimator."""

    def test_tokens_saved_nudge(self, low_engine: InterventionEngine):
        """NUDGE at count=1 with hard_stop=3 → (3-1)*1500 = 3000."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE
        assert decision.estimated_tokens_saved == (3 - 1) * 1500  # 3000

    def test_tokens_saved_hard_stop(self, low_engine: InterventionEngine):
        """HARD_STOP at count=3 with hard_stop=3 → (3-3)*1500 = 0."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=3,
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.HARD_STOP
        assert decision.estimated_tokens_saved == 0

    def test_tokens_saved_is_zero_for_pass(self, engine: InterventionEngine):
        """PASS decision should have 0 estimated tokens saved."""
        ctx = _make_context(active_signals=[])
        decision = engine.decide(ctx)
        assert decision.estimated_tokens_saved == 0

    def test_tokens_saved_never_negative(self, low_engine: InterventionEngine):
        """Even if count > hard_stop_threshold, savings should never go negative."""
        ctx = _make_context(
            active_signals=[SignalType.RUNAWAY_GENERATION],
            consecutive_stagnation_count=100,
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = low_engine.decide(ctx)
        assert decision.estimated_tokens_saved >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 13. _stage_str_to_int mapping
# ═══════════════════════════════════════════════════════════════════════════════


class TestStageStrToInt:
    """Tests for the module-level _stage_str_to_int helper."""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("pass", 0),
            ("nudge", 1),
            ("override", 2),
            ("hard_stop", 3),
        ],
    )
    def test_known_stages(self, input_str: str, expected: int):
        """Lowercase stage names map to correct IntEnum values."""
        assert _stage_str_to_int(input_str) == expected

    @pytest.mark.parametrize(
        "input_str",
        [
            "PASS", "Pass", "NUDGE", "Nudge",
            "OVERRIDE", "Override", "HARD_STOP", "Hard_Stop",
        ],
    )
    def test_case_insensitive(self, input_str: str):
        """Mapping is case-insensitive."""
        result = _stage_str_to_int(input_str)
        assert result in {0, 1, 2, 3}

    def test_unknown_stage_defaults_to_zero(self):
        """Unknown stage strings default to 0 (PASS)."""
        assert _stage_str_to_int("banana") == 0
        assert _stage_str_to_int("") == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 14. _tool_calls_signature sorting
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolCallsSignature:
    """Tests for InterventionEngine._tool_calls_signature static method."""

    def test_empty_list(self):
        """Empty tool_calls list produces empty signature."""
        assert InterventionEngine._tool_calls_signature([]) == ""

    def test_single_tool_call(self):
        """Single tool call signature includes name and arg types."""
        tc = [{"name": "search", "args": {"query": "hello", "limit": 5}}]
        sig = InterventionEngine._tool_calls_signature(tc)
        assert "search" in sig

    def test_deterministic_ordering(self):
        """Signature is deterministic regardless of input order."""
        tc_a = [
            {"name": "beta", "args": {"x": 1}},
            {"name": "alpha", "args": {"y": "hi"}},
        ]
        tc_b = [
            {"name": "alpha", "args": {"y": "hi"}},
            {"name": "beta", "args": {"x": 1}},
        ]
        sig_a = InterventionEngine._tool_calls_signature(tc_a)
        sig_b = InterventionEngine._tool_calls_signature(tc_b)
        assert sig_a == sig_b

    def test_different_args_different_types_produce_different_signatures(self):
        """Calls with the same name but different arg types produce different sigs."""
        tc_int = [{"name": "search", "args": {"query": 42}}]
        tc_str = [{"name": "search", "args": {"query": "42"}}]
        sig_int = InterventionEngine._tool_calls_signature(tc_int)
        sig_str = InterventionEngine._tool_calls_signature(tc_str)
        assert sig_int != sig_str

    def test_non_dict_args_produces_question_mark(self):
        """When args is not a dict, signature uses '?' for arg types."""
        tc = [{"name": "tool", "args": "raw_string"}]
        sig = InterventionEngine._tool_calls_signature(tc)
        assert "?" in sig

    def test_no_name_field(self):
        """Missing 'name' key should not crash; uses empty string."""
        tc = [{"args": {"x": 1}}]
        sig = InterventionEngine._tool_calls_signature(tc)
        assert isinstance(sig, str)

    def test_no_args_field(self):
        """Missing 'args' key defaults to empty dict → no arg types."""
        tc = [{"name": "tool"}]
        sig = InterventionEngine._tool_calls_signature(tc)
        assert "tool" in sig


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Engine.process() fail-safe
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# 16. Full escalation ladder walk-through
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullEscalationLadder:
    """Walk the escalation ladder from PASS through HARD_STOP over successive turns."""

    def test_step_by_step_escalation(self, low_engine: InterventionEngine):
        """
        With thresholds 1/2/3 and monotonic constraint:
        Turn 1: count=1, PASS→NUDGE
        Turn 2: count=2, NUDGE→OVERRIDE
        Turn 3: count=3, OVERRIDE→HARD_STOP
        """
        # Turn 1: PASS → NUDGE
        ctx1 = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
            turn_number=1,
        )
        d1 = low_engine.decide(ctx1)
        assert d1.stage == InterventionStage.NUDGE

        # Turn 2: NUDGE → OVERRIDE
        ctx2 = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=2,
            current_stage=InterventionStage.NUDGE,
            turn_number=2,
        )
        d2 = low_engine.decide(ctx2)
        assert d2.stage == InterventionStage.OVERRIDE

        # Turn 3: OVERRIDE → HARD_STOP
        ctx3 = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=3,
            current_stage=InterventionStage.OVERRIDE,
            turn_number=3,
        )
        d3 = low_engine.decide(ctx3)
        assert d3.stage == InterventionStage.HARD_STOP
        assert d3.should_terminate is True


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Edge cases & miscellaneous
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Assorted edge-case scenarios."""

    def test_multiple_signal_types(self, low_engine: InterventionEngine):
        """Multiple distinct signals are all forwarded in the decision."""
        signals = [
            SignalType.STATE_STAGNATION,
            SignalType.FUTILE_ACTION,
            SignalType.TRANSCRIPT_CORRUPTION,
        ]
        ctx = _make_context(
            active_signals=signals,
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = low_engine.decide(ctx)
        assert set(decision.signals) == set(signals)

    def test_pass_with_signals_below_threshold_no_coaching(
        self, engine: InterventionEngine
    ):
        """PASS with signals but below threshold should have no coaching."""
        ctx = _make_context(
            active_signals=[SignalType.FUTILE_ACTION],
            consecutive_stagnation_count=1,
        )
        decision = engine.decide(ctx)
        assert decision.coaching_message is None

    def test_engine_config_property(self, engine: InterventionEngine):
        """Engine.config property returns the config."""
        assert engine.config.nudge_threshold == 3

    def test_nudge_state_patch_has_nudge_count(self, low_engine: InterventionEngine):
        """NUDGE decision patch includes nudge_count."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
            total_interventions=0,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.NUDGE
        assert "nudge_count" in decision.state_patch

    def test_override_state_patch_has_override_count(
        self, low_engine: InterventionEngine
    ):
        """OVERRIDE decision patch includes override_count."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=2,
            current_stage=InterventionStage.NUDGE,
            total_interventions=1,
        )
        decision = low_engine.decide(ctx)
        assert decision.stage == InterventionStage.OVERRIDE
        assert "override_count" in decision.state_patch

    def test_hard_stop_termination_reason_contains_signals(
        self, low_engine: InterventionEngine
    ):
        """HARD_STOP termination reason lists the active signals."""
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION, SignalType.FUTILE_ACTION],
            consecutive_stagnation_count=3,
            current_stage=InterventionStage.OVERRIDE,
        )
        decision = low_engine.decide(ctx)
        assert "STATE_STAGNATION" in decision.termination_reason
        assert "FUTILE_ACTION" in decision.termination_reason

    def test_last_similarity_score_in_patch(self, engine: InterventionEngine):
        """state_patch should include last_similarity_score."""
        ctx = _make_context(
            active_signals=[],
            semantic_similarity_score=0.95,
        )
        decision = engine.decide(ctx)
        assert decision.state_patch["last_similarity_score"] == 0.95

    def test_dropped_this_turn_recorded_in_patch(self, engine: InterventionEngine):
        """dropped_this_turn list is forwarded into state_patch."""
        ctx = _make_context(
            active_signals=[SignalType.TOOL_TRANSACTION_ORPHAN],
            consecutive_stagnation_count=1,
            dropped_this_turn=["call_001", "call_002"],
        )
        decision = engine.decide(ctx)
        patch = decision.state_patch
        assert patch.get("dropped_this_session") == ["call_001", "call_002"]

    def test_coaching_message_truncated_in_patch(self, low_engine: InterventionEngine):
        """Coaching message in patch is capped at 500 chars."""
        # Use a custom template that generates a very long message
        cfg = InterventionConfig(
            nudge_threshold=1,
            override_threshold=2,
            hard_stop_threshold=3,
            nudge_template="A" * 1000 + " {n_turns} {outcome_summary} {suggestion}",
        )
        eng = InterventionEngine(config=cfg)
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = eng.decide(ctx)
        assert len(decision.state_patch["last_coaching_message"]) <= 500

    def test_coaching_history_truncated_in_patch(self, low_engine: InterventionEngine):
        """coaching_history entries in patch are capped at 200 chars."""
        cfg = InterventionConfig(
            nudge_threshold=1,
            override_threshold=2,
            hard_stop_threshold=3,
            nudge_template="B" * 500 + " {n_turns} {outcome_summary} {suggestion}",
        )
        eng = InterventionEngine(config=cfg)
        ctx = _make_context(
            active_signals=[SignalType.STATE_STAGNATION],
            consecutive_stagnation_count=1,
            current_stage=InterventionStage.PASS,
        )
        decision = eng.decide(ctx)
        for entry in decision.state_patch.get("coaching_history", []):
            assert len(entry) <= 200
