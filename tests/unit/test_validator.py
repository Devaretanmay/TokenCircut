"""
Comprehensive unit tests for TranscriptValidator.

Covers the 10 invariants documented in validator.py, plus helper functions,
signal emission, incremental validation, and statelessness guarantees.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from tokencircuit.types import (
    CanonicalMessage,
    CanonicalRole,
    SignalType,
    TransactionStatus,
)
from tokencircuit.ledger import ToolTransactionLedger
from tokencircuit.validator import (
    TranscriptValidator,
    ValidationResult,
    _hash_args,
    _hash_content,
    _is_args_valid,
    _extract_type_signature,
)


# ---------------------------------------------------------------------------
# Helpers — build messages quickly
# ---------------------------------------------------------------------------

def _system(content: str = "You are a helpful assistant.", *, idx: int = 0) -> CanonicalMessage:
    return CanonicalMessage(role=CanonicalRole.SYSTEM, content=content, source_index=idx)


def _human(content: str, *, idx: int) -> CanonicalMessage:
    return CanonicalMessage(role=CanonicalRole.HUMAN, content=content, source_index=idx)


def _ai(
    content: str = "",
    *,
    idx: int,
    tool_calls: list[dict[str, Any]] | None = None,
) -> CanonicalMessage:
    return CanonicalMessage(
        role=CanonicalRole.AI,
        content=content,
        tool_calls=tool_calls or [],
        source_index=idx,
    )


def _tool(content: str, *, call_id: str, idx: int) -> CanonicalMessage:
    return CanonicalMessage(
        role=CanonicalRole.TOOL,
        content=content,
        tool_call_id=call_id,
        source_index=idx,
    )


def _tc(call_id: str, name: str = "search", args: dict | None = None) -> dict:
    """Shortcut to build a tool_call dict."""
    return {"id": call_id, "name": name, "args": args or {"query": "test"}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger() -> ToolTransactionLedger:
    return ToolTransactionLedger(orphan_timeout_turns=5)


@pytest.fixture
def validator(ledger: ToolTransactionLedger) -> TranscriptValidator:
    return TranscriptValidator(ledger=ledger, auto_repair=True, max_orphan_tolerance=2)


@pytest.fixture
def strict_validator(ledger: ToolTransactionLedger) -> TranscriptValidator:
    return TranscriptValidator(
        ledger=ledger,
        auto_repair=False,
        strict_mode=True,
        max_orphan_tolerance=0,
    )


# ===========================================================================
# 1. Clean transcript — happy path
# ===========================================================================

class TestCleanTranscript:
    """A well-formed transcript should pass validation with no drops."""

    def test_clean_transcript_is_valid(self, validator: TranscriptValidator) -> None:
        """Verify is_valid=True when every tool result matches a prior call."""
        msgs = [
            _system(idx=0),
            _human("hi", idx=1),
            _ai(idx=2, tool_calls=[_tc("c1")]),
            _tool("result", call_id="c1", idx=3),
            _ai("done", idx=4),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is True
        assert result.dropped_indices == []
        assert result.dropped_call_ids == []
        assert result.signals == []

    def test_clean_transcript_preserves_all_messages(self, validator: TranscriptValidator) -> None:
        """All messages appear in validated_messages."""
        msgs = [
            _system(idx=0),
            _human("hi", idx=1),
            _ai(idx=2, tool_calls=[_tc("c1")]),
            _tool("result", call_id="c1", idx=3),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert len(result.validated_messages) == len(msgs)

    def test_multiple_tool_calls_clean(self, validator: TranscriptValidator) -> None:
        """Multiple tool calls in one AI message, all resolved cleanly."""
        msgs = [
            _human("go", idx=0),
            _ai(idx=1, tool_calls=[_tc("c1"), _tc("c2", name="write")]),
            _tool("r1", call_id="c1", idx=2),
            _tool("r2", call_id="c2", idx=3),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is True
        assert len(result.validated_messages) == 4


# ===========================================================================
# 2. Invariant 1: CALL-BEFORE-RESULT
# ===========================================================================

class TestCallBeforeResult:
    """Tool result referencing a non-existent call_id must be dropped."""

    def test_result_with_unknown_call_id_dropped(self, validator: TranscriptValidator) -> None:
        """A tool message whose call_id never appears in any AI is dropped."""
        msgs = [
            _human("hi", idx=0),
            _ai("thinking", idx=1),
            _tool("phantom", call_id="no_such_call", idx=2),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert 2 in result.dropped_indices
        assert "no_such_call" in result.dropped_call_ids

    def test_valid_result_not_dropped(self, validator: TranscriptValidator) -> None:
        """A result with a matching call is NOT dropped."""
        msgs = [
            _ai(idx=0, tool_calls=[_tc("c1")]),
            _tool("ok", call_id="c1", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert 1 not in result.dropped_indices


# ===========================================================================
# 3. Invariant 2: RESULT-AFTER-CALL
# ===========================================================================

class TestResultAfterCall:
    """Tool result must appear after its corresponding AI call message."""

    def test_result_before_call_is_dropped(self, validator: TranscriptValidator) -> None:
        """A tool message at an index <= its AI call index is dropped."""
        msgs = [
            _tool("too early", call_id="c1", idx=0),
            _ai(idx=1, tool_calls=[_tc("c1")]),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert 0 in result.dropped_indices

    def test_result_at_same_index_as_call_is_dropped(self, validator: TranscriptValidator) -> None:
        """Edge case: source_index == ai_index should still be dropped."""
        msgs = [
            _ai(idx=5, tool_calls=[_tc("c1")]),
            CanonicalMessage(
                role=CanonicalRole.TOOL, content="x", tool_call_id="c1", source_index=5,
            ),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert 5 in result.dropped_indices


# ===========================================================================
# 4. Invariant 3: NO-ORPHAN-RESULTS (auto_repair=True)
# ===========================================================================

class TestNoOrphanResults:
    """Orphaned tool results (no matching call) are dropped with auto_repair."""

    def test_orphan_dropped_with_auto_repair(self, validator: TranscriptValidator) -> None:
        """Orphan result is removed when auto_repair=True."""
        msgs = [
            _human("x", idx=0),
            _tool("orphan", call_id="ghost", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert 1 in result.dropped_indices

    def test_orphan_not_dropped_without_auto_repair(self, strict_validator: TranscriptValidator) -> None:
        """Without auto_repair the tool message is NOT added to dropped_indices
        (the code continues past the orphan check without dropping)."""
        msgs = [
            _human("x", idx=0),
            _tool("orphan", call_id="ghost", idx=1),
        ]
        result = strict_validator.validate(msgs, turn_number=1)
        # With auto_repair=False, the orphan is not added to dropped_indices
        assert 1 not in result.dropped_indices


# ===========================================================================
# 5. Invariant 4: NO-DUPLICATE-RESULTS
# ===========================================================================

class TestNoDuplicateResults:
    """Only the first tool result per call_id is kept; duplicates are dropped."""

    def test_duplicate_result_dropped(self, validator: TranscriptValidator) -> None:
        """Second tool message for the same call_id is dropped."""
        msgs = [
            _ai(idx=0, tool_calls=[_tc("c1")]),
            _tool("first", call_id="c1", idx=1),
            _tool("second", call_id="c1", idx=2),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert 2 in result.dropped_indices
        assert 1 not in result.dropped_indices

    def test_triple_duplicate_drops_second_and_third(self, validator: TranscriptValidator) -> None:
        """All results after the first for the same call_id are dropped."""
        msgs = [
            _ai(idx=0, tool_calls=[_tc("c1")]),
            _tool("1st", call_id="c1", idx=1),
            _tool("2nd", call_id="c1", idx=2),
            _tool("3rd", call_id="c1", idx=3),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert 2 in result.dropped_indices
        assert 3 in result.dropped_indices
        assert 1 not in result.dropped_indices


# ===========================================================================
# 6. Invariant 5: ATOMIC-CALL-DROP (malformed AI → drop all its results)
# ===========================================================================

class TestAtomicCallDrop:
    """If an AI message is deemed malformed, ALL matching tool results are dropped."""

    def test_malformed_ai_drops_all_matching_results(self, validator: TranscriptValidator) -> None:
        """A malformed AI message's tool results are all dropped atomically."""
        bad_tc = {"id": "c1", "name": "t", "args": {"_raw": "not json"}}
        msgs = [
            _ai(idx=0, tool_calls=[bad_tc, _tc("c2")]),
            _tool("r1", call_id="c1", idx=1),
            _tool("r2", call_id="c2", idx=2),
        ]
        result = validator.validate(msgs, turn_number=1)
        # Both tool results are dropped because the entire AI message is malformed
        assert 1 in result.dropped_indices
        assert 2 in result.dropped_indices

    def test_malformed_ai_still_appears_with_empty_tool_calls(self, validator: TranscriptValidator) -> None:
        """The AI message itself is kept but with tool_calls cleared to []."""
        bad_tc = {"id": "c1", "name": "t", "args": {"_raw": "bad"}}
        msgs = [
            _ai("I'll search", idx=0, tool_calls=[bad_tc]),
            _tool("r1", call_id="c1", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        ai_out = [m for m in result.validated_messages if m.role == CanonicalRole.AI]
        assert len(ai_out) == 1
        assert ai_out[0].tool_calls == []


# ===========================================================================
# 7. Invariant 7: MALFORMED-ARGS-DROP
# ===========================================================================

class TestMalformedArgsDrop:
    """Invalid args cause entire AI tool_calls to be voided."""

    def test_raw_only_dict_is_malformed(self, validator: TranscriptValidator) -> None:
        """A tool_call with args={'_raw': ...} (single key) triggers malformed."""
        msgs = [
            _ai(idx=0, tool_calls=[{"id": "c1", "name": "t", "args": {"_raw": "x"}}]),
            _tool("r", call_id="c1", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is False
        assert 1 in result.dropped_indices

    def test_non_dict_args_is_malformed(self, validator: TranscriptValidator) -> None:
        """args that is a list instead of dict is malformed."""
        msgs = [
            _ai(idx=0, tool_calls=[{"id": "c1", "name": "t", "args": [1, 2]}]),
            _tool("r", call_id="c1", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is False

    def test_mixed_good_and_bad_args_drops_all(self, validator: TranscriptValidator) -> None:
        """One bad tool_call in an AI message causes ALL calls to be voided."""
        msgs = [
            _ai(
                idx=0,
                tool_calls=[
                    _tc("c1"),  # good
                    {"id": "c2", "name": "t", "args": {"_raw": "bad"}},  # bad
                ],
            ),
            _tool("r1", call_id="c1", idx=1),
            _tool("r2", call_id="c2", idx=2),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert 1 in result.dropped_indices
        assert 2 in result.dropped_indices


# ===========================================================================
# 8. Invariant 8: CALL-ID-REQUIRED
# ===========================================================================

class TestCallIdRequired:
    """Tool calls without an 'id' field are malformed (triggers invariant 7)."""

    def test_missing_id_triggers_malformed(self, validator: TranscriptValidator) -> None:
        """A tool_call without 'id' marks the entire AI message malformed."""
        msgs = [
            _ai(idx=0, tool_calls=[{"name": "search", "args": {"q": "test"}}]),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is False
        # The AI message should have its tool_calls stripped
        ai_out = result.validated_messages[0]
        assert ai_out.tool_calls == []

    def test_empty_id_triggers_malformed(self, validator: TranscriptValidator) -> None:
        """An empty-string id is treated the same as missing."""
        msgs = [
            _ai(idx=0, tool_calls=[{"id": "", "name": "search", "args": {"q": "x"}}]),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is False


# ===========================================================================
# 9. Invariant 10: SYSTEM-MESSAGES-PASSTHROUGH
# ===========================================================================

class TestSystemMessagesPassthrough:
    """System messages are never dropped or modified by the validator."""

    def test_system_message_always_kept(self, validator: TranscriptValidator) -> None:
        """System messages survive even if surrounding messages are dropped."""
        msgs = [
            _system("sys prompt", idx=0),
            _tool("orphan", call_id="ghost", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        sys_msgs = [m for m in result.validated_messages if m.role == CanonicalRole.SYSTEM]
        assert len(sys_msgs) == 1
        assert sys_msgs[0].content == "sys prompt"

    def test_multiple_system_messages_preserved(self, validator: TranscriptValidator) -> None:
        """Multiple system messages all survive."""
        msgs = [
            _system("first", idx=0),
            _system("second", idx=1),
            _human("hi", idx=2),
        ]
        result = validator.validate(msgs, turn_number=1)
        sys_msgs = [m for m in result.validated_messages if m.role == CanonicalRole.SYSTEM]
        assert len(sys_msgs) == 2


# ===========================================================================
# 10. Orphan tolerance → TRANSCRIPT_CORRUPTION signal
# ===========================================================================

class TestOrphanTolerance:
    """orphan_count > max_orphan_tolerance triggers TRANSCRIPT_CORRUPTION."""

    def test_corruption_signal_when_tolerance_exceeded(self) -> None:
        """Exceeding max_orphan_tolerance emits TRANSCRIPT_CORRUPTION."""
        ledger = ToolTransactionLedger()
        v = TranscriptValidator(ledger=ledger, auto_repair=True, max_orphan_tolerance=1)
        msgs = [
            _human("hi", idx=0),
            _tool("o1", call_id="x1", idx=1),
            _tool("o2", call_id="x2", idx=2),
        ]
        result = v.validate(msgs, turn_number=1)
        assert SignalType.TRANSCRIPT_CORRUPTION in result.signals

    def test_no_corruption_signal_within_tolerance(self) -> None:
        """Orphans within tolerance do NOT emit TRANSCRIPT_CORRUPTION (from orphans alone)."""
        ledger = ToolTransactionLedger()
        v = TranscriptValidator(ledger=ledger, auto_repair=True, max_orphan_tolerance=5)
        msgs = [
            _human("hi", idx=0),
            _tool("o1", call_id="x1", idx=1),
        ]
        result = v.validate(msgs, turn_number=1)
        assert SignalType.TRANSCRIPT_CORRUPTION not in result.signals

    def test_malformed_ai_also_triggers_corruption(self, validator: TranscriptValidator) -> None:
        """Malformed AI (even with zero orphans) triggers TRANSCRIPT_CORRUPTION."""
        msgs = [
            _ai(idx=0, tool_calls=[{"name": "t", "args": {"q": "x"}}]),  # no id
        ]
        result = validator.validate(msgs, turn_number=1)
        assert SignalType.TRANSCRIPT_CORRUPTION in result.signals


# ===========================================================================
# 11. TOOL_TRANSACTION_ORPHAN signal
# ===========================================================================

class TestOrphanSignal:
    """TOOL_TRANSACTION_ORPHAN is emitted whenever orphans exist."""

    def test_orphan_signal_emitted(self, validator: TranscriptValidator) -> None:
        """At least one orphan → TOOL_TRANSACTION_ORPHAN signal."""
        msgs = [
            _human("hi", idx=0),
            _tool("orphan", call_id="missing", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert SignalType.TOOL_TRANSACTION_ORPHAN in result.signals

    def test_no_orphan_signal_on_clean_transcript(self, validator: TranscriptValidator) -> None:
        """Clean transcript → no TOOL_TRANSACTION_ORPHAN signal."""
        msgs = [
            _ai(idx=0, tool_calls=[_tc("c1")]),
            _tool("ok", call_id="c1", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert SignalType.TOOL_TRANSACTION_ORPHAN not in result.signals


# ===========================================================================
# 12. validate_incremental
# ===========================================================================

class TestValidateIncremental:
    """Tests for the incremental validation path."""

    def test_incremental_drops_unknown_call_id(self, validator: TranscriptValidator, ledger: ToolTransactionLedger) -> None:
        """A tool result with no ledger entry is dropped incrementally."""
        # Ledger is empty — no calls registered
        new_msgs = [_tool("result", call_id="unknown", idx=10)]
        valid, dropped = validator.validate_incremental(new_msgs, turn_number=5)
        assert len(valid) == 0
        assert "unknown" in dropped

    def test_incremental_drops_duplicate_committed(self, validator: TranscriptValidator, ledger: ToolTransactionLedger) -> None:
        """A tool result for an already-committed transaction is dropped."""
        # Manually register a call and commit it
        ledger.register_call(
            call_id="c1", tool_name="t", arguments_hash="h",
            arguments_type_signature="t(str)", source_message_index=0, turn_number=1,
        )
        ledger.register_result(
            call_id="c1", result_hash="rh", result_content_prefix="ok",
            result_length=2, source_message_index=1, turn_number=1,
        )
        new_msgs = [_tool("dup", call_id="c1", idx=5)]
        valid, dropped = validator.validate_incremental(new_msgs, turn_number=2)
        assert "c1" in dropped
        assert len(valid) == 0

    def test_incremental_keeps_pending_result(self, validator: TranscriptValidator, ledger: ToolTransactionLedger) -> None:
        """A tool result for a PENDING transaction is kept."""
        ledger.register_call(
            call_id="c1", tool_name="t", arguments_hash="h",
            arguments_type_signature="t(str)", source_message_index=0, turn_number=1,
        )
        new_msgs = [_tool("ok", call_id="c1", idx=5)]
        valid, dropped = validator.validate_incremental(new_msgs, turn_number=2)
        assert len(valid) == 1
        assert dropped == []

    def test_incremental_passes_non_tool_messages(self, validator: TranscriptValidator) -> None:
        """Non-tool messages are always passed through incrementally."""
        new_msgs = [
            _human("hello", idx=10),
            _ai("thinking", idx=11),
        ]
        valid, dropped = validator.validate_incremental(new_msgs, turn_number=3)
        assert len(valid) == 2
        assert dropped == []


# ===========================================================================
# 13. _hash_args consistency
# ===========================================================================

class TestHashArgs:
    """_hash_args must produce deterministic, consistent hashes."""

    def test_same_dict_same_hash(self) -> None:
        """Identical dicts produce the same hash."""
        d = {"query": "hello", "limit": 10}
        assert _hash_args(d) == _hash_args(d)

    def test_key_order_independent(self) -> None:
        """Dicts with same keys but inserted in different order get same hash."""
        a = {"b": 2, "a": 1}
        b = {"a": 1, "b": 2}
        assert _hash_args(a) == _hash_args(b)

    def test_different_dicts_different_hash(self) -> None:
        """Distinct dicts produce distinct hashes."""
        assert _hash_args({"a": 1}) != _hash_args({"a": 2})

    def test_string_input(self) -> None:
        """String args are hashed directly."""
        h = _hash_args("raw string")
        assert isinstance(h, str) and len(h) == 64  # SHA-256 hex digest

    def test_non_dict_non_str(self) -> None:
        """Non-dict, non-str inputs are str()-ified then hashed."""
        h = _hash_args(42)
        assert isinstance(h, str) and len(h) == 64

    def test_hash_content_deterministic(self) -> None:
        """_hash_content is also deterministic."""
        assert _hash_content("abc") == _hash_content("abc")
        assert _hash_content("abc") != _hash_content("xyz")


# ===========================================================================
# 14. _is_args_valid
# ===========================================================================

class TestIsArgsValid:
    """_is_args_valid rejects single-key {'_raw': ...} dicts."""

    def test_normal_dict_valid(self) -> None:
        assert _is_args_valid({"query": "test"}) is True

    def test_raw_only_invalid(self) -> None:
        """{'_raw': 'anything'} with no other keys is invalid."""
        assert _is_args_valid({"_raw": "bad data"}) is False

    def test_raw_with_other_keys_valid(self) -> None:
        """{'_raw': ..., 'extra': ...} is valid (more than one key)."""
        assert _is_args_valid({"_raw": "x", "other": 1}) is True

    def test_list_invalid(self) -> None:
        assert _is_args_valid([1, 2]) is False

    def test_string_invalid(self) -> None:
        assert _is_args_valid("not a dict") is False

    def test_none_invalid(self) -> None:
        assert _is_args_valid(None) is False

    def test_int_invalid(self) -> None:
        assert _is_args_valid(42) is False

    def test_empty_dict_valid(self) -> None:
        """Empty dict {} is a valid args value."""
        assert _is_args_valid({}) is True


# ===========================================================================
# 15. Statelessness — ledger.reset() called each validate()
# ===========================================================================

class TestStatelessness:
    """Validator must be stateless: ledger is reset on every validate() call."""

    def test_ledger_reset_between_calls(self, validator: TranscriptValidator, ledger: ToolTransactionLedger) -> None:
        """Running validate() twice on different transcripts yields independent results."""
        msgs1 = [
            _ai(idx=0, tool_calls=[_tc("c1")]),
            _tool("r1", call_id="c1", idx=1),
        ]
        r1 = validator.validate(msgs1, turn_number=1)
        assert r1.is_valid is True

        # Second validate with a completely different transcript
        msgs2 = [
            _ai(idx=0, tool_calls=[_tc("c2")]),
            _tool("r2", call_id="c2", idx=1),
        ]
        r2 = validator.validate(msgs2, turn_number=2)
        assert r2.is_valid is True
        # "c1" should NOT be in the ledger after second validate
        assert ledger.get_transaction("c1") is None

    def test_orphan_from_prior_run_not_carried(self, validator: TranscriptValidator) -> None:
        """Orphans from a previous validate() do not leak into the next run."""
        bad = [
            _human("x", idx=0),
            _tool("orphan", call_id="old_ghost", idx=1),
        ]
        r1 = validator.validate(bad, turn_number=1)
        assert r1.is_valid is False

        clean = [
            _ai(idx=0, tool_calls=[_tc("c1")]),
            _tool("ok", call_id="c1", idx=1),
        ]
        r2 = validator.validate(clean, turn_number=2)
        assert r2.is_valid is True
        assert r2.dropped_indices == []


# ===========================================================================
# Additional edge-case tests
# ===========================================================================

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_empty_transcript(self, validator: TranscriptValidator) -> None:
        """An empty list should be valid with no drops."""
        result = validator.validate([], turn_number=1)
        assert result.is_valid is True
        assert result.validated_messages == []

    def test_only_human_messages(self, validator: TranscriptValidator) -> None:
        """Transcript with only human messages — valid."""
        msgs = [_human("hi", idx=0), _human("more", idx=1)]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is True
        assert len(result.validated_messages) == 2

    def test_tool_message_without_tool_call_id_dropped(self, validator: TranscriptValidator) -> None:
        """A tool-role message with no tool_call_id at all is dropped."""
        msg = CanonicalMessage(
            role=CanonicalRole.TOOL, content="no id", source_index=0,
        )
        result = validator.validate([msg], turn_number=1)
        assert 0 in result.dropped_indices

    def test_validation_result_fields(self, validator: TranscriptValidator) -> None:
        """Verify ValidationResult dataclass has expected fields."""
        result = validator.validate([], turn_number=1)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "validated_messages")
        assert hasattr(result, "dropped_indices")
        assert hasattr(result, "dropped_call_ids")
        assert hasattr(result, "signals")
        assert hasattr(result, "repair_actions")

    def test_repair_summary_accumulates(self, validator: TranscriptValidator) -> None:
        """get_repair_summary() accumulates counts across multiple validate() calls."""
        msgs = [
            _ai(idx=0, tool_calls=[_tc("c1")]),
            _tool("r1", call_id="c1", idx=1),
            _tool("r2", call_id="c1", idx=2),  # duplicate
        ]
        validator.validate(msgs, turn_number=1)
        validator.validate(msgs, turn_number=2)
        summary = validator.get_repair_summary()
        assert summary["total_validated"] == 2
        assert summary["duplicates_removed"] >= 2

    def test_extract_type_signature_dict_args(self) -> None:
        """_extract_type_signature builds 'name(type,type)' from dict args."""
        sig = _extract_type_signature("search", {"q": "hello", "n": 5})
        assert sig == "search(str,int)"

    def test_extract_type_signature_non_dict_args(self) -> None:
        """Non-dict args produce 'name()' signature."""
        sig = _extract_type_signature("search", "not a dict")  # type: ignore[arg-type]
        assert sig == "search()"

    def test_ai_without_tool_calls_passes(self, validator: TranscriptValidator) -> None:
        """Plain AI messages (no tool_calls) are always valid."""
        msgs = [
            _ai("just chatting", idx=0),
            _ai("more thoughts", idx=1),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is True
        assert len(result.validated_messages) == 2

    def test_consecutive_ai_messages_valid(self, validator: TranscriptValidator) -> None:
        """Invariant 9: consecutive AI messages without intervening results are valid."""
        msgs = [
            _ai(idx=0, tool_calls=[_tc("c1")]),
            _ai(idx=1, tool_calls=[_tc("c2")]),
            _tool("r1", call_id="c1", idx=2),
            _tool("r2", call_id="c2", idx=3),
        ]
        result = validator.validate(msgs, turn_number=1)
        assert result.is_valid is True


# ===========================================================================
# Parametrized tests for _is_args_valid
# ===========================================================================

class TestIsArgsValidParametrized:
    """Parametrized boundary tests for _is_args_valid."""

    @pytest.mark.parametrize(
        "args, expected",
        [
            ({"key": "value"}, True),
            ({}, True),
            ({"_raw": "x", "extra": 1}, True),
            ({"_raw": "bad"}, False),
            ("string", False),
            (123, False),
            (None, False),
            ([], False),
            ({"a": 1, "b": 2, "c": 3}, True),
        ],
        ids=[
            "normal-dict",
            "empty-dict",
            "raw-with-extra",
            "raw-only",
            "string",
            "int",
            "none",
            "list",
            "multi-key-dict",
        ],
    )
    def test_is_args_valid(self, args: Any, expected: bool) -> None:
        assert _is_args_valid(args) is expected
