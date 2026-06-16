"""Comprehensive unit tests for ToolTransactionLedger."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from tokencircuit.ledger import (
    ToolTransactionLedger,
    _classify_outcome,
    _classify_outcome_cached,
)
from tokencircuit.types import (
    TransactionOutcome,
    TransactionStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_classify_cache():
    """Reset the LRU cache between tests to avoid cross-test pollution."""
    _classify_outcome_cached.cache_clear()
    yield
    _classify_outcome_cached.cache_clear()


@pytest.fixture()
def ledger() -> ToolTransactionLedger:
    """Fresh ledger with default settings."""
    return ToolTransactionLedger()


@pytest.fixture()
def ledger_timeout_1() -> ToolTransactionLedger:
    """Ledger with orphan_timeout_turns=1 for fast timeout tests."""
    return ToolTransactionLedger(orphan_timeout_turns=1)


def _register_call(
    ledger: ToolTransactionLedger,
    call_id: str = "call-1",
    tool_name: str = "read_file",
    turn: int = 0,
):
    """Helper to register a call with sensible defaults."""
    return ledger.register_call(
        call_id=call_id,
        tool_name=tool_name,
        source_message_index=0,
        turn_number=turn,
    )


def _register_result(
    ledger: ToolTransactionLedger,
    call_id: str = "call-1",
    content: str = "file contents here",
    turn: int = 1,
    length: int | None = None,
):
    """Helper to register a result with sensible defaults."""
    return ledger.register_result(
        call_id=call_id,
        result_content_prefix=content,
        result_length=length if length is not None else len(content),
        source_message_index=1,
        turn_number=turn,
    )


# =========================================================================
# 1. register_call creates PENDING transaction
# =========================================================================


class TestRegisterCall:
    """Tests for register_call behaviour."""

    def test_creates_pending_transaction(self, ledger: ToolTransactionLedger):
        """register_call should create a transaction in PENDING status."""
        txn = _register_call(ledger, call_id="c1", turn=0)

        assert txn.status == TransactionStatus.PENDING
        assert txn.call.call_id == "c1"
        assert txn.call.tool_name == "read_file"
        assert txn.result is None
        assert txn.outcome == TransactionOutcome.UNKNOWN

    def test_records_turn_number(self, ledger: ToolTransactionLedger):
        """The call's turn_number should be stored."""
        txn = _register_call(ledger, call_id="c1", turn=5)
        assert txn.call.turn_number == 5

    def test_updates_current_turn(self, ledger: ToolTransactionLedger):
        """register_call should advance the ledger's current_turn."""
        _register_call(ledger, call_id="c1", turn=3)
        assert ledger.current_turn == 3

    def test_current_turn_never_goes_backward(self, ledger: ToolTransactionLedger):
        """current_turn should be monotonically non-decreasing."""
        _register_call(ledger, call_id="c1", turn=5)
        _register_call(ledger, call_id="c2", turn=2)
        assert ledger.current_turn == 5

    # 2. Duplicate call_id is idempotent
    def test_duplicate_call_id_is_idempotent(self, ledger: ToolTransactionLedger):
        """Re-registering the same call_id must return the existing transaction unchanged."""
        txn1 = _register_call(ledger, call_id="dup")
        txn2 = _register_call(ledger, call_id="dup")
        assert txn1 is txn2
        assert len(ledger.get_pending()) == 1

    def test_duplicate_call_id_after_commit(self, ledger: ToolTransactionLedger):
        """Re-registering after COMMITTED should still be idempotent."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_result(ledger, call_id="c1", turn=1)
        txn = _register_call(ledger, call_id="c1", turn=2)
        assert txn.status == TransactionStatus.COMMITTED


# =========================================================================
# 3-6. register_result status transitions
# =========================================================================


class TestRegisterResult:
    """Tests for register_result behaviour."""

    # 3. Commits a PENDING transaction
    def test_commits_pending_transaction(self, ledger: ToolTransactionLedger):
        """Providing a result for a PENDING call should COMMIT it."""
        _register_call(ledger, call_id="c1", turn=0)
        txn, status = _register_result(ledger, call_id="c1", turn=1)

        assert status == TransactionStatus.COMMITTED
        assert txn is not None
        assert txn.status == TransactionStatus.COMMITTED
        assert txn.result is not None
        assert txn.committed_at_turn == 1

    def test_committed_transaction_records_outcome(self, ledger: ToolTransactionLedger):
        """Committed transaction should carry a classified outcome."""
        _register_call(ledger, call_id="c1", turn=0)
        txn, _ = _register_result(ledger, call_id="c1", content="all good", turn=1)
        assert txn.outcome == TransactionOutcome.SUCCESS

    def test_result_prefix_truncated_to_200(self, ledger: ToolTransactionLedger):
        """result_content_prefix stored on the record should be at most 200 chars."""
        long_content = "x" * 500
        _register_call(ledger, call_id="c1", turn=0)
        txn, _ = _register_result(
            ledger, call_id="c1", content=long_content, turn=1, length=500
        )
        assert len(txn.result.result_content_prefix) <= 200

    # 4. Unknown call_id → ORPHANED
    def test_unknown_call_id_returns_orphaned(self, ledger: ToolTransactionLedger):
        """Result for a never-registered call_id must return (None, ORPHANED)."""
        txn, status = _register_result(ledger, call_id="ghost", turn=1)
        assert txn is None
        assert status == TransactionStatus.ORPHANED

    # 5. Already COMMITTED → DUPLICATE
    def test_already_committed_returns_duplicate(self, ledger: ToolTransactionLedger):
        """A second result for the same call_id should return DUPLICATE."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_result(ledger, call_id="c1", turn=1)
        txn, status = _register_result(ledger, call_id="c1", content="retry", turn=2)

        assert status == TransactionStatus.DUPLICATE
        assert txn is not None
        assert txn.status == TransactionStatus.COMMITTED  # original stays

    # 6. ORPHANED transaction stays ORPHANED on late result
    def test_orphaned_stays_orphaned_on_late_result(
        self, ledger_timeout_1: ToolTransactionLedger
    ):
        """A result arriving after the call was orphaned must keep ORPHANED status."""
        _register_call(ledger_timeout_1, call_id="c1", turn=0)
        ledger_timeout_1.advance_turn(1)  # ages it out (timeout=1)
        txn, status = _register_result(ledger_timeout_1, call_id="c1", turn=2)

        assert status == TransactionStatus.ORPHANED
        assert txn is not None
        assert txn.status == TransactionStatus.ORPHANED

    def test_register_result_updates_current_turn(self, ledger: ToolTransactionLedger):
        """register_result should advance the ledger's current_turn."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_result(ledger, call_id="c1", turn=7)
        assert ledger.current_turn == 7


# =========================================================================
# 7-8. advance_turn orphaning
# =========================================================================


class TestAdvanceTurn:
    """Tests for advance_turn ageing logic."""

    # 7. Ages out PENDING → ORPHANED after orphan_timeout_turns
    def test_ages_out_pending_after_timeout(
        self, ledger_timeout_1: ToolTransactionLedger
    ):
        """PENDING calls older than orphan_timeout_turns should become ORPHANED."""
        _register_call(ledger_timeout_1, call_id="c1", turn=0)
        orphaned = ledger_timeout_1.advance_turn(1)  # age = 1 >= timeout(1)

        assert len(orphaned) == 1
        assert orphaned[0].status == TransactionStatus.ORPHANED
        assert orphaned[0].outcome == TransactionOutcome.UNKNOWN

    def test_ages_out_with_default_timeout(self, ledger: ToolTransactionLedger):
        """With default orphan_timeout_turns=3, calls should age out at turn=3."""
        _register_call(ledger, call_id="c1", turn=0)
        assert ledger.advance_turn(1) == []  # age 1 < 3
        assert ledger.advance_turn(2) == []  # age 2 < 3
        orphaned = ledger.advance_turn(3)     # age 3 >= 3
        assert len(orphaned) == 1

    # 8. Doesn't orphan recent calls
    def test_does_not_orphan_recent_calls(self, ledger: ToolTransactionLedger):
        """Calls within the timeout window must stay PENDING."""
        _register_call(ledger, call_id="c1", turn=5)
        orphaned = ledger.advance_turn(6)
        assert orphaned == []
        assert ledger.get_pending()[0].call.call_id == "c1"

    def test_advance_turn_skips_committed(self, ledger: ToolTransactionLedger):
        """advance_turn should not touch COMMITTED transactions."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_result(ledger, call_id="c1", turn=0)
        orphaned = ledger.advance_turn(100)
        assert orphaned == []

    def test_advance_turn_skips_already_orphaned(
        self, ledger_timeout_1: ToolTransactionLedger
    ):
        """Already-ORPHANED transactions should not re-appear in the returned list."""
        _register_call(ledger_timeout_1, call_id="c1", turn=0)
        ledger_timeout_1.advance_turn(1)
        orphaned_again = ledger_timeout_1.advance_turn(5)
        assert orphaned_again == []

    def test_advance_turn_multiple_calls(self, ledger_timeout_1: ToolTransactionLedger):
        """Multiple PENDING calls at different turns should age independently."""
        _register_call(ledger_timeout_1, call_id="c1", turn=0)
        _register_call(ledger_timeout_1, call_id="c2", turn=1)
        orphaned = ledger_timeout_1.advance_turn(1)  # c1 aged (1-0>=1), c2 not (1-1<1)
        assert len(orphaned) == 1
        assert orphaned[0].call.call_id == "c1"

    def test_advance_turn_updates_current_turn(self, ledger: ToolTransactionLedger):
        """advance_turn should update current_turn."""
        ledger.advance_turn(10)
        assert ledger.current_turn == 10


# =========================================================================
# 9. Query helpers: get_pending, get_orphaned, get_committed_since
# =========================================================================


class TestQueryHelpers:
    """Tests for query/filter methods."""

    def test_get_pending_returns_only_pending(self, ledger: ToolTransactionLedger):
        """get_pending must only include PENDING transactions."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_call(ledger, call_id="c2", turn=0)
        _register_result(ledger, call_id="c2", turn=1)

        pending = ledger.get_pending()
        assert len(pending) == 1
        assert pending[0].call.call_id == "c1"

    def test_get_orphaned_returns_only_orphaned(
        self, ledger_timeout_1: ToolTransactionLedger
    ):
        """get_orphaned must only include ORPHANED transactions."""
        _register_call(ledger_timeout_1, call_id="c1", turn=0)
        _register_call(ledger_timeout_1, call_id="c2", turn=0)
        _register_result(ledger_timeout_1, call_id="c2", turn=1)
        ledger_timeout_1.advance_turn(1)

        orphaned = ledger_timeout_1.get_orphaned()
        assert len(orphaned) == 1
        assert orphaned[0].call.call_id == "c1"

    def test_get_committed_since_filters_by_turn(self, ledger: ToolTransactionLedger):
        """get_committed_since(turn) should only return commits at or after that turn."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_result(ledger, call_id="c1", turn=1)
        _register_call(ledger, call_id="c2", turn=2)
        _register_result(ledger, call_id="c2", turn=3)

        since_2 = ledger.get_committed_since(2)
        assert len(since_2) == 1
        assert since_2[0].call.call_id == "c2"

    def test_get_committed_since_includes_boundary(self, ledger: ToolTransactionLedger):
        """Turn boundary should be inclusive (>=)."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_result(ledger, call_id="c1", turn=5)
        assert len(ledger.get_committed_since(5)) == 1
        assert len(ledger.get_committed_since(6)) == 0

    def test_get_committed_since_empty_ledger(self, ledger: ToolTransactionLedger):
        """Empty ledger returns empty list."""
        assert ledger.get_committed_since(0) == []

    def test_get_transaction_by_call_id(self, ledger: ToolTransactionLedger):
        """get_transaction should return the transaction or None."""
        _register_call(ledger, call_id="c1", turn=0)
        assert ledger.get_transaction("c1") is not None
        assert ledger.get_transaction("nonexistent") is None

    def test_total_committed_property(self, ledger: ToolTransactionLedger):
        """total_committed property should count only COMMITTED."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_call(ledger, call_id="c2", turn=0)
        _register_result(ledger, call_id="c1", turn=1)
        assert ledger.total_committed == 1

    def test_total_orphaned_property(self, ledger_timeout_1: ToolTransactionLedger):
        """total_orphaned property should count only ORPHANED."""
        _register_call(ledger_timeout_1, call_id="c1", turn=0)
        ledger_timeout_1.advance_turn(1)
        assert ledger_timeout_1.total_orphaned == 1


# =========================================================================
# 10. get_consecutive_outcomes
# =========================================================================


class TestConsecutiveOutcomes:
    """Tests for get_consecutive_outcomes streak counter."""

    def test_counts_streak_from_most_recent(self, ledger: ToolTransactionLedger):
        """Should count consecutive most-recent committed txns with matching outcome."""
        # 3 empty results in a row
        for i in range(3):
            cid = f"c{i}"
            _register_call(ledger, call_id=cid, turn=i)
            _register_result(ledger, call_id=cid, content="", turn=i, length=0)

        assert ledger.get_consecutive_outcomes(TransactionOutcome.EMPTY) == 3

    def test_streak_breaks_on_different_outcome(self, ledger: ToolTransactionLedger):
        """Streak must break when a different outcome is encountered."""
        # Turn 0: success
        _register_call(ledger, call_id="c0", turn=0)
        _register_result(ledger, call_id="c0", content="good data", turn=0)
        # Turn 1-3: empty
        for i in range(1, 4):
            _register_call(ledger, call_id=f"c{i}", turn=i)
            _register_result(ledger, call_id=f"c{i}", content="", turn=i, length=0)

        assert ledger.get_consecutive_outcomes(TransactionOutcome.EMPTY) == 3
        assert ledger.get_consecutive_outcomes(TransactionOutcome.SUCCESS) == 0

    def test_streak_zero_when_no_match(self, ledger: ToolTransactionLedger):
        """Returns 0 when the most recent committed doesn't match."""
        _register_call(ledger, call_id="c0", turn=0)
        _register_result(ledger, call_id="c0", content="all good", turn=0)
        assert ledger.get_consecutive_outcomes(TransactionOutcome.EMPTY) == 0

    def test_streak_empty_ledger(self, ledger: ToolTransactionLedger):
        """Empty ledger should return 0."""
        assert ledger.get_consecutive_outcomes(TransactionOutcome.SUCCESS) == 0

    def test_streak_ignores_pending_and_orphaned(
        self, ledger_timeout_1: ToolTransactionLedger
    ):
        """Only COMMITTED transactions should count toward the streak."""
        _register_call(ledger_timeout_1, call_id="c_pending", turn=0)
        # Don't commit c_pending
        _register_call(ledger_timeout_1, call_id="c_committed", turn=0)
        _register_result(
            ledger_timeout_1, call_id="c_committed", content="good", turn=0
        )
        assert (
            ledger_timeout_1.get_consecutive_outcomes(TransactionOutcome.SUCCESS) == 1
        )

    def test_streak_respects_limit(self, ledger: ToolTransactionLedger):
        """The limit parameter should cap how far back we look."""
        for i in range(15):
            _register_call(ledger, call_id=f"c{i}", turn=i)
            _register_result(ledger, call_id=f"c{i}", content="", turn=i, length=0)

        # Default limit=10 should cap at 10
        assert ledger.get_consecutive_outcomes(TransactionOutcome.EMPTY) == 10
        # Explicit smaller limit
        assert (
            ledger.get_consecutive_outcomes(TransactionOutcome.EMPTY, limit=5) == 5
        )


# =========================================================================
# 11. Outcome classification via _classify_outcome
# =========================================================================


class TestOutcomeClassification:
    """Tests for _classify_outcome function."""

    @pytest.mark.parametrize(
        "content, length",
        [
            ("", 0),
            ("   ", 3),
            ("\n\t", 2),
        ],
        ids=["empty-string", "whitespace-only", "newline-tab"],
    )
    def test_empty_content_classified_as_empty(self, content: str, length: int):
        """Empty or whitespace-only content should be classified EMPTY."""
        assert _classify_outcome(content, length) == TransactionOutcome.EMPTY

    @pytest.mark.parametrize(
        "content",
        [
            "timeout while waiting for response",
            "Request timed out after 30s",
            "rate limit exceeded, retry after 60s",
            "Error 429: Too Many Requests",
            "connection error: refused",
            "network error occurred",
            "temporarily unavailable",
            "503 Service Unavailable",
        ],
        ids=[
            "timeout",
            "timed-out",
            "rate-limit",
            "429",
            "connection-refused",
            "network-error",
            "temporarily",
            "503",
        ],
    )
    def test_transient_error_patterns(self, content: str):
        """Content matching transient error patterns should be TRANSIENT_ERROR."""
        assert _classify_outcome(content, len(content)) == TransactionOutcome.TRANSIENT_ERROR

    @pytest.mark.parametrize(
        "content",
        [
            "Error: not found",
            "404 page does not exist",
            "permission denied",
            "403 Forbidden",
            "401 unauthorized access",
            "invalid argument supplied",
            "malformed request body",
            "bad request 400",
            "deprecated API endpoint",
            "feature removed in v2",
            "unsupported operation",
        ],
        ids=[
            "not-found",
            "404",
            "permission-denied",
            "403",
            "401",
            "invalid",
            "malformed",
            "400",
            "deprecated",
            "removed",
            "unsupported",
        ],
    )
    def test_permanent_error_patterns(self, content: str):
        """Content matching permanent error patterns should be PERMANENT_ERROR."""
        assert _classify_outcome(content, len(content)) == TransactionOutcome.PERMANENT_ERROR

    @pytest.mark.parametrize(
        "content",
        [
            "Error: something went wrong",
            '{"error": "some message"}',
            '{"message": "an error occurred"}',
        ],
        ids=[
            "generic-error-prefix",
            "json-error-field",
            "json-message-error",
        ],
    )
    def test_generic_error_patterns(self, content: str):
        """Generic error patterns (that aren't transient) should be PERMANENT_ERROR."""
        assert _classify_outcome(content, len(content)) == TransactionOutcome.PERMANENT_ERROR

    @pytest.mark.parametrize(
        "content",
        [
            "The file contents are as follows...",
            "Here is the result: 42",
            '{"data": [1, 2, 3]}',
            "Operation completed successfully.",
        ],
        ids=["file-read", "calculation", "json-data", "success-message"],
    )
    def test_success_content(self, content: str):
        """Normal content without error patterns should be SUCCESS."""
        assert _classify_outcome(content, len(content)) == TransactionOutcome.SUCCESS

    def test_transient_takes_priority_over_permanent(self):
        """If content matches both transient and permanent patterns, transient wins."""
        # Contains both "timeout" (transient) and "not found" (permanent)
        content = "timeout while checking not found resource"
        assert _classify_outcome(content, len(content)) == TransactionOutcome.TRANSIENT_ERROR


# =========================================================================
# 12. _classify_outcome caching behaviour
# =========================================================================


class TestClassifyOutcomeCaching:
    """Tests for hash-based LRU caching in _classify_outcome."""

    def test_small_content_uses_small_key(self):
        """Content under 100 chars should use 'small' as the hash key (fast path)."""
        # We test this indirectly: calling twice with the same content should hit cache.
        _classify_outcome_cached.cache_clear()
        result1 = _classify_outcome("hello", 5)
        info_after_first = _classify_outcome_cached.cache_info()
        result2 = _classify_outcome("hello", 5)
        info_after_second = _classify_outcome_cached.cache_info()

        assert result1 == result2
        assert info_after_second.hits == info_after_first.hits + 1

    def test_large_content_uses_sha256_key(self):
        """Content >= 100 chars should be hashed via SHA-256."""
        _classify_outcome_cached.cache_clear()
        large = "a" * 200
        result1 = _classify_outcome(large, 200)
        result2 = _classify_outcome(large, 200)
        info = _classify_outcome_cached.cache_info()

        assert result1 == result2 == TransactionOutcome.SUCCESS
        assert info.hits >= 1

    def test_different_content_same_length_no_false_cache_hit(self):
        """Two different large strings should not collide in the cache."""
        _classify_outcome_cached.cache_clear()
        a = "a" * 200
        b = "b" * 200
        r1 = _classify_outcome(a, 200)
        r2 = _classify_outcome(b, 200)
        # Both SUCCESS but should be separate cache entries
        info = _classify_outcome_cached.cache_info()
        assert info.misses >= 2

    def test_cache_maxsize_eviction(self):
        """After maxsize (100) entries, oldest should be evicted."""
        _classify_outcome_cached.cache_clear()
        # Fill cache with 101 unique small entries
        for i in range(101):
            _classify_outcome(f"unique-{i}", len(f"unique-{i}"))

        info = _classify_outcome_cached.cache_info()
        # The cache should have had all 101 misses; size should be 100
        assert info.misses == 101
        assert info.currsize == 100


# =========================================================================
# 13. max_pending warning
# =========================================================================


class TestMaxPendingWarning:
    """Tests for the max_pending overflow warning."""

    def test_warning_logged_when_exceeding_max_pending(self, caplog):
        """A warning should be emitted when pending count exceeds max_pending."""
        ledger = ToolTransactionLedger(max_pending=2)
        with caplog.at_level(logging.WARNING, logger="tokencircuit.ledger"):
            _register_call(ledger, call_id="c1", turn=0)
            _register_call(ledger, call_id="c2", turn=0)
            _register_call(ledger, call_id="c3", turn=0)  # 3 > 2 → warning

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert "pending" in warning_records[0].message.lower()
        assert "max_pending" in warning_records[0].message.lower()

    def test_no_warning_at_or_below_max_pending(self, caplog):
        """No warning when pending count is at or below max_pending."""
        ledger = ToolTransactionLedger(max_pending=5)
        with caplog.at_level(logging.WARNING, logger="tokencircuit.ledger"):
            for i in range(5):
                _register_call(ledger, call_id=f"c{i}", turn=0)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0


# =========================================================================
# 14. reset() clears everything
# =========================================================================


class TestReset:
    """Tests for the reset method."""

    def test_reset_clears_transactions(self, ledger: ToolTransactionLedger):
        """reset() should remove all transactions."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_call(ledger, call_id="c2", turn=1)
        _register_result(ledger, call_id="c1", turn=1)

        ledger.reset()

        assert ledger.get_pending() == []
        assert ledger.get_orphaned() == []
        assert ledger.get_committed_since(0) == []
        assert ledger.get_transaction("c1") is None

    def test_reset_zeros_current_turn(self, ledger: ToolTransactionLedger):
        """reset() should set current_turn back to 0."""
        _register_call(ledger, call_id="c1", turn=42)
        ledger.reset()
        assert ledger.current_turn == 0

    def test_reset_counters_zero(self, ledger: ToolTransactionLedger):
        """total_committed and total_orphaned should be 0 after reset."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_result(ledger, call_id="c1", turn=1)
        ledger.reset()
        assert ledger.total_committed == 0
        assert ledger.total_orphaned == 0

    def test_ledger_usable_after_reset(self, ledger: ToolTransactionLedger):
        """Ledger should be fully functional after reset."""
        _register_call(ledger, call_id="c1", turn=0)
        ledger.reset()

        txn = _register_call(ledger, call_id="c_new", turn=0)
        assert txn.status == TransactionStatus.PENDING
        assert len(ledger.get_pending()) == 1


# =========================================================================
# 15. orphan_timeout_turns validation
# =========================================================================


class TestOrphanTimeoutValidation:
    """Tests for constructor validation of orphan_timeout_turns."""

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_orphan_timeout_below_1_raises_value_error(self, value: int):
        """orphan_timeout_turns < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="orphan_timeout_turns must be >= 1"):
            ToolTransactionLedger(orphan_timeout_turns=value)

    def test_orphan_timeout_exactly_1_is_valid(self):
        """orphan_timeout_turns = 1 is the minimum valid value."""
        ledger = ToolTransactionLedger(orphan_timeout_turns=1)
        assert ledger._orphan_timeout == 1

    def test_large_orphan_timeout_works(self):
        """Very large orphan_timeout_turns should work."""
        ledger = ToolTransactionLedger(orphan_timeout_turns=1000)
        assert ledger._orphan_timeout == 1000


# =========================================================================
# 16. Edge cases
# =========================================================================


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_very_long_content_classification(self):
        """Very long content should be classified correctly (only first 1000 chars hashed)."""
        # 10K of harmless text → SUCCESS
        long_content = "x" * 10_000
        assert _classify_outcome(long_content, 10_000) == TransactionOutcome.SUCCESS

    def test_very_long_content_with_error_at_start(self):
        """Error at the start of long content should be detected."""
        content = "Error: something broke" + "x" * 10_000
        assert _classify_outcome(content, len(content)) == TransactionOutcome.PERMANENT_ERROR

    def test_very_long_content_with_transient_error_at_start(self):
        """Transient error at the start of long content should be detected."""
        content = "timeout occurred" + "x" * 10_000
        assert _classify_outcome(content, len(content)) == TransactionOutcome.TRANSIENT_ERROR

    def test_content_with_mixed_error_patterns(self):
        """Content with both transient and permanent patterns; transient first check wins."""
        content = "connection error and also 404 not found"
        result = _classify_outcome(content, len(content))
        # "connection error" is transient → checked first → TRANSIENT_ERROR
        assert result == TransactionOutcome.TRANSIENT_ERROR

    def test_length_zero_content_nonempty_string(self):
        """Passing length=0 but non-empty content → should be EMPTY (length check)."""
        assert _classify_outcome("has text", 0) == TransactionOutcome.EMPTY

    def test_numeric_only_content(self):
        """Purely numeric content (no error patterns) → SUCCESS."""
        assert _classify_outcome("12345", 5) == TransactionOutcome.SUCCESS

    def test_url_like_content(self):
        """URL content without error patterns → SUCCESS."""
        content = "https://example.com/api/v2/resource?id=42"
        assert _classify_outcome(content, len(content)) == TransactionOutcome.SUCCESS

    def test_json_success_response(self):
        """JSON with data field (no error field) → SUCCESS."""
        content = '{"status": "ok", "data": [1, 2, 3]}'
        assert _classify_outcome(content, len(content)) == TransactionOutcome.SUCCESS

    def test_multiple_calls_interleaved_results(self, ledger: ToolTransactionLedger):
        """Multiple calls committed out of order should all work correctly."""
        _register_call(ledger, call_id="c1", turn=0)
        _register_call(ledger, call_id="c2", turn=0)
        _register_call(ledger, call_id="c3", turn=0)

        # Commit in reverse order
        _, s3 = _register_result(ledger, call_id="c3", turn=1)
        _, s1 = _register_result(ledger, call_id="c1", turn=1)
        _, s2 = _register_result(ledger, call_id="c2", turn=1)

        assert s1 == s2 == s3 == TransactionStatus.COMMITTED
        assert ledger.total_committed == 3
        assert ledger.get_pending() == []

    def test_empty_string_call_id(self, ledger: ToolTransactionLedger):
        """Empty string call_id should work (no special treatment)."""
        txn = _register_call(ledger, call_id="", turn=0)
        assert txn.call.call_id == ""

        result_txn, status = _register_result(ledger, call_id="", turn=1)
        assert status == TransactionStatus.COMMITTED

    def test_special_chars_in_call_id(self, ledger: ToolTransactionLedger):
        """Special characters in call_id should work fine."""
        call_id = "call/with:special-chars_and.dots#hash"
        _register_call(ledger, call_id=call_id, turn=0)
        txn, status = _register_result(ledger, call_id=call_id, turn=1)
        assert status == TransactionStatus.COMMITTED

    def test_high_turn_numbers(self, ledger: ToolTransactionLedger):
        """Very large turn numbers should work correctly."""
        _register_call(ledger, call_id="c1", turn=999_999)
        _register_result(ledger, call_id="c1", turn=1_000_000)
        assert ledger.current_turn == 1_000_000

    def test_same_turn_for_call_and_result(self, ledger: ToolTransactionLedger):
        """Call and result in the same turn should commit successfully."""
        _register_call(ledger, call_id="c1", turn=5)
        txn, status = _register_result(ledger, call_id="c1", turn=5)
        assert status == TransactionStatus.COMMITTED
        assert txn.committed_at_turn == 5


# =========================================================================
# Integration-style: full lifecycle scenarios
# =========================================================================


class TestFullLifecycle:
    """End-to-end lifecycle tests combining multiple operations."""

    def test_typical_multi_turn_session(self, ledger: ToolTransactionLedger):
        """Simulate a realistic multi-turn agent session."""
        # Turn 0: Agent makes two tool calls
        _register_call(ledger, call_id="t0-c1", tool_name="search", turn=0)
        _register_call(ledger, call_id="t0-c2", tool_name="read", turn=0)
        assert len(ledger.get_pending()) == 2

        # Turn 1: Results come back
        ledger.advance_turn(1)
        _register_result(ledger, call_id="t0-c1", content="found it", turn=1)
        _register_result(ledger, call_id="t0-c2", content="file data", turn=1)
        assert ledger.total_committed == 2
        assert len(ledger.get_pending()) == 0

        # Turn 1: Agent makes another call
        _register_call(ledger, call_id="t1-c1", tool_name="write", turn=1)

        # Turn 2: Result comes back
        ledger.advance_turn(2)
        _register_result(ledger, call_id="t1-c1", content="ok", turn=2)
        assert ledger.total_committed == 3

    def test_orphan_then_new_calls_after_reset_scenario(
        self, ledger_timeout_1: ToolTransactionLedger
    ):
        """Test orphaning followed by reset and fresh usage."""
        _register_call(ledger_timeout_1, call_id="old", turn=0)
        ledger_timeout_1.advance_turn(1)
        assert ledger_timeout_1.total_orphaned == 1

        ledger_timeout_1.reset()
        assert ledger_timeout_1.total_orphaned == 0

        _register_call(ledger_timeout_1, call_id="new", turn=0)
        _register_result(ledger_timeout_1, call_id="new", turn=0)
        assert ledger_timeout_1.total_committed == 1
