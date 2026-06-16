"""ToolTransactionLedger — tracks tool call/result lifecycle as immutable transactions."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

from .types import (
    ToolCallRecord,
    ToolResultRecord,
    ToolTransaction,
    TransactionOutcome,
    TransactionStatus,
)

logger = logging.getLogger("tokencircuit.ledger")

# Patterns for outcome classification
_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Match error indicators near the start of the message
    re.compile(r"^\s*(?:error|exception|fail(?:ed|ure)|traceback)[\s:]", re.IGNORECASE),
    # Match common JSON error structures
    re.compile(r"\"error\"\s*:\s*(?:\"[^\"]+\"|{)", re.IGNORECASE),
    re.compile(r"\"message\"\s*:\s*\"[^\"]*(?:error|fail|exception)[^\"]*\"", re.IGNORECASE),
    re.compile(r"(?i)\b(4\d{2}|5\d{2})\b.*\b(status|code|error)\b"),
)

_TRANSIENT_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(timeout|timed?\s*out)\b"),
    re.compile(r"(?i)\b(rate.?limit|throttl|retry|429|503)\b"),
    re.compile(r"(?i)\b(connection|network)\s*(error|refused|reset)\b"),
    re.compile(r"(?i)\btemporar(y|ily)\b"),
)

_PERMANENT_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(not.?found|404|does\s*not\s*exist)\b"),
    re.compile(r"(?i)\b(permission|forbidden|403|unauthorized|401)\b"),
    re.compile(r"(?i)\b(invalid|malformed|bad\s*request|400)\b"),
    re.compile(r"(?i)\b(deprecated|removed|unsupported)\b"),
)


@lru_cache(maxsize=100)
def _classify_outcome_cached(content_hash: str, content: str, length: int) -> TransactionOutcome:
    """Cached version of outcome classification."""
    if length == 0 or not content.strip():
        return TransactionOutcome.EMPTY

    # Check for transient errors first
    for pattern in _TRANSIENT_ERROR_PATTERNS:
        if pattern.search(content):
            return TransactionOutcome.TRANSIENT_ERROR

    # Check permanent errors
    for pattern in _PERMANENT_ERROR_PATTERNS:
        if pattern.search(content):
            return TransactionOutcome.PERMANENT_ERROR

    # Generic error check
    for pattern in _ERROR_PATTERNS:
        if pattern.search(content):
            return TransactionOutcome.PERMANENT_ERROR

    return TransactionOutcome.SUCCESS


def _classify_outcome(content: str, length: int) -> TransactionOutcome:
    """
    Classify tool result as success, empty, or error based on content.
    Uses hash-based caching to avoid expensive regex on repeats.
    """
    if length < 100:
        # Small contents are fast, don't bother hashing, use string as cache key
        return _classify_outcome_cached(content, content, length)

    content_hash = hashlib.sha256(content[:1000].encode()).hexdigest()
    return _classify_outcome_cached(content_hash, content, length)


class ToolTransactionLedger:
    """
    Tracks the lifecycle of tool call → tool result transactions.

    Invariants:
    - Every tool_call generates a PENDING transaction.
    - A tool_result COMMITS the transaction (pairing by call_id).
    - A tool_result with no matching PENDING call_id is ORPHANED.
    - A second result for an already COMMITTED call_id is DUPLICATE.
    - PENDING transactions exceeding orphan_timeout_turns are ORPHANED.
    """

    def __init__(
        self,
        *,
        orphan_timeout_turns: int = 3,
        max_pending: int = 20,
    ) -> None:
        if orphan_timeout_turns < 1:
            raise ValueError("orphan_timeout_turns must be >= 1")
        self._orphan_timeout = orphan_timeout_turns
        self._max_pending = max_pending
        # OrderedDict preserves insertion order for deterministic iteration
        self._transactions: OrderedDict[str, ToolTransaction] = OrderedDict()
        self._current_turn: int = 0

        # O(1) Trackers
        self._pending_count: int = 0
        self._committed_count: int = 0
        self._orphaned_count: int = 0
        self._recent_outcomes: list[TransactionOutcome] = []

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

        If the call_id already exists:
        - If PENDING/COMMITTED: log warning and return existing (idempotent).
        - This handles cases where the same AI message is re-processed.
        """
        existing = self._transactions.get(call_id)
        if existing is not None:
            logger.debug(
                "Ledger: call_id %r already registered (status=%s), skipping",
                call_id,
                existing.status.value,
            )
            return existing

        record = ToolCallRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments_hash=arguments_hash,
            arguments_type_signature=arguments_type_signature,
            source_message_index=source_message_index,
            turn_number=turn_number,
        )
        txn = ToolTransaction(call=record, status=TransactionStatus.PENDING)
        self._transactions[call_id] = txn
        self._pending_count += 1
        self._current_turn = max(self._current_turn, turn_number)

        if self._pending_count > self._max_pending:
            logger.warning(
                "Ledger: %d pending transactions exceeds max_pending=%d",
                self._pending_count,
                self._max_pending,
            )

        return txn

    def register_result(
        self,
        call_id: str,
        result_hash: str,
        result_content_prefix: str,
        result_length: int,
        source_message_index: int,
        turn_number: int,
    ) -> tuple[Optional[ToolTransaction], TransactionStatus]:
        """
        Register a tool result. Attempts to COMMIT the matching transaction.

        Returns:
            (updated_transaction, status)
            - If call_id found and PENDING: (committed_txn, COMMITTED)
            - If call_id found and already COMMITTED: (existing_txn, DUPLICATE)
            - If call_id not found: (None, ORPHANED)
        """
        self._current_turn = max(self._current_turn, turn_number)
        outcome = _classify_outcome(result_content_prefix, result_length)
        result_record = ToolResultRecord(
            call_id=call_id,
            result_hash=result_hash,
            result_content_prefix=result_content_prefix[:200],
            result_length=result_length,
            source_message_index=source_message_index,
            turn_number=turn_number,
            outcome=outcome,
        )

        existing = self._transactions.get(call_id)

        if existing is None:
            # No matching call — orphaned result
            logger.debug("Ledger: result for unknown call_id %r → ORPHANED", call_id)
            return None, TransactionStatus.ORPHANED

        if existing.status == TransactionStatus.COMMITTED:
            # Already committed — duplicate result
            logger.debug("Ledger: duplicate result for call_id %r", call_id)
            return existing, TransactionStatus.DUPLICATE

        if existing.status == TransactionStatus.ORPHANED:
            # Was timed out but result arrived late — still treat as orphaned
            logger.debug(
                "Ledger: late result for orphaned call_id %r", call_id
            )
            return existing, TransactionStatus.ORPHANED

        # PENDING → COMMITTED
        committed = ToolTransaction(
            call=existing.call,
            result=result_record,
            status=TransactionStatus.COMMITTED,
            outcome=outcome,
            committed_at_turn=turn_number,
        )
        self._transactions[call_id] = committed
        self._pending_count -= 1
        self._committed_count += 1
        self._recent_outcomes.append(outcome)
        return committed, TransactionStatus.COMMITTED

    def advance_turn(self, turn_number: int) -> list[ToolTransaction]:
        """
        Called at the start of each turn. Ages out stale PENDING transactions.

        Returns:
            List of transactions transitioned to ORPHANED.
        """
        self._current_turn = max(self._current_turn, turn_number)
        newly_orphaned: list[ToolTransaction] = []

        for call_id, txn in list(self._transactions.items()):
            if txn.status != TransactionStatus.PENDING:
                continue
            age = self._current_turn - txn.call.turn_number
            if age >= self._orphan_timeout:
                orphaned = ToolTransaction(
                    call=txn.call,
                    result=None,
                    status=TransactionStatus.ORPHANED,
                    outcome=TransactionOutcome.UNKNOWN,
                    committed_at_turn=None,
                )
                self._transactions[call_id] = orphaned
                self._pending_count -= 1
                self._orphaned_count += 1
                newly_orphaned.append(orphaned)
                logger.debug(
                    "Ledger: aged out call_id %r after %d turns → ORPHANED",
                    call_id,
                    age,
                )

        return newly_orphaned

    def get_pending(self) -> list[ToolTransaction]:
        """Return all PENDING transactions."""
        return [t for t in self._transactions.values() if t.status == TransactionStatus.PENDING]

    def get_orphaned(self) -> list[ToolTransaction]:
        """Return all ORPHANED transactions."""
        return [t for t in self._transactions.values() if t.status == TransactionStatus.ORPHANED]

    def get_committed_since(self, turn: int) -> list[ToolTransaction]:
        """Return transactions committed at or after the given turn."""
        return [
            t
            for t in self._transactions.values()
            if t.status == TransactionStatus.COMMITTED and (t.committed_at_turn or 0) >= turn
        ]

    def get_transaction(self, call_id: str) -> Optional[ToolTransaction]:
        """Look up a transaction by call_id."""
        return self._transactions.get(call_id)

    @property
    def total_committed(self) -> int:
        """Total committed transactions."""
        return self._committed_count

    @property
    def total_orphaned(self) -> int:
        """Total orphaned transactions."""
        return self._orphaned_count

    @property
    def current_turn(self) -> int:
        return self._current_turn

    def get_consecutive_outcomes(self, outcome: TransactionOutcome, *, limit: int = 10) -> int:
        """
        Count consecutive most-recent committed transactions with the given outcome.
        Used by InterventionEngine to detect repeated failures.
        """
        count = 0
        # Iterate backwards over the most recent outcomes
        for o in reversed(self._recent_outcomes[-limit:]):
            if o == outcome:
                count += 1
            else:
                break
        return count

    def reset(self) -> None:
        """Clear all tracked transactions."""
        self._transactions.clear()
        self._current_turn = 0
        self._pending_count = 0
        self._committed_count = 0
        self._orphaned_count = 0
        self._recent_outcomes.clear()
