"""
TranscriptValidator — enforces immutable transaction model on message transcripts.

THE 10 INVARIANTS:
==================
1. CALL-BEFORE-RESULT: Every ToolMessage (role=tool) MUST reference a tool_call_id
   that exists in a PRIOR AI message's tool_calls list.

2. RESULT-AFTER-CALL: ToolMessages must appear AFTER the AI message containing
   their corresponding tool_call.

3. NO-ORPHAN-RESULTS: A ToolMessage whose tool_call_id does not match ANY
   registered tool_call is dropped (orphaned result).

4. NO-DUPLICATE-RESULTS: Only ONE ToolMessage per tool_call_id is permitted.
   Subsequent duplicates are dropped.

5. ATOMIC-CALL-DROP: If an AI message with tool_calls is dropped, ALL matching
   ToolMessages for those call_ids MUST also be dropped atomically.

6. ATOMIC-RESULT-DROP: If a ToolMessage is dropped, the corresponding tool_call
   in the AI message is NOT dropped (the call was issued; we just lost the result).

7. MALFORMED-ARGS-DROP: If a tool_call has un-parseable arguments (not valid JSON,
   not a dict), the ENTIRE AI message's tool_calls are dropped as a unit,
   AND all matching ToolMessages are dropped.

8. CALL-ID-REQUIRED: Tool calls without an 'id' field are considered malformed
   and trigger invariant 7.

9. CONSECUTIVE-AI-MERGE: Consecutive AI messages without intervening tool results
   are valid (LangGraph pattern for multi-step reasoning). Not dropped.

10. SYSTEM-MESSAGES-PASSTHROUGH: System messages are never dropped or modified
    by the validator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .ledger import ToolTransactionLedger
from .types import (
    CanonicalMessage,
    CanonicalRole,
    SignalType,
)

logger = logging.getLogger("tokencircuit.validator")


@dataclass
class ValidationResult:
    """Output of TranscriptValidator.validate()."""

    is_valid: bool
    validated_messages: list[CanonicalMessage]
    dropped_indices: list[int] = field(default_factory=list)
    dropped_call_ids: list[str] = field(default_factory=list)
    signals: list[SignalType] = field(default_factory=list)
    repair_actions: list[str] = field(default_factory=list)


def _is_args_valid(args: Any) -> bool:
    """
    Check if tool_call arguments are valid.
    Must be a dict. Raw strings that aren't valid JSON dicts are invalid.
    """
    if isinstance(args, dict):
        # Check if it's a raw-fallback marker
        return "_raw" not in args or len(args) > 1
    return False


class TranscriptValidator:
    """
    Validates and repairs message transcripts before LLM consumption.

    Core Principle: Tool calls are IMMUTABLE TRANSACTIONS.
    Dropping a call drops all its results. Dropping a result does NOT drop the call.
    Malformed args cause the entire AI message's tool_calls to be treated as void.
    """

    def __init__(
        self,
        *,
        ledger: ToolTransactionLedger,
        auto_repair: bool = True,
        max_orphan_tolerance: int = 2,
    ) -> None:
        self._ledger = ledger
        self._auto_repair = auto_repair
        self._max_orphan_tolerance = max_orphan_tolerance

    def validate(
        self,
        messages: list[CanonicalMessage],
        turn_number: int,
    ) -> ValidationResult:
        """
        Validate the canonical message transcript.
        Stateless: Rebuilds ledger from the full transcript on every call.
        """
        self._ledger.reset()

        dropped_indices: set[int] = set()
        dropped_call_ids: list[str] = []
        repair_actions: list[str] = []
        signals: list[SignalType] = []

        # Maps call_id -> source AI message index
        call_id_to_ai_index: dict[str, int] = {}
        malformed_ai_indices: set[int] = set()
        malformed_call_ids: set[str] = set()

        # Simulated turn tracking to preserve relative age for ledger timeouts
        # We start turns at 1 and increment on every AI message
        local_turn = 1
        ai_turn_map: dict[int, int] = {}

        # PASS 1: Identify all tool_calls and check structural validity
        for msg in messages:
            if msg.role != CanonicalRole.AI:
                continue

            ai_turn_map[msg.source_index] = local_turn
            local_turn += 1

            if not msg.tool_calls:
                continue

            has_malformed = False
            for tc in msg.tool_calls:
                call_id = tc.get("id", "")
                args = tc.get("args", {})
                if not call_id or not _is_args_valid(args):
                    has_malformed = True
                    break

            if has_malformed:
                malformed_ai_indices.add(msg.source_index)
                for tc in msg.tool_calls:
                    cid = tc.get("id", "")
                    if cid:
                        malformed_call_ids.add(cid)
                repair_actions.append(f"MALFORMED_ARGS: index {msg.source_index}")
            else:
                for tc in msg.tool_calls:
                    call_id = tc.get("id", "")
                    if call_id:
                        call_id_to_ai_index[call_id] = msg.source_index

        # Set ledger's current turn to match the simulated history
        self._ledger.advance_turn(local_turn)

        # PASS 2: Register calls with ledger, validate results
        seen_result_ids: set[str] = set()
        orphan_count: int = 0

        for msg in messages:
            if msg.role == CanonicalRole.TOOL and not msg.tool_call_id:
                dropped_indices.add(msg.source_index)
                repair_actions.append(f"EMPTY_TOOL_CALL_ID: index {msg.source_index}")
                continue

            if msg.role == CanonicalRole.AI and msg.tool_calls:
                if msg.source_index not in malformed_ai_indices:
                    msg_turn = ai_turn_map.get(msg.source_index, local_turn)
                    for tc in msg.tool_calls:
                        call_id = tc.get("id", "")
                        if call_id:
                            self._ledger.register_call(
                                call_id=call_id,
                                tool_name=tc.get("name", "unknown"),
                                source_message_index=msg.source_index,
                                turn_number=msg_turn,
                            )

            if msg.role == CanonicalRole.TOOL and msg.tool_call_id:
                tcid = msg.tool_call_id
                if tcid in malformed_call_ids:
                    dropped_indices.add(msg.source_index)
                    dropped_call_ids.append(tcid)
                    continue

                if tcid in seen_result_ids:
                    dropped_indices.add(msg.source_index)
                    dropped_call_ids.append(tcid)
                    continue
                seen_result_ids.add(tcid)

                if tcid not in call_id_to_ai_index:
                    if self._auto_repair:
                        dropped_indices.add(msg.source_index)
                        dropped_call_ids.append(tcid)
                        orphan_count += 1
                    continue

                ai_index = call_id_to_ai_index.get(tcid)
                if ai_index is not None and msg.source_index <= ai_index:
                    if self._auto_repair:
                        dropped_indices.add(msg.source_index)
                        dropped_call_ids.append(tcid)
                    continue

                # Register result using the turn of the original call
                call_turn = ai_turn_map.get(ai_index, local_turn)
                self._ledger.register_result(
                    call_id=tcid,
                    result_content_prefix=msg.content[:200],
                    result_length=len(msg.content),
                    source_message_index=msg.source_index,
                    turn_number=call_turn,
                )

        # Build output and apply Invariant 11 (No dangling calls)
        validated: list[CanonicalMessage] = []
        for msg in messages:
            if msg.source_index in dropped_indices:
                continue

            if msg.role == CanonicalRole.SYSTEM:
                validated.append(msg)
                continue

            if msg.source_index in malformed_ai_indices and msg.role == CanonicalRole.AI:
                validated.append(CanonicalMessage(
                    role=msg.role, content=msg.content, tool_calls=[],
                    tool_call_id=msg.tool_call_id, source_index=msg.source_index, name=msg.name
                ))
                continue

            validated.append(msg)

        if self._auto_repair and validated:
            resolved_call_ids = {m.tool_call_id for m in validated if m.role == CanonicalRole.TOOL and m.tool_call_id}
            last_ai_idx = -1
            for i, m in enumerate(validated):
                if m.role == CanonicalRole.AI and m.tool_calls:
                    last_ai_idx = i

            repaired = []
            for i, msg in enumerate(validated):
                if msg.role == CanonicalRole.AI and msg.tool_calls and i != last_ai_idx:
                    resolved = [tc for tc in msg.tool_calls if tc.get("id") in resolved_call_ids]
                    if len(resolved) != len(msg.tool_calls):
                        repair_actions.append(f"DANGLING_CALL: index {msg.source_index}")
                        msg = CanonicalMessage(
                            role=msg.role, content=msg.content, tool_calls=resolved,
                            tool_call_id=msg.tool_call_id, source_index=msg.source_index, name=msg.name
                        )
                repaired.append(msg)
            validated = repaired

        if orphan_count > 0:
            signals.append(SignalType.TOOL_TRANSACTION_ORPHAN)

        if orphan_count > self._max_orphan_tolerance or malformed_ai_indices:
            signals.append(SignalType.TRANSCRIPT_CORRUPTION)

        return ValidationResult(
            is_valid=not dropped_indices and not malformed_ai_indices,
            validated_messages=validated,
            dropped_indices=sorted(list(dropped_indices)),
            dropped_call_ids=dropped_call_ids,
            signals=signals,
            repair_actions=repair_actions,
        )


