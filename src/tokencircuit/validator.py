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
    """Validates and repairs message transcripts before LLM consumption."""

    def __init__(
        self,
        *,
        ledger: ToolTransactionLedger,
        auto_recovery: bool = True,
        max_orphan_tolerance: int = 2,
    ) -> None:
        self._ledger = ledger
        self._auto_recovery = auto_recovery
        self._max_orphan_tolerance = max_orphan_tolerance

    def validate(
        self,
        messages: list[CanonicalMessage],
        turn_number: int,
    ) -> ValidationResult:
        self._ledger.reset()

        dropped_indices: set[int] = set()
        dropped_call_ids: list[str] = []
        repair_actions: list[str] = []
        signals: list[SignalType] = []

        # Pass 1: Structural analysis
        (
            ai_turn_map,
            malformed_ai_indices,
            malformed_call_ids,
            call_id_to_ai_index,
        ) = self._analyze_structure(messages)
        local_turn = max(ai_turn_map.values()) + 1 if ai_turn_map else 1
        self._ledger.advance_turn(local_turn)

        # Pass 2: Ledger registration & result validation
        orphan_count = self._process_messages(
            messages,
            ai_turn_map,
            malformed_ai_indices,
            malformed_call_ids,
            call_id_to_ai_index,
            dropped_indices,
            dropped_call_ids,
            repair_actions,
            local_turn,
        )

        # Reconstruct & Repair
        validated = self._reconstruct_transcript(
            messages, dropped_indices, malformed_ai_indices
        )
        if self._auto_recovery and validated:
            validated = self._repair_dangling_calls(validated, repair_actions)

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

    def _analyze_structure(self, messages: list[CanonicalMessage]):
        ai_turn_map = {}
        malformed_ai_indices = set()
        malformed_call_ids = set()
        call_id_to_ai_index = {}
        local_turn = 1

        for msg in messages:
            if msg.role != CanonicalRole.AI:
                continue

            ai_turn_map[msg.source_index] = local_turn
            local_turn += 1

            if not msg.tool_calls:
                continue

            has_malformed = any(
                not tc.get("id") or not _is_args_valid(tc.get("args"))
                for tc in msg.tool_calls
            )

            if has_malformed:
                malformed_ai_indices.add(msg.source_index)
                malformed_call_ids.update(
                    tc.get("id", "") for tc in msg.tool_calls if tc.get("id")
                )
            else:
                for tc in msg.tool_calls:
                    if cid := tc.get("id"):
                        call_id_to_ai_index[cid] = msg.source_index

        return (
            ai_turn_map,
            malformed_ai_indices,
            malformed_call_ids,
            call_id_to_ai_index,
        )

    def _process_messages(
        self,
        messages,
        ai_turn_map,
        malformed_ai_indices,
        malformed_call_ids,
        call_id_to_ai_index,
        dropped_indices,
        dropped_call_ids,
        repair_actions,
        local_turn,
    ):
        seen_result_ids = set()
        orphan_count = 0

        for msg in messages:
            if msg.role == CanonicalRole.TOOL and not msg.tool_call_id:
                dropped_indices.add(msg.source_index)
                continue

            is_valid_ai = (
                msg.role == CanonicalRole.AI
                and msg.tool_calls
                and msg.source_index not in malformed_ai_indices
            )
            if is_valid_ai:
                msg_turn = ai_turn_map.get(msg.source_index, local_turn)
                for tc in msg.tool_calls:
                    if cid := tc.get("id"):
                        self._ledger.register_call(
                            cid, tc.get("name", "unknown"), msg.source_index, msg_turn
                        )

            if msg.role == CanonicalRole.TOOL and msg.tool_call_id:
                tcid = msg.tool_call_id
                if tcid in malformed_call_ids or tcid in seen_result_ids:
                    dropped_indices.add(msg.source_index)
                    dropped_call_ids.append(tcid)
                    continue
                seen_result_ids.add(tcid)

                ai_index = call_id_to_ai_index.get(tcid)
                if ai_index is None:
                    if self._auto_recovery:
                        dropped_indices.add(msg.source_index)
                        dropped_call_ids.append(tcid)
                        orphan_count += 1
                    continue

                if msg.source_index <= ai_index:
                    dropped_indices.add(msg.source_index)
                    dropped_call_ids.append(tcid)
                    continue

                self._ledger.register_result(
                    tcid,
                    msg.content[:200],
                    len(msg.content),
                    msg.source_index,
                    ai_turn_map.get(ai_index, local_turn),
                )

        return orphan_count

    def _reconstruct_transcript(self, messages, dropped_indices, malformed_ai_indices):
        validated = []
        for msg in messages:
            if msg.source_index in dropped_indices:
                continue
            is_malformed_ai = (
                msg.source_index in malformed_ai_indices
                and msg.role == CanonicalRole.AI
            )
            if is_malformed_ai:
                validated.append(
                    CanonicalMessage(
                        role=msg.role,
                        content=msg.content,
                        tool_calls=[],
                        tool_call_id=msg.tool_call_id,
                        source_index=msg.source_index,
                        name=msg.name,
                    )
                )
            else:
                validated.append(msg)
        return validated

    def _repair_dangling_calls(self, validated, repair_actions):
        resolved_call_ids = {
            m.tool_call_id
            for m in validated
            if m.role == CanonicalRole.TOOL and m.tool_call_id
        }
        repaired = []
        for msg in validated:
            if msg.role == CanonicalRole.AI and msg.tool_calls:
                resolved = [
                    tc for tc in msg.tool_calls if tc.get("id") in resolved_call_ids
                ]
                if len(resolved) != len(msg.tool_calls):
                    repair_actions.append(f"DANGLING_CALL: index {msg.source_index}")
                    msg = CanonicalMessage(
                        role=msg.role,
                        content=msg.content,
                        tool_calls=resolved,
                        tool_call_id=msg.tool_call_id,
                        source_index=msg.source_index,
                        name=msg.name,
                    )
            repaired.append(msg)
        return repaired
