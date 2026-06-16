"""_tc_intervention state schema and reducer for LangGraph state channels."""

from __future__ import annotations

from typing import Any, TypedDict


class InterventionStateSchema(TypedDict, total=False):
    """
    The _tc_intervention state channel injected into LangGraph graph state.
    All fields optional to support incremental patches.
    """

    # Stage tracking
    current_stage: str
    stage_entered_at_turn: int
    previous_stage: str

    # Counters
    turn_counter: int
    consecutive_stagnation_count: int
    total_interventions: int
    nudge_count: int
    override_count: int

    # Cooldown
    cooldown_remaining: int
    last_escalation_turn: int
    last_deescalation_turn: int

    # Semantic window
    fingerprint_hashes: list[str]
    last_similarity_score: float

    # Transaction ledger (serialized subset)
    pending_transaction_ids: list[str]
    orphaned_transaction_ids: list[str]
    committed_transaction_count: int
    dropped_this_session: list[str]

    # Coaching
    last_coaching_message: str
    coaching_history: list[str]
    strategies_attempted: list[str]


_LIST_FIELDS = frozenset({
    "fingerprint_hashes",
    "pending_transaction_ids",
    "orphaned_transaction_ids",
    "dropped_this_session",
    "coaching_history",
    "strategies_attempted",
})

_COUNTER_FIELDS = frozenset({
    "turn_counter",
    "total_interventions",
    "nudge_count",
    "override_count",
    "committed_transaction_count",
})


def default_intervention_state() -> InterventionStateSchema:
    """Return the default initial value for the _tc_intervention state channel."""
    return InterventionStateSchema(
        current_stage="pass",
        stage_entered_at_turn=0,
        previous_stage="pass",
        turn_counter=0,
        consecutive_stagnation_count=0,
        total_interventions=0,
        nudge_count=0,
        override_count=0,
        cooldown_remaining=0,
        last_escalation_turn=0,
        last_deescalation_turn=0,
        fingerprint_hashes=[],
        last_similarity_score=0.0,
        pending_transaction_ids=[],
        orphaned_transaction_ids=[],
        committed_transaction_count=0,
        dropped_this_session=[],
        last_coaching_message="",
        coaching_history=[],
        strategies_attempted=[],
    )


def tc_state_reducer(
    existing: InterventionStateSchema | None,
    update: InterventionStateSchema | dict[str, Any],
) -> InterventionStateSchema:
    """
    Reducer for the _tc_intervention state channel.

    Merge semantics:
    - List fields: APPEND with deduplication.
    - Counter fields: MAX(existing, update).
    - All other fields: update overwrites existing.
    """
    if existing is None:
        existing = default_intervention_state()

    result: dict[str, Any] = dict(existing)

    for key, value in update.items():
        if key in _LIST_FIELDS:
            existing_list: list[Any] = result.get(key, [])
            if isinstance(value, list):
                seen = set()
                merged: list[Any] = []
                for item in existing_list + value:
                    item_key = str(item)
                    if item_key not in seen:
                        seen.add(item_key)
                        merged.append(item)
                result[key] = merged
            else:
                result[key] = value
        elif key in _COUNTER_FIELDS:
            existing_val = result.get(key, 0)
            if isinstance(value, (int, float)) and isinstance(existing_val, (int, float)):
                result[key] = max(existing_val, value)
            else:
                result[key] = value
        else:
            result[key] = value

    return InterventionStateSchema(**result)  # type: ignore[typeddict-item]
