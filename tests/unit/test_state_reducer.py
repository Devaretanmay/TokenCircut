"""Tests for tc_state_reducer and default_intervention_state."""

from __future__ import annotations

import pytest

from tokencircuit.state_schema import (
    InterventionStateSchema,
    _COUNTER_FIELDS,
    _LIST_FIELDS,
    default_intervention_state,
    tc_state_reducer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def default_state() -> InterventionStateSchema:
    """A fresh default state for reuse across tests."""
    return default_intervention_state()


# ── default_intervention_state ────────────────────────────────────────────────


class TestDefaultInterventionState:
    """Verify default_intervention_state() returns correct initial values."""

    def test_default_stage_is_pass(self, default_state: InterventionStateSchema) -> None:
        """current_stage should default to 'pass'."""
        assert default_state["current_stage"] == "pass"

    def test_default_previous_stage_is_pass(self, default_state: InterventionStateSchema) -> None:
        """previous_stage should default to 'pass'."""
        assert default_state["previous_stage"] == "pass"

    def test_default_counters_are_zero(self, default_state: InterventionStateSchema) -> None:
        """All counter fields should default to 0."""
        for field in _COUNTER_FIELDS:
            assert default_state[field] == 0, f"{field} should be 0"

    def test_default_lists_are_empty(self, default_state: InterventionStateSchema) -> None:
        """All list fields should default to empty lists."""
        for field in _LIST_FIELDS:
            assert default_state[field] == [], f"{field} should be []"

    def test_default_similarity_score_is_zero(self, default_state: InterventionStateSchema) -> None:
        """last_similarity_score should default to 0.0."""
        assert default_state["last_similarity_score"] == 0.0

    def test_default_coaching_message_is_empty(self, default_state: InterventionStateSchema) -> None:
        """last_coaching_message should default to empty string."""
        assert default_state["last_coaching_message"] == ""

    def test_default_cooldown_fields_are_zero(self, default_state: InterventionStateSchema) -> None:
        """Cooldown-related fields should default to 0."""
        assert default_state["cooldown_remaining"] == 0
        assert default_state["last_escalation_turn"] == 0
        assert default_state["last_deescalation_turn"] == 0

    def test_default_returns_typed_dict(self, default_state: InterventionStateSchema) -> None:
        """Return value must be a dict (TypedDict is a dict at runtime)."""
        assert isinstance(default_state, dict)


# ── tc_state_reducer: None existing ───────────────────────────────────────────


class TestReducerNoneExisting:
    """When existing is None, reducer creates defaults then applies the update."""

    def test_none_existing_returns_defaults_with_update(self) -> None:
        """Passing None as existing should start from defaults, then apply the update."""
        result = tc_state_reducer(None, {"current_stage": "nudge"})
        assert result["current_stage"] == "nudge"
        # Other fields should be default values
        assert result["turn_counter"] == 0
        assert result["fingerprint_hashes"] == []

    def test_none_existing_with_empty_update(self) -> None:
        """None existing + empty update should equal the raw defaults."""
        result = tc_state_reducer(None, {})
        expected = default_intervention_state()
        assert dict(result) == dict(expected)


# ── tc_state_reducer: List fields (APPEND with dedup) ─────────────────────────


class TestReducerListFields:
    """List fields should APPEND with deduplication."""

    @pytest.mark.parametrize("field", sorted(_LIST_FIELDS))
    def test_list_append_new_items(
        self, default_state: InterventionStateSchema, field: str
    ) -> None:
        """New items are appended to an empty list field."""
        update = {field: ["a", "b"]}
        result = tc_state_reducer(default_state, update)
        assert result[field] == ["a", "b"]

    @pytest.mark.parametrize("field", sorted(_LIST_FIELDS))
    def test_list_append_deduplicates(
        self, default_state: InterventionStateSchema, field: str
    ) -> None:
        """Duplicate items between existing and update are removed."""
        state_with_items = tc_state_reducer(default_state, {field: ["a", "b"]})
        result = tc_state_reducer(state_with_items, {field: ["b", "c"]})
        assert result[field] == ["a", "b", "c"]

    def test_list_dedup_within_update_itself(
        self, default_state: InterventionStateSchema
    ) -> None:
        """Duplicates *within* the update list itself are also collapsed."""
        result = tc_state_reducer(default_state, {"fingerprint_hashes": ["x", "x", "y"]})
        assert result["fingerprint_hashes"] == ["x", "y"]

    def test_list_dedup_preserves_first_occurrence_order(
        self, default_state: InterventionStateSchema
    ) -> None:
        """Deduplication should keep the first occurrence order: existing first, then new."""
        state = tc_state_reducer(default_state, {"coaching_history": ["c", "a", "b"]})
        result = tc_state_reducer(state, {"coaching_history": ["b", "d", "a"]})
        # existing=[c, a, b] + update=[b, d, a] → dedup keeping order → [c, a, b, d]
        assert result["coaching_history"] == ["c", "a", "b", "d"]

    def test_list_field_with_non_list_value_overwrites(
        self, default_state: InterventionStateSchema
    ) -> None:
        """If a list field receives a non-list value, it overwrites directly."""
        result = tc_state_reducer(default_state, {"fingerprint_hashes": "not-a-list"})
        assert result["fingerprint_hashes"] == "not-a-list"

    def test_list_append_multiple_fields_at_once(
        self, default_state: InterventionStateSchema
    ) -> None:
        """Multiple list fields can be updated in a single call."""
        update = {
            "fingerprint_hashes": ["h1"],
            "coaching_history": ["msg1"],
            "strategies_attempted": ["s1"],
        }
        result = tc_state_reducer(default_state, update)
        assert result["fingerprint_hashes"] == ["h1"]
        assert result["coaching_history"] == ["msg1"]
        assert result["strategies_attempted"] == ["s1"]


# ── tc_state_reducer: Counter fields (MAX) ────────────────────────────────────


class TestReducerCounterFields:
    """Counter fields should use MAX(existing, update)."""

    @pytest.mark.parametrize("field", sorted(_COUNTER_FIELDS))
    def test_counter_max_when_update_is_higher(
        self, default_state: InterventionStateSchema, field: str
    ) -> None:
        """When update > existing, result should be the update value."""
        state = tc_state_reducer(default_state, {field: 5})
        result = tc_state_reducer(state, {field: 10})
        assert result[field] == 10

    @pytest.mark.parametrize("field", sorted(_COUNTER_FIELDS))
    def test_counter_max_when_existing_is_higher(
        self, default_state: InterventionStateSchema, field: str
    ) -> None:
        """When existing > update, result should be the existing value."""
        state = tc_state_reducer(default_state, {field: 10})
        result = tc_state_reducer(state, {field: 3})
        assert result[field] == 10

    @pytest.mark.parametrize("field", sorted(_COUNTER_FIELDS))
    def test_counter_max_when_equal(
        self, default_state: InterventionStateSchema, field: str
    ) -> None:
        """When existing == update, result should be that value."""
        state = tc_state_reducer(default_state, {field: 7})
        result = tc_state_reducer(state, {field: 7})
        assert result[field] == 7

    def test_counter_non_numeric_overwrites(
        self, default_state: InterventionStateSchema
    ) -> None:
        """If a counter field receives a non-numeric value, it overwrites directly."""
        result = tc_state_reducer(default_state, {"turn_counter": "not-a-number"})
        assert result["turn_counter"] == "not-a-number"


# ── tc_state_reducer: Other fields (overwrite) ────────────────────────────────


class TestReducerOtherFields:
    """Non-list, non-counter fields should be overwritten by the update."""

    def test_overwrite_current_stage(
        self, default_state: InterventionStateSchema
    ) -> None:
        """current_stage is an 'other' field and should be overwritten."""
        result = tc_state_reducer(default_state, {"current_stage": "override"})
        assert result["current_stage"] == "override"

    def test_overwrite_similarity_score(
        self, default_state: InterventionStateSchema
    ) -> None:
        """last_similarity_score should be overwritten."""
        result = tc_state_reducer(default_state, {"last_similarity_score": 0.95})
        assert result["last_similarity_score"] == 0.95

    def test_overwrite_coaching_message(
        self, default_state: InterventionStateSchema
    ) -> None:
        """last_coaching_message should be overwritten."""
        state = tc_state_reducer(default_state, {"last_coaching_message": "first"})
        result = tc_state_reducer(state, {"last_coaching_message": "second"})
        assert result["last_coaching_message"] == "second"

    def test_overwrite_preserves_unmentioned_fields(
        self, default_state: InterventionStateSchema
    ) -> None:
        """Fields not present in the update should remain at their existing values."""
        state = tc_state_reducer(
            default_state,
            {"current_stage": "nudge", "last_similarity_score": 0.5},
        )
        result = tc_state_reducer(state, {"current_stage": "override"})
        assert result["current_stage"] == "override"
        assert result["last_similarity_score"] == 0.5


# ── tc_state_reducer: Empty update ────────────────────────────────────────────


class TestReducerEmptyUpdate:
    """An empty update dict should return the existing state unchanged."""

    def test_empty_update_on_default(
        self, default_state: InterventionStateSchema
    ) -> None:
        """Empty update on fresh default returns identical state."""
        result = tc_state_reducer(default_state, {})
        assert dict(result) == dict(default_state)

    def test_empty_update_on_modified_state(
        self, default_state: InterventionStateSchema
    ) -> None:
        """Empty update on a previously modified state keeps all modifications."""
        modified = tc_state_reducer(
            default_state,
            {
                "current_stage": "nudge",
                "turn_counter": 5,
                "fingerprint_hashes": ["abc"],
            },
        )
        result = tc_state_reducer(modified, {})
        assert dict(result) == dict(modified)


# ── tc_state_reducer: Mixed update ────────────────────────────────────────────


class TestReducerMixedUpdate:
    """A single update containing list, counter, and regular fields."""

    def test_mixed_update_applies_all_semantics(
        self, default_state: InterventionStateSchema
    ) -> None:
        """List → append/dedup, counter → MAX, other → overwrite, all in one call."""
        # First, set up some initial state
        initial = tc_state_reducer(
            default_state,
            {
                "fingerprint_hashes": ["h1", "h2"],
                "turn_counter": 5,
                "current_stage": "pass",
            },
        )

        # Now apply a mixed update
        mixed_update = {
            # list field: should append with dedup
            "fingerprint_hashes": ["h2", "h3"],
            # counter field: 3 < 5, should keep 5
            "turn_counter": 3,
            # counter field: 10 > 0, should become 10
            "total_interventions": 10,
            # other field: should overwrite
            "current_stage": "override",
            "last_similarity_score": 0.88,
        }
        result = tc_state_reducer(initial, mixed_update)

        assert result["fingerprint_hashes"] == ["h1", "h2", "h3"]
        assert result["turn_counter"] == 5  # MAX(5, 3)
        assert result["total_interventions"] == 10  # MAX(0, 10)
        assert result["current_stage"] == "override"
        assert result["last_similarity_score"] == 0.88
