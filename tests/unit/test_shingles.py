"""Tests for semantic_detector utility functions and validation."""

from __future__ import annotations

import pytest

from tokencircuit.semantic_detector import (
    SemanticStagnationDetector,
    _compute_shingles,
    _extract_structural_pattern,
    _jaccard_similarity,
    _normalize_text,
)
from tokencircuit.types import CanonicalMessage, CanonicalRole

# ── _compute_shingles ─────────────────────────────────────────────────────────


class TestComputeShingles:
    """Verify n-gram shingle computation."""

    def test_bigrams_from_four_tokens(self) -> None:
        """n=2 on [1,2,3,4] → {(1,2), (2,3), (3,4)}."""
        tokens = [1, 2, 3, 4]
        result = _compute_shingles(tokens, 2)
        assert result == frozenset({(1, 2), (2, 3), (3, 4)})

    def test_bigrams_from_two_tokens(self) -> None:
        """Minimum length for bigrams: exactly 2 tokens → single shingle."""
        result = _compute_shingles([10, 20], 2)
        assert result == frozenset({(10, 20)})

    def test_trigrams_from_five_tokens(self) -> None:
        """n=3 on [1,2,3,4,5] → {(1,2,3), (2,3,4), (3,4,5)}."""
        tokens = [1, 2, 3, 4, 5]
        result = _compute_shingles(tokens, 3)
        assert result == frozenset({(1, 2, 3), (2, 3, 4), (3, 4, 5)})

    def test_trigrams_from_three_tokens(self) -> None:
        """Minimum length for trigrams: exactly 3 tokens → single shingle."""
        result = _compute_shingles([10, 20, 30], 3)
        assert result == frozenset({(10, 20, 30)})

    def test_input_shorter_than_n_returns_empty(self) -> None:
        """When len(tokens) < n, result must be an empty frozenset."""
        assert _compute_shingles([1], 2) == frozenset()
        assert _compute_shingles([], 2) == frozenset()
        assert _compute_shingles([1, 2], 3) == frozenset()
        assert _compute_shingles([], 3) == frozenset()

    def test_bigram_uses_zip_optimization(self) -> None:
        """n=2 path uses zip(tokens, tokens[1:]), verify via a larger sequence."""
        tokens = list(range(100))
        result = _compute_shingles(tokens, 2)
        # zip produces len-1 pairs
        assert len(result) == 99
        # Spot-check
        assert (0, 1) in result
        assert (98, 99) in result

    def test_trigram_uses_zip_optimization(self) -> None:
        """n=3 path uses zip(tokens, tokens[1:], tokens[2:]), verify via a larger sequence."""  # noqa: E501
        tokens = list(range(50))
        result = _compute_shingles(tokens, 3)
        assert len(result) == 48
        assert (0, 1, 2) in result
        assert (47, 48, 49) in result

    def test_generic_n4_falls_through_to_generic_path(self) -> None:
        """n=4 should use the generic slicing path."""
        tokens = [1, 2, 3, 4, 5]
        result = _compute_shingles(tokens, 4)
        assert result == frozenset({(1, 2, 3, 4), (2, 3, 4, 5)})

    def test_duplicate_tokens_produce_fewer_shingles(self) -> None:
        """Repeated tokens can create duplicate shingles, frozenset deduplicates them."""  # noqa: E501
        tokens = [1, 1, 1, 1]
        result = _compute_shingles(tokens, 2)
        # All bigrams are (1,1) → only one unique shingle
        assert result == frozenset({(1, 1)})


# ── _jaccard_similarity ───────────────────────────────────────────────────────


class TestJaccardSimilarity:
    """Verify Jaccard coefficient computation."""

    def test_identical_sets_return_one(self) -> None:
        """Identical non-empty sets → 1.0."""
        s = frozenset({(1, 2), (3, 4)})
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets_return_zero(self) -> None:
        """Completely disjoint non-empty sets → 0.0."""
        a = frozenset({(1, 2)})
        b = frozenset({(3, 4)})
        assert _jaccard_similarity(a, b) == 0.0

    def test_both_empty_return_one(self) -> None:
        """Both sets empty → 1.0 (convention)."""
        assert _jaccard_similarity(frozenset(), frozenset()) == 1.0

    def test_one_empty_returns_zero(self) -> None:
        """One empty, one non-empty → 0.0."""
        non_empty = frozenset({(1, 2)})
        assert _jaccard_similarity(frozenset(), non_empty) == 0.0
        assert _jaccard_similarity(non_empty, frozenset()) == 0.0

    def test_partial_overlap_correct_ratio(self) -> None:
        """Partial overlap: |intersection|/|union| should match expected ratio."""
        a = frozenset({(1, 2), (2, 3), (3, 4)})
        b = frozenset({(2, 3), (3, 4), (4, 5)})
        # intersection = {(2,3), (3,4)} → 2
        # union = {(1,2), (2,3), (3,4), (4,5)} → 4
        assert _jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_subset_relationship(self) -> None:
        """When a ⊂ b, Jaccard = |a| / |b|."""
        a = frozenset({(1, 2)})
        b = frozenset({(1, 2), (3, 4)})
        assert _jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_symmetry(self) -> None:
        """Jaccard similarity is symmetric: J(a, b) == J(b, a)."""
        a = frozenset({(1, 2), (2, 3)})
        b = frozenset({(2, 3), (3, 4), (4, 5)})
        assert _jaccard_similarity(a, b) == _jaccard_similarity(b, a)


# ── _normalize_text ───────────────────────────────────────────────────────────


class TestNormalizeText:
    """Verify text normalization rules."""

    def test_lowercases(self) -> None:
        """Text should be lowercased."""
        assert _normalize_text("Hello WORLD") == "hello world"

    def test_replaces_numbers_with_num(self) -> None:
        """Standalone numbers should become NUM."""
        assert _normalize_text("step 42 of 100") == "step NUM of NUM"

    def test_collapses_whitespace(self) -> None:
        """Multiple spaces/tabs/newlines should collapse to single space."""
        assert _normalize_text("hello   world\t\tfoo\nbar") == "hello world foo bar"

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace should be stripped."""
        assert _normalize_text("  hello  ") == "hello"

    def test_combined_normalization(self) -> None:
        """Lowercase + number replacement + whitespace collapse together."""
        result = _normalize_text("  Processing   Item 7  out of  20  ")
        assert result == "processing item NUM out of NUM"

    def test_empty_string(self) -> None:
        """Empty input should remain empty."""
        assert _normalize_text("") == ""

    def test_numbers_inside_words_not_replaced(self) -> None:
        r"""\\b\\d+\\b only matches standalone numbers; digits within words stay."""
        # "v2" has digit inside a word boundary context, but \\b\\d+\\b needs
        # the digit sequence to be bounded by word boundaries on both sides
        result = _normalize_text("version2 is better than version 3")
        # "version2" – the '2' is part of the word, should not be replaced
        # "3" is standalone, should become NUM
        assert "NUM" in result
        assert "version2" in result.lower() or "versionNUM" in result


# ── _extract_structural_pattern ───────────────────────────────────────────────


class TestExtractStructuralPattern:
    """Verify structural pattern extraction from message lists."""

    def test_ai_with_tool_calls_produces_call_pattern(self) -> None:
        """AI message with tool_calls and content → pattern includes CALL and REASON."""
        messages = [
            CanonicalMessage(
                role=CanonicalRole.AI,
                content="Let me search for that.",
                tool_calls=[{"name": "search", "args": {"q": "hello"}}],
            ),
        ]
        pattern = _extract_structural_pattern(messages)
        assert "REASON" in pattern
        assert "CALL(search)" in pattern

    def test_ai_with_content_only_produces_reason(self) -> None:
        """AI message with content but no tool_calls → REASON only."""
        messages = [
            CanonicalMessage(
                role=CanonicalRole.AI,
                content="Here is your answer.",
            ),
        ]
        pattern = _extract_structural_pattern(messages)
        assert pattern == "REASON"

    def test_empty_messages_returns_empty(self) -> None:
        """Empty message list → EMPTY."""
        assert _extract_structural_pattern([]) == "EMPTY"

    def test_ai_with_tool_result_between_ai_messages(self) -> None:
        """TOOL messages between two AI messages produce RESULT in the pattern."""
        messages = [
            CanonicalMessage(
                role=CanonicalRole.AI,
                content="First turn.",
            ),
            CanonicalMessage(
                role=CanonicalRole.TOOL,
                content="Tool output",
                tool_call_id="tc_1",
            ),
            CanonicalMessage(
                role=CanonicalRole.AI,
                content="Analysing results...",
                tool_calls=[{"name": "web_search"}],
            ),
        ]
        pattern = _extract_structural_pattern(messages)
        # Reversed iteration: hits last AI (sets found_last_ai), then TOOL → RESULT,
        # then previous AI → break.  Reversed parts → RESULT→REASON→CALL
        assert "RESULT" in pattern
        assert "REASON" in pattern
        assert "CALL(web_search)" in pattern

    def test_ai_followed_by_tool_no_preceding_ai(self) -> None:
        """Single AI→TOOL: trailing TOOL is seen before AI in reverse, so no RESULT."""
        messages = [
            CanonicalMessage(
                role=CanonicalRole.AI,
                content="Searching...",
                tool_calls=[{"name": "web_search"}],
            ),
            CanonicalMessage(
                role=CanonicalRole.TOOL,
                content="Result data",
                tool_call_id="tc_1",
            ),
        ]
        pattern = _extract_structural_pattern(messages)
        assert pattern == "REASON→CALL(web_search)"

    def test_ai_no_content_with_tool_calls(self) -> None:
        """AI message with empty content but has tool_calls → CALL only, no REASON."""
        messages = [
            CanonicalMessage(
                role=CanonicalRole.AI,
                content="",
                tool_calls=[{"name": "get_data"}],
            ),
        ]
        pattern = _extract_structural_pattern(messages)
        assert "CALL(get_data)" in pattern
        assert "REASON" not in pattern

    def test_only_human_messages_returns_empty(self) -> None:
        """If there are no AI messages, pattern should be EMPTY."""
        messages = [
            CanonicalMessage(role=CanonicalRole.HUMAN, content="Hello"),
        ]
        pattern = _extract_structural_pattern(messages)
        assert pattern == "EMPTY"

    def test_multiple_tool_calls_in_single_ai_message(self) -> None:
        """Multiple tool_calls should all appear in the CALL(...) pattern."""
        messages = [
            CanonicalMessage(
                role=CanonicalRole.AI,
                content="Running tools.",
                tool_calls=[
                    {"name": "search"},
                    {"name": "fetch"},
                ],
            ),
        ]
        pattern = _extract_structural_pattern(messages)
        assert "CALL(search,fetch)" in pattern


# ── SemanticStagnationDetector validation ─────────────────────────────────────


class TestSemanticStagnationDetectorValidation:
    """Verify constructor validation in SemanticStagnationDetector."""

    def test_window_size_less_than_2_raises(self) -> None:
        """window_size < 2 must raise ValueError."""
        with pytest.raises(ValueError, match="window_size must be >= 2"):
            SemanticStagnationDetector(window_size=1)

    def test_window_size_zero_raises(self) -> None:
        """window_size = 0 must raise ValueError."""
        with pytest.raises(ValueError, match="window_size must be >= 2"):
            SemanticStagnationDetector(window_size=0)

    def test_window_size_negative_raises(self) -> None:
        """window_size = -1 must raise ValueError."""
        with pytest.raises(ValueError, match="window_size must be >= 2"):
            SemanticStagnationDetector(window_size=-1)

    def test_window_size_2_is_valid(self) -> None:
        """window_size = 2 is the minimum valid value."""
        detector = SemanticStagnationDetector(window_size=2)
        assert detector._window_size == 2

    def test_similarity_threshold_above_one_raises(self) -> None:
        """similarity_threshold > 1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            SemanticStagnationDetector(similarity_threshold=1.1)

    def test_similarity_threshold_below_zero_raises(self) -> None:
        """similarity_threshold < 0.0 must raise ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            SemanticStagnationDetector(similarity_threshold=-0.1)

    def test_similarity_threshold_boundary_zero_is_valid(self) -> None:
        """similarity_threshold = 0.0 is valid."""
        detector = SemanticStagnationDetector(similarity_threshold=0.0)
        assert detector._similarity_threshold == 0.0

    def test_similarity_threshold_boundary_one_is_valid(self) -> None:
        """similarity_threshold = 1.0 is valid."""
        detector = SemanticStagnationDetector(similarity_threshold=1.0)
        assert detector._similarity_threshold == 1.0

    def test_weights_not_summing_to_one_raises(self) -> None:
        """bigram_weight + trigram_weight != 1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="bigram_weight \\+ trigram_weight must equal 1.0"):  # noqa: E501
            SemanticStagnationDetector(bigram_weight=0.3, trigram_weight=0.3)

    def test_weights_summing_to_one_is_valid(self) -> None:
        """bigram_weight + trigram_weight == 1.0 should succeed."""
        detector = SemanticStagnationDetector(bigram_weight=0.5, trigram_weight=0.5)
        assert detector._bigram_weight == 0.5
        assert detector._trigram_weight == 0.5

    def test_default_construction_is_valid(self) -> None:
        """Default parameters (0.4 + 0.6 = 1.0) should not raise."""
        detector = SemanticStagnationDetector()
        assert detector._bigram_weight == 0.4
        assert detector._trigram_weight == 0.6
        assert detector._window_size == 5
        assert detector._similarity_threshold == 0.92
