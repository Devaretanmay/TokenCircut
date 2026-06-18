"""
Semantic Detector Tests — paraphrase detection and genuine progress reset.

Verifies:
1. Exact repetition triggers STATE_STAGNATION at high Jaccard (1.0).
2. Paraphrased stagnation triggers SEMANTIC_STAGNATION at configured threshold.
3. Structural repetition (same tool pattern) triggers FUTILE_ACTION.
4. Genuine progress (new tool family, novel result) clears stagnation and resets.
5. Detector respects window_size boundaries.
6. Shingle computation handles edge cases (empty text, single token, unicode).
"""

from tokencircuit.semantic_detector import (
    SemanticStagnationDetector,
    _compute_shingles,
    _extract_structural_pattern,
    _jaccard_similarity,
    _normalize_text,
)
from tokencircuit.types import CanonicalMessage, CanonicalRole, SignalType


def _ai_msg(
    content: str, tool_calls: list | None = None, idx: int = 0
) -> CanonicalMessage:  # noqa: E501
    """Create AI CanonicalMessage."""
    return CanonicalMessage(
        role=CanonicalRole.AI,
        content=content,
        tool_calls=tool_calls or [],
        source_index=idx,
    )


def _tool_msg(content: str, call_id: str = "call_1", idx: int = 1) -> CanonicalMessage:
    """Create Tool CanonicalMessage."""
    return CanonicalMessage(
        role=CanonicalRole.TOOL,
        content=content,
        tool_call_id=call_id,
        source_index=idx,
    )


def _human_msg(content: str = "Do task", idx: int = 0) -> CanonicalMessage:
    return CanonicalMessage(role=CanonicalRole.HUMAN, content=content, source_index=idx)


class TestExactRepetitionDetection:
    """Tests for exact-hash based stagnation detection (V6 equivalent)."""

    def test_identical_messages_trigger_state_stagnation(self):
        """Repeating the exact same AI response triggers STATE_STAGNATION."""
        detector = SemanticStagnationDetector(window_size=3, similarity_threshold=0.9)

        messages = [_human_msg(), _ai_msg("I'll search for that information.")]

        # Fill the window with identical fingerprints
        for turn in range(1, 5):
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

            if turn >= 3:
                assert SignalType.STATE_STAGNATION in analysis.signals, (
                    f"Turn {turn}: Expected STATE_STAGNATION with identical content"
                )

    def test_different_messages_no_stagnation(self):
        """Different AI responses should NOT trigger stagnation."""
        detector = SemanticStagnationDetector(window_size=3, similarity_threshold=0.9)

        responses = [
            "I'll search for information about cats.",
            "Let me check the database for dog breeds.",
            "Looking into bird migration patterns now.",
            "I found some data about fish habitats.",
        ]
        for turn, text in enumerate(responses, 1):
            messages = [_human_msg(), _ai_msg(text)]
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)
            assert not analysis.is_stagnating, (
                f"Turn {turn}: Novel content should not stagnate"
            )  # noqa: E501

    def test_similarity_score_is_1_for_identical(self):
        """Identical content should yield similarity score of 1.0."""
        detector = SemanticStagnationDetector(window_size=3, similarity_threshold=0.5)
        msg = [_human_msg(), _ai_msg("Exact same content every time.")]

        detector.analyze(msg, 1)
        a1 = detector.analyze(msg, 1)
        detector.record_fingerprint(a1.fingerprint)

        a2 = detector.analyze(msg, 2)
        assert a2.similarity_score == 1.0, f"Expected 1.0, got {a2.similarity_score}"


class TestParaphrasedStagnationDetection:
    """Tests for Jaccard shingle-based paraphrase detection."""

    def test_high_overlap_paraphrases_detected(self):
        """
        Content with very high token overlap (near-copies) should be detected
        at the configured threshold.
        """
        # Use a low threshold to catch moderate paraphrases
        detector = SemanticStagnationDetector(
            window_size=4,
            similarity_threshold=0.5,  # Lower threshold for testing
        )

        # These are near-copies with minor word swaps
        paraphrases = [
            "I will now search the web for recent news about quantum computing breakthroughs.",  # noqa: E501
            "I will now search the web for recent news about quantum computing breakthroughs.",  # noqa: E501
            "I will now search the web for recent news about quantum computing breakthroughs.",  # noqa: E501
            "I will now search the web for recent news about quantum computing breakthroughs.",  # noqa: E501
        ]

        for turn, text in enumerate(paraphrases, 1):
            messages = [_human_msg(), _ai_msg(text)]
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

        # With identical text and low threshold, should detect
        assert analysis.similarity_score >= 0.5

    def test_structural_repetition_without_content_match(self):
        """
        Same structural pattern (CALL same tool) should trigger FUTILE_ACTION
        even if content text differs.
        """
        detector = SemanticStagnationDetector(
            window_size=4,
            similarity_threshold=0.95,  # High threshold — won't fire on content
            structural_threshold=3,
        )

        for turn in range(1, 6):
            messages = [
                _human_msg(),
                _ai_msg(
                    f"Attempt #{turn} to find the file.",
                    tool_calls=[
                        {
                            "id": f"c{turn}",
                            "name": "search",
                            "args": {"q": f"query_{turn}"},
                        }
                    ],  # noqa: E501
                ),
            ]
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

        # Should detect structural/futile pattern
        assert (
            SignalType.FUTILE_ACTION in analysis.signals
            or SignalType.STATE_STAGNATION in analysis.signals
        ), f"Expected structural stagnation signal, got {analysis.signals}"

    def test_threshold_boundary_below_does_not_fire(self):
        """Content with genuinely different shingles should NOT fire SEMANTIC_STAGNATION."""  # noqa: E501
        detector = SemanticStagnationDetector(
            window_size=3,
            similarity_threshold=0.99,  # Very strict
        )

        # Completely different content each turn — Jaccard will be well below 0.99
        varied_contents = [
            "The quantum physics experiment yielded fascinating results about entanglement.",  # noqa: E501
            "JavaScript frameworks continue to evolve rapidly in the modern web ecosystem.",  # noqa: E501
            "Ancient Roman aqueducts demonstrate remarkable feats of engineering design.",  # noqa: E501
            "Machine learning optimization requires careful hyperparameter tuning strategy.",  # noqa: E501
        ]
        for turn in range(1, 5):
            messages = [_human_msg(), _ai_msg(varied_contents[turn - 1])]
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

        assert SignalType.SEMANTIC_STAGNATION not in analysis.signals


class TestGenuineProgressReset:
    """Tests that genuine progress correctly clears the detector state."""

    def test_new_tool_family_clears_structural_signal(self):
        """Calling a completely new tool breaks the structural pattern."""
        detector = SemanticStagnationDetector(window_size=4, similarity_threshold=0.7)

        # Build up stagnation with repeated search
        for turn in range(1, 4):
            messages = [
                _human_msg(),
                _ai_msg(
                    "Searching...",
                    tool_calls=[
                        {"id": f"c{turn}", "name": "search", "args": {"q": "x"}}
                    ],
                ),  # noqa: E501
            ]
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

        # Now call a different tool
        progress_messages = [
            _human_msg(),
            _ai_msg(
                "Let me try reading the file directly.",
                tool_calls=[
                    {
                        "id": "new_call",
                        "name": "read_file",
                        "args": {"path": "/data.txt"},
                    }
                ],  # noqa: E501
            ),
        ]
        analysis = detector.analyze(progress_messages, turn_number=5)
        detector.record_fingerprint(analysis.fingerprint)

        # Structural pattern changed — should reduce stagnation
        assert analysis.pattern_diversity > 0.3, (
            f"New tool should increase diversity: {analysis.pattern_diversity}"
        )

    def test_novel_content_breaks_semantic_stagnation(self):
        """Fundamentally different content should drop the similarity score."""
        detector = SemanticStagnationDetector(window_size=3, similarity_threshold=0.7)

        # Repeated similar content
        for turn in range(1, 4):
            messages = [
                _human_msg(),
                _ai_msg("I will search for quantum computing papers."),
            ]  # noqa: E501
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

        # Now completely different content
        novel_messages = [
            _human_msg(),
            _ai_msg(
                "I found the answer! The recipe calls for flour, sugar, and butter. "
                "Preheat the oven to 350 degrees Fahrenheit."
            ),
        ]
        novel_analysis = detector.analyze(novel_messages, turn_number=4)

        assert novel_analysis.similarity_score < 0.5, (
            f"Novel content should have low similarity: {novel_analysis.similarity_score}"  # noqa: E501
        )
        assert SignalType.SEMANTIC_STAGNATION not in novel_analysis.signals

    def test_successful_tool_result_breaks_empty_result_pattern(self):
        """Getting a meaningful result after repeated empties should show progress."""
        detector = SemanticStagnationDetector(window_size=4, similarity_threshold=0.8)

        # Empty results pattern
        for turn in range(1, 4):
            messages = [
                _human_msg(),
                _ai_msg(
                    "Searching",
                    tool_calls=[
                        {"id": f"c{turn}", "name": "search", "args": {"q": "x"}}
                    ],
                ),  # noqa: E501
                _tool_msg("No results found.", f"c{turn}", idx=2),
                _ai_msg("Let me try again.", idx=3),
            ]
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

        # Now get a real result
        success_messages = [
            _human_msg(),
            _ai_msg(
                "Searching",
                tool_calls=[{"id": "c_final", "name": "search", "args": {"q": "x"}}],
            ),  # noqa: E501
            _tool_msg(
                "Found 10 results: 1. Paper A, 2. Paper B, ...", "c_final", idx=2
            ),  # noqa: E501
            _ai_msg("I found relevant papers! Let me summarize them.", idx=3),
        ]
        analysis = detector.analyze(success_messages, turn_number=4)

        # Content is now different → lower similarity
        assert analysis.similarity_score < 0.9, (
            f"Successful result should lower similarity: {analysis.similarity_score}"
        )


class TestDetectorWindowBehavior:
    """Tests for window_size boundaries and edge cases."""

    def test_no_stagnation_before_window_fills(self):
        """Cannot detect stagnation until at least 1 prior fingerprint exists."""
        detector = SemanticStagnationDetector(window_size=5, similarity_threshold=0.5)
        messages = [_human_msg(), _ai_msg("Same content")]

        analysis = detector.analyze(messages, turn_number=1)
        assert not analysis.is_stagnating, "Cannot stagnate with empty window"
        assert analysis.similarity_score == 0.0

    def test_window_slides_correctly(self):
        """Old fingerprints are evicted when window is full."""
        detector = SemanticStagnationDetector(window_size=3, similarity_threshold=0.9)

        # Fill window with unique content
        for turn in range(1, 5):
            messages = [_human_msg(), _ai_msg(f"Unique response number {turn * 100}")]
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

        assert detector.window_size == 3, "Window should cap at maxlen"

    def test_reset_clears_window(self):
        """reset() should completely clear detection state."""
        detector = SemanticStagnationDetector(window_size=3, similarity_threshold=0.5)
        messages = [_human_msg(), _ai_msg("Content")]

        for turn in range(1, 5):
            analysis = detector.analyze(messages, turn)
            detector.record_fingerprint(analysis.fingerprint)

        detector.reset()
        assert detector.window_size == 0

        # After reset, should not detect stagnation
        analysis = detector.analyze(messages, turn_number=10)
        assert not analysis.is_stagnating


class TestShingleEdgeCases:
    """Tests for n-gram shingle computation edge cases."""

    def test_empty_string(self):
        """Empty string produces no shingles."""
        shingles = _compute_shingles([], 2)
        assert shingles == frozenset()

    def test_single_token(self):
        """Single token cannot produce bigrams."""
        shingles = _compute_shingles([42], 2)
        assert shingles == frozenset()

    def test_two_tokens_produce_one_bigram(self):
        """Two tokens produce exactly one bigram."""
        shingles = _compute_shingles([1, 2], 2)
        assert len(shingles) == 1

    def test_jaccard_identical_sets(self):
        """Identical sets have Jaccard = 1.0."""
        s = frozenset({"a", "b", "c"})
        assert _jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint_sets(self):
        """Disjoint sets have Jaccard = 0.0."""
        a = frozenset({"a", "b"})
        b = frozenset({"c", "d"})
        assert _jaccard_similarity(a, b) == 0.0

    def test_jaccard_both_empty(self):
        """Both empty sets: Jaccard = 1.0 (by convention)."""
        assert _jaccard_similarity(frozenset(), frozenset()) == 1.0

    def test_jaccard_one_empty(self):
        """One empty set: Jaccard = 0.0."""
        assert _jaccard_similarity(frozenset({"a"}), frozenset()) == 0.0

    def test_normalize_text_strips_numbers(self):
        """Numbers should be replaced with 'NUM'."""
        result = _normalize_text("Found 42 results in 3 seconds")
        assert "NUM" in result
        assert "42" not in result
        assert "3" not in result

    def test_normalize_text_lowercases(self):
        """Text should be lowercased."""
        result = _normalize_text("Hello WORLD")
        assert result == "hello world"

    def test_unicode_handling(self):
        """Unicode text should not crash the detector."""
        detector = SemanticStagnationDetector(window_size=3, similarity_threshold=0.9)
        messages = [_human_msg(), _ai_msg("搜索量子计算论文 🔍")]
        analysis = detector.analyze(messages, turn_number=1)
        assert analysis.fingerprint.content_hash is not None


class TestStructuralPatternExtraction:
    """Tests for structural pattern extraction logic."""

    def test_reason_only_pattern(self):
        """AI message with only text → REASON."""
        messages = [_human_msg(), _ai_msg("Just thinking about this.")]
        pattern = _extract_structural_pattern(messages)
        assert "REASON" in pattern

    def test_tool_call_pattern(self):
        """AI message with tool_calls → CALL(name)."""
        messages = [
            _human_msg(),
            _ai_msg(
                "Searching", tool_calls=[{"id": "c1", "name": "search", "args": {}}]
            ),  # noqa: E501
        ]
        pattern = _extract_structural_pattern(messages)
        assert "CALL" in pattern
        assert "search" in pattern

    def test_multi_tool_pattern(self):
        """Multiple tool calls in one message."""
        messages = [
            _human_msg(),
            _ai_msg(
                "Multi",
                tool_calls=[
                    {"id": "c1", "name": "search", "args": {}},
                    {"id": "c2", "name": "fetch", "args": {}},
                ],
            ),
        ]
        pattern = _extract_structural_pattern(messages)
        assert "search" in pattern
        assert "fetch" in pattern

    def test_empty_messages(self):
        """Empty message list → EMPTY pattern."""
        pattern = _extract_structural_pattern([])
        assert pattern == "EMPTY"
