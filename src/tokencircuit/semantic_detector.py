"""
SemanticStagnationDetector — zero-dependency semantic loop detection via token n-gram shingling.

Uses tiktoken (cl100k_base) to tokenize, then computes Jaccard similarity over
2-gram and 3-gram shingle sets of normalized assistant text to detect paraphrased
loops without requiring embedding models.

Detection Modes:
1. EXACT HASH: Identical content_hash across window (V6 equivalent).
2. STRUCTURAL: Same tool-calling pattern repeats (e.g., "REASON→TOOL_CALL→OBSERVE").
3. SHINGLE SIMILARITY: Jaccard coefficient of token n-gram sets exceeds threshold.

A turn is stagnating if ANY of:
- content_hash matches ≥ (window_size - 1) entries (exact repeat)
- structural_pattern matches ≥ (window_size - 1) entries (structural loop)
- Average Jaccard similarity to window ≥ semantic_similarity_threshold
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque
from typing import Any, Optional

from .types import (
    CanonicalMessage,
    CanonicalRole,
    SemanticFingerprint,
    SignalType,
)

logger = logging.getLogger("tokencircuit.semantic_detector")

# Lazy-loaded tiktoken encoder
_encoder: Optional[object] = None


def _get_encoder() -> object:
    """Lazy-load tiktoken cl100k_base encoder."""
    global _encoder
    if _encoder is None:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _compute_shingles(token_ids: list[int], n: int) -> frozenset[tuple[int, ...]]:
    """Compute n-gram shingles from token IDs efficiently."""
    if len(token_ids) < n:
        return frozenset()

    if n == 2:
        return frozenset(zip(token_ids, token_ids[1:]))
    if n == 3:
        return frozenset(zip(token_ids, token_ids[1:], token_ids[2:]))

    # Generic case for n > 3
    return frozenset(
        tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)
    )


def _jaccard_similarity(a: frozenset[Any], b: frozenset[Any]) -> float:
    """Compute Jaccard coefficient between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union > 0 else 0.0


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip numbers."""
    text = text.lower()
    # Remove specific numeric values but keep structure
    text = re.sub(r"\b\d+\b", "NUM", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_structural_pattern(messages: list[CanonicalMessage]) -> str:
    """
    Extract the structural pattern of the last AI turn.
    Pattern encodes the sequence of message types:
    e.g., "AI_REASON→TOOL_CALL(search)→TOOL_RESULT→AI_REASON"
    """
    if not messages:
        return "EMPTY"

    # Find the last AI message and its associated tool interactions
    pattern_parts: list[str] = []
    found_last_ai = False

    for msg in reversed(messages):
        if msg.role == CanonicalRole.AI and not found_last_ai:
            found_last_ai = True
            # To maintain original order [REASON, CALL] after reverse:
            # Original insert(0, CALL) then insert(0, REASON)
            if msg.tool_calls:
                tool_names = [tc.get("name", "?") for tc in msg.tool_calls]
                pattern_parts.append(f"CALL({','.join(tool_names)})")
            if msg.content.strip():
                pattern_parts.append("REASON")
        elif msg.role == CanonicalRole.TOOL and found_last_ai:
            pattern_parts.append("RESULT")
        elif msg.role == CanonicalRole.AI and found_last_ai:
            # Previous AI message — stop
            break

    if not pattern_parts:
        return "EMPTY"

    pattern_parts.reverse()
    return "→".join(pattern_parts)


class StagnationAnalysis:
    """Output of SemanticStagnationDetector.analyze()."""

    __slots__ = (
        "is_stagnating",
        "similarity_score",
        "pattern_diversity",
        "signals",
        "fingerprint",
        "window_summary",
    )

    def __init__(
        self,
        *,
        is_stagnating: bool,
        similarity_score: float,
        pattern_diversity: float,
        signals: list[SignalType],
        fingerprint: SemanticFingerprint,
        window_summary: str = "",
    ) -> None:
        self.is_stagnating = is_stagnating
        self.similarity_score = similarity_score
        self.pattern_diversity = pattern_diversity
        self.signals = signals
        self.fingerprint = fingerprint
        self.window_summary = window_summary


class SemanticStagnationDetector:
    """
    Detects semantic-level stagnation using token n-gram Jaccard similarity.

    Zero external embedding dependencies — uses tiktoken cl100k_base for tokenization,
    then computes 2-gram and 3-gram shingle sets for Jaccard comparison.
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        similarity_threshold: float = 0.92,
        structural_threshold: int = 3,
        bigram_weight: float = 0.4,
        trigram_weight: float = 0.6,
    ) -> None:
        """
        Args:
            window_size: Sliding window of fingerprints to compare against.
            similarity_threshold: Jaccard score above which stagnation fires.
            structural_threshold: How many identical patterns trigger structural signal.
            bigram_weight: Weight of 2-gram similarity in combined score.
            trigram_weight: Weight of 3-gram similarity in combined score.
        """
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0.0, 1.0]")
        if abs(bigram_weight + trigram_weight - 1.0) > 1e-6:
            raise ValueError("bigram_weight + trigram_weight must equal 1.0")

        self._window_size = window_size
        self._similarity_threshold = similarity_threshold
        self._structural_threshold = structural_threshold
        self._bigram_weight = bigram_weight
        self._trigram_weight = trigram_weight
        self._window: deque[SemanticFingerprint] = deque(maxlen=window_size)

    def hydrate_from_history(self, messages: list[CanonicalMessage]) -> None:
        """
        Rebuild the sliding window from message history.
        This allows stateless operation by recomputing fingerprints from the transcript.
        """
        self.reset()
        # Group messages into turns (AI message and subsequent tool results)
        turns: list[list[CanonicalMessage]] = []
        current_turn: list[CanonicalMessage] = []

        for msg in messages:
            if msg.role == CanonicalRole.AI:
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            elif current_turn:
                current_turn.append(msg)

        if current_turn:
            turns.append(current_turn)

        # Compute fingerprints for all but the last turn (which is analyzed separately)
        # and fill the window.
        for i, turn_msgs in enumerate(turns[:-1]):
            fp = self.compute_fingerprint(turn_msgs, i + 1)
            self.record_fingerprint(fp)

    def analyze(
        self,
        messages: list[CanonicalMessage],
        turn_number: int,
    ) -> StagnationAnalysis:
        """
        Analyze the current turn for semantic stagnation.

        Steps:
        1. Compute SemanticFingerprint for the current turn.
        2. Compare shingle sets against the sliding window.
        3. Check structural pattern diversity.
        4. Check exact hash repetition.
        5. Emit signals based on thresholds.
        """
        fingerprint = self.compute_fingerprint(messages, turn_number)
        signals: list[SignalType] = []

        if len(self._window) == 0:
            # Not enough history — no stagnation possible
            return StagnationAnalysis(
                is_stagnating=False,
                similarity_score=0.0,
                pattern_diversity=1.0,
                signals=[],
                fingerprint=fingerprint,
                window_summary="Insufficient history (0 prior turns)",
            )

        # ----- Exact hash check (V6 compatibility) -----
        hash_matches = sum(
            1 for fp in self._window if fp.content_hash == fingerprint.content_hash
        )
        window_len = len(self._window)

        if hash_matches >= min(window_len, self._window_size - 1):
            signals.append(SignalType.STATE_STAGNATION)

        # ----- Structural pattern check -----
        pattern_matches = sum(
            1 for fp in self._window
            if fp.structural_pattern == fingerprint.structural_pattern
        )
        unique_patterns = len(set(fp.structural_pattern for fp in self._window))
        pattern_diversity = unique_patterns / max(window_len, 1)

        if pattern_matches >= self._structural_threshold:
            # Same structure repeated — check if tool signature also repeats
            sig_matches = sum(
                1 for fp in self._window
                if fp.tool_signature == fingerprint.tool_signature
                and fp.tool_signature != "NO_TOOL_CALL"
            )
            if sig_matches >= self._structural_threshold:
                signals.append(SignalType.FUTILE_ACTION)

        # ----- Shingle-based semantic similarity -----
        similarity_score = self._compute_window_similarity(fingerprint)

        if similarity_score >= self._similarity_threshold and window_len >= 2:
            signals.append(SignalType.SEMANTIC_STAGNATION)

        is_stagnating = len(signals) > 0

        # Build summary
        summary_parts = [
            f"window={window_len}/{self._window_size}",
            f"sim={similarity_score:.3f}",
            f"hash_matches={hash_matches}",
            f"pattern_diversity={pattern_diversity:.2f}",
        ]
        if signals:
            summary_parts.append(f"signals={[s.value for s in signals]}")

        return StagnationAnalysis(
            is_stagnating=is_stagnating,
            similarity_score=similarity_score,
            pattern_diversity=pattern_diversity,
            signals=signals,
            fingerprint=fingerprint,
            window_summary=", ".join(summary_parts),
        )

    def _compute_window_similarity(self, fingerprint: SemanticFingerprint) -> float:
        """Compute weighted average Jaccard similarity against the window."""
        if not self._window or (not fingerprint.bigram_set and not fingerprint.trigram_set):
            return 0.0

        total_sim = 0.0
        count = 0

        current_bigrams = fingerprint.bigram_set
        current_trigrams = fingerprint.trigram_set

        for fp in self._window:
            if not fp.bigram_set and not fp.trigram_set:
                continue

            bi_sim = _jaccard_similarity(current_bigrams, fp.bigram_set)
            tri_sim = _jaccard_similarity(current_trigrams, fp.trigram_set)

            combined = (self._bigram_weight * bi_sim) + (self._trigram_weight * tri_sim)
            total_sim += combined
            count += 1

        return total_sim / count if count > 0 else 0.0

    def compute_fingerprint(
        self,
        messages: list[CanonicalMessage],
        turn_number: int,
    ) -> SemanticFingerprint:
        """Compute a SemanticFingerprint for the given messages."""
        # Extract the last AI message content for fingerprinting
        ai_content = ""
        tool_signature = "NO_TOOL_CALL"

        for msg in reversed(messages):
            if msg.role == CanonicalRole.AI:
                ai_content = msg.content
                if msg.tool_calls:
                    sorted_calls = sorted(msg.tool_calls, key=lambda x: x.get("name", "?"))
                    names = [tc.get("name", "?") for tc in sorted_calls]
                    arg_types = []
                    for tc in sorted_calls:
                        args = tc.get("args", {})
                        if isinstance(args, dict):
                            at = ",".join(type(v).__name__ for v in args.values())
                        else:
                            at = "?"
                        arg_types.append(at)
                    tool_signature = "+".join(
                        f"{n}({a})" for n, a in zip(names, arg_types)
                    )
                break

        # Compute content hash
        content_hash = hashlib.sha256(ai_content.encode()).hexdigest()

        # Compute structural pattern
        structural_pattern = _extract_structural_pattern(messages)

        # Compute shingle set (combined 2-grams and 3-grams)
        normalized = _normalize_text(ai_content)
        encoder = _get_encoder()
        token_ids = encoder.encode(normalized)  # type: ignore[union-attr]

        bigrams = _compute_shingles(token_ids, 2)
        trigrams = _compute_shingles(token_ids, 3)

        return SemanticFingerprint(
            turn_number=turn_number,
            content_hash=content_hash,
            tool_signature=tool_signature,
            structural_pattern=structural_pattern,
            bigram_set=bigrams,
            trigram_set=trigrams,
        )

    def record_fingerprint(self, fingerprint: SemanticFingerprint) -> None:
        """Add a fingerprint to the sliding window."""
        self._window.append(fingerprint)

    def get_window(self) -> list[SemanticFingerprint]:
        """Return the current sliding window."""
        return list(self._window)

    def reset(self) -> None:
        """Clear the sliding window."""
        self._window.clear()

    @property
    def window_size(self) -> int:
        """Current number of fingerprints in the window."""
        return len(self._window)

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold
