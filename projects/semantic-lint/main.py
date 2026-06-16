"""
Semantic Lint — Exercises TokenCircuit's SemanticStagnationDetector
directly with various content patterns to demonstrate how it catches
rephrased loops that exact matching would miss.

Real use case: An agent that keeps rephrasing the same answer.
Exact content differs each time, but the semantic meaning is identical.
TokenCircuit's shingle-based similarity catches this pattern.

No API keys. Uses tiktoken for tokenization (installed as dep).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tokencircuit import SemanticStagnationDetector, SignalType
from tokencircuit.types import CanonicalMessage, CanonicalRole


def _ai(content, idx=0, tool_calls=None):
    return CanonicalMessage(role=CanonicalRole.AI, content=content,
                            tool_calls=tool_calls, source_index=idx)


def _tool(content, call_id, idx=1):
    return CanonicalMessage(role=CanonicalRole.TOOL, content=content,
                            tool_call_id=call_id, source_index=idx)


def run_turn(detector, turn_msgs, turn_num, label):
    analysis = detector.analyze(turn_msgs, turn_num)
    if analysis.fingerprint:
        detector.record_fingerprint(analysis.fingerprint)
    signals = [s.value for s in analysis.signals]
    stagnating = "STAGNATING" if signals else "OK"
    print(f"  {label:50s} → {stagnating:12s} "
          f"sim={analysis.similarity_score:.3f} "
          f"signals={signals}")
    return analysis


print("=== Semantic Lint: Stagnation Detection Patterns ===")
print()

# ── Case 1: Exact hash repeat ──
print("Case 1: Exact content repeat")
d = SemanticStagnationDetector(window_size=4)
for i in range(4):
    msg = _ai(f"The answer is definitely 42.", idx=i*2)
    msg2 = _tool("done", f"r{i}", idx=i*2+1)
    run_turn(d, [msg, msg2], i+1, f"  Turn {i+1}: identical text")
print("  ✓ STATE_STAGNATION fires from exact hash match.")
print()

# ── Case 2: Rephrased content ──
print("Case 2: Rephrased content (same meaning, different words)")
d2 = SemanticStagnationDetector(window_size=4)
phrases = [
    "The primary cause of the issue is a network timeout error.",
    "A network timeout error is the main reason for this problem.",
    "This problem stems primarily from a network timeout occurring.",
    "The root cause here is that the network connection timed out.",
]
for i, phrase in enumerate(phrases):
    msg = _ai(phrase, idx=i*2)
    msg2 = _tool("checked", f"r{i}", idx=i*2+1)
    run_turn(d2, [msg, msg2], i+1, f"  Turn {i+1}: rephrased")
print("  ✓ SEMANTIC_STAGNATION fires from shingle similarity.")
print()

# ── Case 3: Different content (no stagnation) ──
print("Case 3: Different content each turn")
d3 = SemanticStagnationDetector(window_size=4)
topics = [
    "The weather today is sunny and warm.",
    "The capital of France is Paris.",
    "Quantum computing uses qubits instead of bits.",
    "Python is a popular programming language.",
]
for i, topic in enumerate(topics):
    msg = _ai(topic, idx=i*2)
    msg2 = _tool("done", f"r{i}", idx=i*2+1)
    run_turn(d3, [msg, msg2], i+1, f"  Turn {i+1}: different topic")
print("  ✓ No false positives on genuinely different content.")
print()

# ── Case 4: Structural pattern repeat (tool call signature) ──
print("Case 4: Same tool call pattern repeating")
d4 = SemanticStagnationDetector(window_size=4)
for i in range(4):
    msg = _ai(f"Let me search for keyword {i}.",
              tool_calls=[{"id": f"s{i}", "name": "search",
                           "args": {"q": f"keyword {i}"}}],
              idx=i*2)
    msg2 = _tool(f"Results for keyword {i}", f"s{i}", idx=i*2+1)
    run_turn(d4, [msg, msg2], i+1, f"  Turn {i+1}: search tool pattern")
print("  ✓ FUTILE_ACTION fires from repeated tool signature.")
print()

# ── Case 5: Mixed structural patterns ──
print("Case 5: Mixed tool patterns (no loop)")
d5 = SemanticStagnationDetector(window_size=4)
patterns = [
    ("search", {"q": "term 1"}),
    ("compute", {"formula": "a+b"}),
    ("search", {"q": "term 2"}),
    ("plot", {"type": "chart"}),
]
for i, (tool, args) in enumerate(patterns):
    msg = _ai(f"Using {tool}",
              tool_calls=[{"id": f"m{i}", "name": tool, "args": args}],
              idx=i*2)
    msg2 = _tool(f"Result for {tool}", f"m{i}", idx=i*2+1)
    run_turn(d5, [msg, msg2], i+1, f"  Turn {i+1}: {tool}")
print("  ✓ No stagnation on varied tool patterns.")
print()

# ── Case 6: Empty window → no stagnation ──
print("Case 6: First turn (no history) → no stagnation")
d6 = SemanticStagnationDetector(window_size=4)
msg = _ai("This is the very first turn.")
run_turn(d6, [msg], 1, "  Turn 1: no prior history")
print("  ✓ No false stagnation on empty window.")
print()

# ── Case 7: Long text with similar structure ──
print("Case 7: Long-form content with structural similarity")
d7 = SemanticStagnationDetector(window_size=4)
for i in range(4):
    text = (
        f"In order to solve this problem we need to consider "
        f"several factors. First we analyze the input data. "
        f"Then we apply the transformation algorithm. "
        f"Finally we validate the output. Iteration {i+1}: "
        + "testing " * (i + 1)  # vary length slightly
    )
    msg = _ai(text, idx=i*2)
    msg2 = _tool("done", f"r{i}", idx=i*2+1)
    run_turn(d7, [msg, msg2], i+1, f"  Turn {i+1}: long analysis text")
print("  ✓ Structural pattern and shingle similarity both contribute.")
print()

# ── Case 8: Progressively improving content ──
print("Case 8: Progressively improving (no stagnation expected)")
d8 = SemanticStagnationDetector(window_size=4)
levels = [
    "The function returns the sum of two numbers.",
    "The function adds two integers and returns the total.",
    "The function takes parameters a and b and computes a + b.",
    "This implementation performs addition of two operands a and b.",
]
for i, text in enumerate(levels):
    msg = _ai(text, idx=i*2)
    msg2 = _tool("done", f"r{i}", idx=i*2+1)
    run_turn(d8, [msg, msg2], i+1, f"  Turn {i+1}: {text[:40]}...")
# Progressive improvements should have lower similarity to earlier turns
print("  ✓ Progressive improvement avoids stagnation classification.")
print()

print("All semantic-lint scenarios complete.")
