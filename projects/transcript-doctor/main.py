"""
Transcript Doctor — Feeds malformed message transcripts through
TokenCircuit's validator to demonstrate the 10 invariants.

Real use case: Your LangGraph agent produces a transcript with
orphaned tool results, duplicate responses, or malformed arguments.
TokenCircuit's TranscriptValidator fixes them before they reach the LLM.

No API keys needed. Everything runs on synthetic transcripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tokencircuit import (
    MessageCanonicalizer,
    TranscriptValidator,
    ToolTransactionLedger,
    SignalType,
)
from tokencircuit.types import CanonicalRole, CanonicalMessage


def _ai(content="", tool_calls=None, idx=0):
    return CanonicalMessage(role=CanonicalRole.AI, content=content,
                            tool_calls=tool_calls, source_index=idx)


def _tool(content, call_id, idx=1):
    return CanonicalMessage(role=CanonicalRole.TOOL, content=content,
                            tool_call_id=call_id, source_index=idx)


def _human(content, idx=0):
    return CanonicalMessage(role=CanonicalRole.HUMAN, content=content,
                            source_index=idx)


def _system(content, idx=0):
    return CanonicalMessage(role=CanonicalRole.SYSTEM, content=content,
                            source_index=idx)


def _tc(call_id, name="search", args=None):
    return {"id": call_id, "name": name, "args": args or {}}


def run_case(num, title, messages, expect_invalid=False, expect_signals=None):
    ledger = ToolTransactionLedger()
    validator = TranscriptValidator(ledger=ledger)
    result = validator.validate(messages, turn_number=1)
    status = "INVALID" if not result.is_valid else "VALID"
    signals = [s.value for s in result.signals]
    dropped = len(result.dropped_indices)
    print(f"  [{num:02d}] {title}")
    print(f"        → {status}, dropped={dropped}, signals={signals}")
    if expect_signals:
        for sig in expect_signals:
            assert sig in [s.value for s in result.signals], \
                f"Expected signal {sig}, got {signals}"
    return result


print("=== Transcript Doctor: 10 Invariants Demonstrated ===")
print()

# Invariant 1 & 2: Call before result, result after call
run_case(1, "Valid: call → result (INV 1,2)", [
    _ai("searching", tool_calls=[_tc("c1")], idx=0),
    _tool("results here", "c1", idx=1),
], expect_invalid=False)

run_case(2, "Orphan: result with no matching call (INV 3)", [
    _ai("thinking", tool_calls=[_tc("c1")], idx=0),
    _tool("orphaned data", "no_such_call", idx=1),
], expect_signals=[SignalType.TOOL_TRANSACTION_ORPHAN.value])

run_case(3, "Duplicate: second result for same call (INV 4)", [
    _ai("call once", tool_calls=[_tc("c1")], idx=0),
    _tool("first result", "c1", idx=1),
    _tool("second result", "c1", idx=2),
], expect_invalid=True)

run_case(4, "Malformed: non-dict args (INV 7)", [
    _ai("bad call", tool_calls=[{"id": "c1", "name": "fn",
                                  "args": "raw_string"}], idx=0),
    _tool("result", "c1", idx=1),
], expect_signals=[SignalType.TRANSCRIPT_CORRUPTION.value])

run_case(5, "Missing call_id (INV 8)", [
    _ai("missing id", tool_calls=[{"name": "fn", "args": {}}], idx=0),
], expect_signals=[SignalType.TRANSCRIPT_CORRUPTION.value])

run_case(6, "Atomic drop: malformed args drops matching tool results (INV 5)", [
    _ai("bad", tool_calls=[{"id": "c1", "name": "fn",
                             "args": {"_raw": "bad"}}], idx=0),
    _tool("this should drop", "c1", idx=1),
], expect_invalid=True)

run_case(7, "Consecutive AI messages valid (INV 9)", [
    _ai("first thought", tool_calls=[_tc("c1")], idx=0),
    _ai("second thought", tool_calls=[_tc("c2")], idx=1),
    _tool("result 1", "c1", idx=2),
    _tool("result 2", "c2", idx=3),
], expect_invalid=False)

run_case(8, "System messages passthrough (INV 10)", [
    _system("Be helpful.", idx=0),
    _ai("ok", tool_calls=[_tc("c1")], idx=1),
    _tool("done", "c1", idx=2),
], expect_invalid=False)

run_case(9, "Result before call (INV 2 violation)", [
    _tool("result appears first", "c1", idx=0),
    _ai("call appears after", tool_calls=[_tc("c1")], idx=1),
], expect_invalid=True)

run_case(10, "Orphan tolerance: above threshold → TRANSCRIPT_CORRUPTION", [
    _ai("call only", tool_calls=[_tc("c1")], idx=0),
    _tool("orphan A", "no_a", idx=1),
    _tool("orphan B", "no_b", idx=2),
    _tool("orphan C", "no_c", idx=3),
], expect_signals=[SignalType.TRANSCRIPT_CORRUPTION.value])

run_case(11, "Auto-repair off: orphans kept as-is", [
    _tool("orphan", "no_call", idx=0),
    _ai("thinking", tool_calls=[_tc("c1")], idx=1),
    _tool("result", "c1", idx=2),
])

# Test auto_repair=False for orphan
print()
print("  [12] Auto-repair OFF → orphans not dropped:")
ledger = ToolTransactionLedger()
validator = TranscriptValidator(ledger=ledger, auto_repair=False)
msgs = [
    _tool("orphan", "no_call", idx=0),
    _ai("legit call", tool_calls=[_tc("c1")], idx=1),
    _tool("legit result", "c1", idx=2),
]
result = validator.validate(msgs, turn_number=1)
print(f"        → validated={len(result.validated_messages)}, "
      f"dropped={len(result.dropped_indices)}")
print(f"        (orphan kept because auto_repair=False)")

run_case(13, "Complex: mixed valid and invalid", [
    _ai("call 1", tool_calls=[_tc("c1")], idx=0),
    _tool("result 1", "c1", idx=1),
    _ai("malformed", tool_calls=[{"id": "c2", "name": "fn",
                                   "args": "bad"}], idx=2),
    _tool("orphan result for c2", "c2", idx=3),
    _ai("call 3", tool_calls=[_tc("c3")], idx=4),
    _tool("result 3", "c3", idx=5),
    _tool("duplicate result 3", "c3", idx=6),
], expect_invalid=True)

# LangChain-like objects
print()
print("  [14] LangChain-style objects canonicalized via MessageCanonicalizer:")
canon = MessageCanonicalizer()
lc_msgs = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Search for X."},
    {"role": "assistant", "content": "Searching.",
     "tool_calls": [{"id": "lc1", "type": "function",
                      "function": {"name": "search",
                                   "arguments": '{"q": "X"}'}}]},
    {"role": "tool", "content": "Found Y.", "tool_call_id": "lc1",
     "name": "search"},
]
cm = canon.canonicalize(lc_msgs)
ledger2 = ToolTransactionLedger()
validator2 = TranscriptValidator(ledger=ledger2)
result2 = validator2.validate(cm, turn_number=1)
print(f"        → {len(result2.validated_messages)} messages kept, "
      f"is_valid={result2.is_valid}")

print()
print("All 14 transcript scenarios complete.")
