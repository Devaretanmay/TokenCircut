from __future__ import annotations

from tokencircuit.canonicalizer import MessageCanonicalizer
from tokencircuit.ledger import ToolTransactionLedger
from tokencircuit.types import CanonicalMessage, CanonicalRole
from tokencircuit.validator import TranscriptValidator


def _ai(idx: int, tool_calls: list[dict]) -> CanonicalMessage:
    return CanonicalMessage(
        role=CanonicalRole.AI,
        content="",
        tool_calls=tool_calls,
        source_index=idx,
    )


def _tool(idx: int, call_id: str) -> CanonicalMessage:
    return CanonicalMessage(
        role=CanonicalRole.TOOL,
        content="ok",
        tool_call_id=call_id,
        source_index=idx,
        name="tool",
    )


def _validator() -> TranscriptValidator:
    return TranscriptValidator(ledger=ToolTransactionLedger())


def test_strips_one_missing_tool_call_from_multi_call_ai_message() -> None:
    messages = [
        _ai(
            0,
            [
                {"id": "kept", "name": "tool", "args": {}},
                {"id": "stripped", "name": "tool", "args": {}},
            ],
        ),
        _tool(1, "kept"),
    ]

    result = _validator().validate(messages, turn_number=1)

    ai = result.validated_messages[0]
    assert [tc["id"] for tc in ai.tool_calls] == ["kept"]
    assert result.validated_messages[1].tool_call_id == "kept"
    assert result.repair_actions == ["DANGLING_CALL: index 0"]


def test_strips_dangling_tool_call_when_tool_message_missing() -> None:
    messages = [_ai(0, [{"id": "missing", "name": "tool", "args": {}}])]

    result = _validator().validate(messages, turn_number=1)

    assert result.validated_messages[0].tool_calls == []
    assert result.repair_actions == ["DANGLING_CALL: index 0"]


def test_strips_malformed_json_tool_call_and_matching_tool_message() -> None:
    messages = MessageCanonicalizer().canonicalize(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "bad-json",
                        "type": "function",
                        "function": {
                            "name": "tool",
                            "arguments": '{"unterminated": ',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "bad-json",
                "name": "tool",
                "content": "would orphan if not stripped",
            },
        ]
    )

    result = _validator().validate(messages, turn_number=1)

    assert result.validated_messages == [
        CanonicalMessage(role=CanonicalRole.AI, content="", source_index=0)
    ]
    assert result.dropped_call_ids == ["bad-json"]


def test_drops_tool_message_appearing_before_its_ai_message() -> None:
    # Invariant 2 (RESULT-AFTER-CALL): ToolMessage source_index <= ai source_index
    # must be dropped even when the tool_call_id is legitimately registered.
    ai_msg = _ai(5, [{"id": "call-A", "name": "tool", "args": {}}])
    # ToolMessage at index 2, AI message at index 5 — result precedes the call.
    tool_msg = _tool(2, "call-A")

    result = _validator().validate([ai_msg, tool_msg], turn_number=1)

    # The ToolMessage must be absent from the validated transcript.
    tool_msgs = [m for m in result.validated_messages if m.role == CanonicalRole.TOOL]
    assert tool_msgs == [], "ToolMessage appearing before its AIMessage must be dropped"
    # "call-A" had no surviving result, so the tool_call is a dangling call and is stripped.
    assert result.repair_actions == ["DANGLING_CALL: index 5"]
    assert "call-A" in result.dropped_call_ids


def test_drops_duplicate_tool_message_same_call_id() -> None:
    # Invariant 4 (NO-DUPLICATE-RESULTS): second ToolMessage with the same
    # tool_call_id must be dropped; only the first survives.
    messages = [
        _ai(0, [{"id": "dup-id", "name": "tool", "args": {}}]),
        _tool(1, "dup-id"),  # first result — kept
        _tool(2, "dup-id"),  # duplicate — must be dropped
    ]

    result = _validator().validate(messages, turn_number=1)

    tool_msgs = [m for m in result.validated_messages if m.role == CanonicalRole.TOOL]
    assert len(tool_msgs) == 1, "Only one ToolMessage per call_id must survive"
    assert tool_msgs[0].source_index == 1, "The first occurrence must be kept"
    assert "dup-id" in result.dropped_call_ids


def test_drops_tool_message_with_no_tool_call_id() -> None:
    # Invariant 8 (CALL-ID-REQUIRED): A ToolMessage with tool_call_id=None (or
    # empty-string) is structurally invalid and must be dropped immediately.
    tool_no_id = CanonicalMessage(
        role=CanonicalRole.TOOL,
        content="result with no id",
        tool_call_id=None,
        source_index=1,
        name="tool",
    )
    messages = [
        _ai(0, [{"id": "legit", "name": "tool", "args": {}}]),
        tool_no_id,
        _tool(2, "legit"),
    ]

    result = _validator().validate(messages, turn_number=1)

    # The id-less ToolMessage must not appear in the output.
    surviving_indices = [m.source_index for m in result.validated_messages]
    assert 1 not in surviving_indices, (
        "ToolMessage with no tool_call_id must be dropped"
    )
    # The legitimate pair must still be intact.
    assert any(
        m.role == CanonicalRole.TOOL and m.tool_call_id == "legit"
        for m in result.validated_messages
    )
