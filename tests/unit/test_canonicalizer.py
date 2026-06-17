"""Comprehensive unit tests for MessageCanonicalizer."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from tokencircuit.canonicalizer import MessageCanonicalizer
from tokencircuit.types import CanonicalMessage, CanonicalRole

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canonicalizer() -> MessageCanonicalizer:
    """Return a fresh MessageCanonicalizer instance for each test."""
    return MessageCanonicalizer()


# ---------------------------------------------------------------------------
# Helpers — fake LangChain messages (no real langchain dependency)
# ---------------------------------------------------------------------------


def _make_langchain_msg(
    cls_name: str,
    content: Any = "",
    *,
    type_attr: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
    name: str | None = None,
) -> Any:
    """Create a dummy object that behaves like a LangChain BaseMessage subclass."""
    cls = type(cls_name, (), {})
    mock = cls()
    mock.content = content
    if type_attr is not None:
        mock.type = type_attr
    if tool_calls is not None:
        mock.tool_calls = tool_calls
    if tool_call_id is not None:
        mock.tool_call_id = tool_call_id
    if name is not None:
        mock.name = name
    return mock


# ===================================================================
# 1. Dict-format messages (OpenAI format) → canonical conversion
# ===================================================================


class TestDictToCanonical:
    """Tests for converting OpenAI-format dict messages to CanonicalMessage."""

    def test_basic_user_message(self, canonicalizer: MessageCanonicalizer) -> None:
        """A simple user dict should convert to HUMAN role with matching content."""
        msgs = [{"role": "user", "content": "Hello!"}]
        result = canonicalizer.canonicalize(msgs)
        assert len(result) == 1
        assert result[0].role == CanonicalRole.HUMAN
        assert result[0].content == "Hello!"
        assert result[0].source_index == 0

    def test_basic_assistant_message(self, canonicalizer: MessageCanonicalizer) -> None:
        """An assistant dict should map to AI role."""
        msgs = [{"role": "assistant", "content": "Hi there."}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].role == CanonicalRole.AI
        assert result[0].content == "Hi there."

    def test_system_message(self, canonicalizer: MessageCanonicalizer) -> None:
        """A system dict should map to SYSTEM role."""
        msgs = [{"role": "system", "content": "You are helpful."}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].role == CanonicalRole.SYSTEM

    def test_multiple_messages_preserve_order(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Multiple messages should be returned in the same order with correct indices."""  # noqa: E501
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        result = canonicalizer.canonicalize(msgs)
        assert len(result) == 3
        assert [m.role for m in result] == [
            CanonicalRole.SYSTEM,
            CanonicalRole.HUMAN,
            CanonicalRole.AI,
        ]
        assert [m.source_index for m in result] == [0, 1, 2]


# ===================================================================
# 2. Role mapping coverage
# ===================================================================


class TestRoleMappings:
    """Tests for all supported role string → CanonicalRole mappings."""

    @pytest.mark.parametrize(
        "role_str, expected_role",
        [
            ("user", CanonicalRole.HUMAN),
            ("human", CanonicalRole.HUMAN),
            ("assistant", CanonicalRole.AI),
            ("ai", CanonicalRole.AI),
            ("system", CanonicalRole.SYSTEM),
            ("tool", CanonicalRole.TOOL),
            ("function", CanonicalRole.TOOL),
        ],
    )
    def test_known_role_mapping(
        self,
        canonicalizer: MessageCanonicalizer,
        role_str: str,
        expected_role: CanonicalRole,
    ) -> None:
        """Each known role string should map to the correct CanonicalRole."""
        msgs = [{"role": role_str, "content": "x"}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].role == expected_role

    def test_unknown_role_defaults_to_ai(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """An unrecognized role string should default to AI."""
        msgs = [{"role": "developer", "content": "x"}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].role == CanonicalRole.AI

    def test_role_case_insensitive(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Role strings should be matched case-insensitively."""
        msgs = [{"role": "USER", "content": "x"}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].role == CanonicalRole.HUMAN

    def test_missing_role_defaults_to_ai(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A dict with no 'role' key should default to AI."""
        msgs = [{"content": "orphan"}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].role == CanonicalRole.AI


# ===================================================================
# 3. Tool call extraction and normalization from dicts
# ===================================================================


class TestToolCallExtractionDict:
    """Tests for extracting and normalizing tool_calls from dict messages."""

    def test_basic_tool_call_extraction(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """tool_calls in OpenAI format should be extracted and normalized."""
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert len(result[0].tool_calls) == 1
        tc = result[0].tool_calls[0]
        assert tc["id"] == "call_123"
        assert tc["name"] == "get_weather"
        assert tc["args"] == {"city": "NYC"}

    def test_multiple_tool_calls(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Multiple tool_calls in a single message should all be normalized."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "name": "foo", "args": {"x": 1}},
                    {"id": "b", "name": "bar", "args": {"y": 2}},
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert len(result[0].tool_calls) == 2
        assert result[0].tool_calls[0]["name"] == "foo"
        assert result[0].tool_calls[1]["name"] == "bar"

    def test_tool_call_id_on_tool_result(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A tool result message should carry its tool_call_id."""
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "get_weather",
                "content": "Sunny, 72°F",
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].role == CanonicalRole.TOOL
        assert result[0].tool_call_id == "call_123"
        assert result[0].name == "get_weather"
        assert result[0].content == "Sunny, 72°F"


# ===================================================================
# 4. Multimodal content blocks
# ===================================================================


class TestMultimodalContent:
    """Tests for list-of-blocks multimodal content handling."""

    def test_text_blocks_joined(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Multiple text blocks should be joined with newlines."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Line 1"},
                    {"type": "text", "text": "Line 2"},
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].content == "Line 1\nLine 2"

    def test_image_blocks_ignored(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Image blocks should be silently dropped; only text extracted."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "https://..."}},
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].content == "Look at this"

    def test_plain_string_blocks(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Plain strings inside a content list should be included."""
        msgs = [
            {
                "role": "user",
                "content": ["raw string block"],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].content == "raw string block"

    def test_mixed_string_and_dict_blocks(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A mix of string blocks and text-dict blocks should all be joined."""
        msgs = [
            {
                "role": "user",
                "content": [
                    "prefix",
                    {"type": "text", "text": "middle"},
                    "suffix",
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].content == "prefix\nmiddle\nsuffix"

    def test_empty_content_list(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """An empty content list should produce an empty string."""
        msgs = [{"role": "user", "content": []}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].content == ""


# ===================================================================
# 5. LangChain BaseMessage conversion (mocked)
# ===================================================================


class TestLangChainConversion:
    """Tests for converting mocked LangChain BaseMessage objects."""

    def test_human_message_by_class_name(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A class named 'HumanMessage' should map to HUMAN role."""
        mock = _make_langchain_msg("HumanMessage", content="hello from user")
        result = canonicalizer.canonicalize([mock])
        assert result[0].role == CanonicalRole.HUMAN
        assert result[0].content == "hello from user"

    def test_ai_message_by_class_name(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A class named 'AIMessage' should map to AI role."""
        mock = _make_langchain_msg("AIMessage", content="response")
        result = canonicalizer.canonicalize([mock])
        assert result[0].role == CanonicalRole.AI

    def test_system_message_by_class_name(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A class named 'SystemMessage' should map to SYSTEM role."""
        mock = _make_langchain_msg("SystemMessage", content="be helpful")
        result = canonicalizer.canonicalize([mock])
        assert result[0].role == CanonicalRole.SYSTEM

    def test_tool_message_by_class_name(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A class named 'ToolMessage' should map to TOOL role."""
        mock = _make_langchain_msg(
            "ToolMessage",
            content="result",
            tool_call_id="tc_1",
            name="my_tool",
        )
        result = canonicalizer.canonicalize([mock])
        assert result[0].role == CanonicalRole.TOOL
        assert result[0].tool_call_id == "tc_1"
        assert result[0].name == "my_tool"

    def test_role_detected_by_type_attr(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """If the class name is generic, 'type' attr should determine role."""
        mock = _make_langchain_msg(
            "BaseMessage", content="x", type_attr="human"
        )
        result = canonicalizer.canonicalize([mock])
        assert result[0].role == CanonicalRole.HUMAN

    def test_system_detected_by_type_attr(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """type='system' on an otherwise generic class should yield SYSTEM."""
        mock = _make_langchain_msg(
            "BaseMessage", content="sys", type_attr="system"
        )
        result = canonicalizer.canonicalize([mock])
        assert result[0].role == CanonicalRole.SYSTEM

    def test_tool_detected_by_type_attr(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """type='tool' on a generic class should yield TOOL."""
        mock = _make_langchain_msg(
            "BaseMessage", content="res", type_attr="tool"
        )
        result = canonicalizer.canonicalize([mock])
        assert result[0].role == CanonicalRole.TOOL

    def test_unknown_langchain_defaults_to_ai(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """An unknown class name with no relevant type attr should default to AI."""
        mock = _make_langchain_msg("CustomMessage", content="x")
        result = canonicalizer.canonicalize([mock])
        assert result[0].role == CanonicalRole.AI

    def test_langchain_with_tool_calls_dict(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """LangChain message with dict-based tool_calls should be normalized."""
        mock = _make_langchain_msg(
            "AIMessage",
            content="",
            tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "hi"}}],
        )
        result = canonicalizer.canonicalize([mock])
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["name"] == "search"
        assert result[0].tool_calls[0]["args"] == {"q": "hi"}

    def test_langchain_with_model_dump_tool_calls(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Tool calls with model_dump() (pydantic v2 style) should be handled."""
        tc_mock = MagicMock()
        # Not a dict, has model_dump
        tc_mock.__class__ = type("ToolCall", (), {})
        tc_mock.model_dump.return_value = {
            "id": "tc2",
            "name": "compute",
            "args": {"x": 42},
        }
        # Make isinstance check fail for dict
        del tc_mock.__getitem__
        mock = _make_langchain_msg("AIMessage", content="", tool_calls=[tc_mock])
        result = canonicalizer.canonicalize([mock])
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["name"] == "compute"

    def test_langchain_multimodal_content(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """LangChain messages with list content should extract text parts."""
        mock = _make_langchain_msg(
            "HumanMessage",
            content=[
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        )
        result = canonicalizer.canonicalize([mock])
        assert result[0].content == "Describe this image"


# ===================================================================
# 6. Tool call normalization — field extraction variants
# ===================================================================


class TestToolCallNormalization:
    """Tests for _normalize_tool_call field extraction from different schemas."""

    def test_id_from_id_field(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """'id' field should be used as the call id."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "abc", "name": "fn", "args": {}}],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["id"] == "abc"

    def test_id_from_call_id_field(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """'call_id' field should be used as the call id when 'id' is missing."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"call_id": "xyz", "name": "fn", "args": {}}],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["id"] == "xyz"

    def test_id_fallback_to_empty(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Missing both 'id' and 'call_id' should result in an empty string."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"name": "fn", "args": {}}],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["id"] == ""

    def test_name_from_name_field(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """'name' field should be used directly."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "1", "name": "my_func", "args": {}}],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["name"] == "my_func"

    def test_name_from_function_name(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """'function.name' should be used when 'name' is absent."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": "deep_func", "arguments": "{}"},
                    }
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["name"] == "deep_func"

    def test_name_fallback_to_unknown(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Missing name in all locations should default to 'unknown'."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "1", "args": {}}],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["name"] == "unknown"

    def test_args_from_args_field(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """'args' dict should be used directly."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "fn", "args": {"a": 1, "b": 2}}
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"a": 1, "b": 2}

    def test_args_from_arguments_field(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """'arguments' field should be used when 'args' is absent."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "fn", "arguments": {"k": "v"}}
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"k": "v"}

    def test_args_from_function_arguments(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """'function.arguments' should be used as last resort."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "fn",
                        "function": {"arguments": '{"deep": true}'},
                    }
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"deep": True}


# ===================================================================
# 7. JSON string args parsing
# ===================================================================


class TestJsonStringArgsParsing:
    """Tests for automatic JSON parsing of string-typed arguments."""

    def test_valid_json_string_parsed(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A valid JSON string in 'args' should be parsed to a dict."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "fn", "args": '{"key": "value"}'}
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"key": "value"}

    def test_valid_json_in_arguments(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A valid JSON string in 'arguments' should also be parsed."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "fn", "arguments": '{"n": 42}'}
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"n": 42}

    def test_complex_json_object(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Complex nested JSON should be parsed correctly."""
        nested = {"items": [1, 2, 3], "meta": {"nested": True}}
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "fn", "args": json.dumps(nested)}
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == nested


# ===================================================================
# 8. Invalid JSON args → {"_raw": ...} fallback
# ===================================================================


class TestInvalidJsonArgsFallback:
    """Tests for fallback behavior when args contain invalid JSON."""

    def test_invalid_json_wrapped_in_raw(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Invalid JSON string should be wrapped in {'_raw': ...}."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "fn", "args": "not json {{{"}
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"_raw": "not json {{{"}

    def test_plain_text_string_wrapped(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A plain text string (not JSON) should be wrapped in {'_raw': ...}."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "fn", "args": "hello world"}
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"_raw": "hello world"}

    def test_non_dict_non_string_args_wrapped(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Non-dict, non-string args (e.g., a list) should be wrapped in {'_raw': str(...)}."""  # noqa: E501
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "1", "name": "fn", "args": [1, 2, 3]}
                ],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"_raw": "[1, 2, 3]"}

    def test_integer_args_wrapped(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Integer args should be wrapped in {'_raw': str(...)}."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "1", "name": "fn", "args": 42}],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls[0]["args"] == {"_raw": "42"}


# ===================================================================
# 9. Cache behavior
# ===================================================================


class TestCacheBehavior:
    """Tests for internal caching of canonicalized messages."""

    def test_same_object_returns_cached_result(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Canonicalizing the same message object twice should use the cache."""
        msg = {"role": "user", "content": "hello"}
        result1 = canonicalizer.canonicalize([msg])
        result2 = canonicalizer.canonicalize([msg])
        # Both should produce equivalent results
        assert result1[0].role == result2[0].role
        assert result1[0].content == result2[0].content

    def test_cached_result_updates_source_index(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A cached message reused at a different index should have the new source_index."""  # noqa: E501
        msg = {"role": "user", "content": "hello"}
        # First call — msg at index 0
        r1 = canonicalizer.canonicalize([msg])
        assert r1[0].source_index == 0
        # Second call — msg at index 1 (preceded by another message)
        other = {"role": "system", "content": "sys"}
        r2 = canonicalizer.canonicalize([other, msg])
        assert r2[1].source_index == 1

    def test_cache_does_not_share_tool_calls_list(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Cached results should copy tool_calls to avoid cross-contamination."""
        msg = {
            "role": "assistant",
            "tool_calls": [{"id": "1", "name": "fn", "args": {}}],
        }
        r1 = canonicalizer.canonicalize([msg])
        r2 = canonicalizer.canonicalize([msg])
        # Mutating one should not affect the other
        r1[0].tool_calls.append({"id": "extra", "name": "hack", "args": {}})
        assert len(r2[0].tool_calls) == 1

    def test_cache_populated_after_first_call(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """The internal cache should have entries after canonicalization."""
        msg = {"role": "user", "content": "x"}
        assert len(canonicalizer._cache) == 0
        canonicalizer.canonicalize([msg])
        assert len(canonicalizer._cache) == 1

    def test_different_objects_same_content_not_cached_together(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Two distinct dict objects with the same content should be cached separately."""  # noqa: E501
        msg1 = {"role": "user", "content": "same"}
        msg2 = {"role": "user", "content": "same"}
        canonicalizer.canonicalize([msg1, msg2])
        # They have different id(), so cache should hold 2 entries
        assert len(canonicalizer._cache) == 2


# ===================================================================
# 10. to_openai_format() round-trip
# ===================================================================


class TestToOpenAIFormat:
    """Tests for converting canonical messages back to OpenAI dict format."""

    def test_simple_roundtrip(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A simple message should survive canonicalize → to_openai_format."""
        original = [{"role": "user", "content": "Hello"}]
        canonical = canonicalizer.canonicalize(original)
        openai_fmt = canonicalizer.to_openai_format(canonical)
        assert openai_fmt[0]["role"] == "user"
        assert openai_fmt[0]["content"] == "Hello"

    def test_system_roundtrip(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """System messages should round-trip correctly."""
        original = [{"role": "system", "content": "Be concise."}]
        canonical = canonicalizer.canonicalize(original)
        openai_fmt = canonicalizer.to_openai_format(canonical)
        assert openai_fmt[0]["role"] == "system"
        assert openai_fmt[0]["content"] == "Be concise."

    def test_assistant_with_tool_calls_roundtrip(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Assistant messages with tool_calls should round-trip correctly."""
        original = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            }
        ]
        canonical = canonicalizer.canonicalize(original)
        openai_fmt = canonicalizer.to_openai_format(canonical)
        assert openai_fmt[0]["role"] == "assistant"
        assert len(openai_fmt[0]["tool_calls"]) == 1
        tc = openai_fmt[0]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        # arguments should be a JSON string
        parsed_args = json.loads(tc["function"]["arguments"])
        assert parsed_args == {"q": "test"}

    def test_tool_result_roundtrip(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Tool result messages should preserve tool_call_id and name."""
        original = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search",
                "content": "results here",
            }
        ]
        canonical = canonicalizer.canonicalize(original)
        openai_fmt = canonicalizer.to_openai_format(canonical)
        assert openai_fmt[0]["role"] == "tool"
        assert openai_fmt[0]["tool_call_id"] == "call_1"
        assert openai_fmt[0]["name"] == "search"
        assert openai_fmt[0]["content"] == "results here"

    def test_role_mapping_back(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Each canonical role should map back to the correct OpenAI role string."""
        canonical = [
            CanonicalMessage(role=CanonicalRole.SYSTEM, content="a"),
            CanonicalMessage(role=CanonicalRole.HUMAN, content="b"),
            CanonicalMessage(role=CanonicalRole.AI, content="c"),
            CanonicalMessage(role=CanonicalRole.TOOL, content="d", tool_call_id="x"),
        ]
        openai_fmt = canonicalizer.to_openai_format(canonical)
        assert [m["role"] for m in openai_fmt] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]

    def test_ai_message_without_content_omits_content_key(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """An AI message with empty content and no tool_calls should omit 'content'."""
        canonical = [CanonicalMessage(role=CanonicalRole.AI, content="")]
        openai_fmt = canonicalizer.to_openai_format(canonical)
        assert "content" not in openai_fmt[0]

    def test_non_ai_empty_content_gets_empty_string(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A non-AI message with empty content should have content=''."""
        canonical = [CanonicalMessage(role=CanonicalRole.HUMAN, content="")]
        openai_fmt = canonicalizer.to_openai_format(canonical)
        assert openai_fmt[0]["content"] == ""

    def test_ai_with_tool_calls_and_no_content_sets_null(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """An AI message with tool_calls but no content should set content to None."""
        canonical = [
            CanonicalMessage(
                role=CanonicalRole.AI,
                content="",
                tool_calls=[{"id": "1", "name": "fn", "args": {"a": 1}}],
            )
        ]
        openai_fmt = canonicalizer.to_openai_format(canonical)
        assert openai_fmt[0]["content"] is None
        assert len(openai_fmt[0]["tool_calls"]) == 1

    def test_name_only_on_tool_role(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """The 'name' field should only appear for TOOL role messages."""
        # AI message with name should NOT include name in output
        canonical_ai = [
            CanonicalMessage(role=CanonicalRole.AI, content="x", name="fn")
        ]
        openai_ai = canonicalizer.to_openai_format(canonical_ai)
        assert "name" not in openai_ai[0]

        # TOOL message with name SHOULD include it
        canonical_tool = [
            CanonicalMessage(
                role=CanonicalRole.TOOL,
                content="res",
                tool_call_id="tc1",
                name="fn",
            )
        ]
        openai_tool = canonicalizer.to_openai_format(canonical_tool)
        assert openai_tool[0]["name"] == "fn"


# ===================================================================
# 12. Edge cases
# ===================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_content_string(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A message with an empty content string should produce content=''."""
        msgs = [{"role": "user", "content": ""}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].content == ""

    def test_none_content(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A message with content=None should produce content=''."""
        msgs = [{"role": "assistant", "content": None}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].content == ""

    def test_missing_content_key(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A dict without a 'content' key should produce content=''."""
        msgs = [{"role": "user"}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].content == ""

    def test_empty_tool_calls_list(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """An empty tool_calls list should result in no tool_calls."""
        msgs = [{"role": "assistant", "content": "x", "tool_calls": []}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls == []

    def test_tool_calls_none(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """tool_calls=None should result in an empty tool_calls list."""
        msgs = [{"role": "assistant", "content": "x", "tool_calls": None}]
        result = canonicalizer.canonicalize(msgs)
        assert result[0].tool_calls == []

    def test_empty_message_sequence(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """An empty sequence should return an empty list."""
        result = canonicalizer.canonicalize([])
        assert result == []

    def test_tool_call_with_empty_function_dict(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A tool_call with an empty 'function' dict should use defaults."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "1", "function": {}}],
            }
        ]
        result = canonicalizer.canonicalize(msgs)
        tc = result[0].tool_calls[0]
        assert tc["name"] == "unknown"
        assert tc["args"] == {}

    def test_large_content_truncation_in_error_fallback(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """Error fallback should truncate content to 500 chars."""
        # Create something that will fail conversion and has a large repr
        class LargeRepr:
            def __repr__(self) -> str:
                return "X" * 1000
        broken = LargeRepr()
        result = canonicalizer.canonicalize([broken])
        assert len(result[0].content) <= 500

    def test_canonical_message_repr(self) -> None:
        """CanonicalMessage.__repr__ should contain role and content preview."""
        msg = CanonicalMessage(
            role=CanonicalRole.HUMAN,
            content="Hello world, this is a test message",
            tool_call_id="tc1",
        )
        r = repr(msg)
        assert "human" in r
        assert "Hello world" in r
        assert "tc1" in r

    def test_canonical_message_repr_with_tool_calls(self) -> None:
        """CanonicalMessage repr should show tool_calls count."""
        msg = CanonicalMessage(
            role=CanonicalRole.AI,
            tool_calls=[
                {"id": "1", "name": "a", "args": {}},
                {"id": "2", "name": "b", "args": {}},
            ],
        )
        r = repr(msg)
        assert "tool_calls=2" in r

    def test_mixed_dict_and_langchain_messages(
        self, canonicalizer: MessageCanonicalizer
    ) -> None:
        """A mixed list of dict and LangChain-like messages should all convert."""
        dict_msg = {"role": "user", "content": "from dict"}
        lc_msg = _make_langchain_msg("AIMessage", content="from langchain")
        result = canonicalizer.canonicalize([dict_msg, lc_msg])
        assert len(result) == 2
        assert result[0].role == CanonicalRole.HUMAN
        assert result[0].content == "from dict"
        assert result[1].role == CanonicalRole.AI
        assert result[1].content == "from langchain"



