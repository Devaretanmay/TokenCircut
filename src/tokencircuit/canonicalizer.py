"""MessageCanonicalizer — normalizes heterogeneous message formats."""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from .types import CanonicalMessage, CanonicalRole

logger = logging.getLogger("tokencircuit.canonicalizer")

_ROLE_MAP: dict[str, CanonicalRole] = {
    "system": CanonicalRole.SYSTEM,
    "human": CanonicalRole.HUMAN,
    "user": CanonicalRole.HUMAN,
    "ai": CanonicalRole.AI,
    "assistant": CanonicalRole.AI,
    "tool": CanonicalRole.TOOL,
    "function": CanonicalRole.TOOL,
}


class MessageCanonicalizer:
    """
    Normalizes heterogeneous message formats into CanonicalMessage instances.

    Handles:
    - LangChain BaseMessage subclasses (HumanMessage, AIMessage, ToolMessage, etc.)
    - OpenAI-format dicts ({"role": "assistant", "content": "..."})
    - Mixed lists of both formats
    """

    def __init__(self) -> None:
        # Internal cache: id(msg) -> CanonicalMessage
        # This speeds up repeated canonicalization of the same message objects
        # within a single session.
        self._cache: dict[int, CanonicalMessage] = {}

    def canonicalize(self, messages: Sequence[Any]) -> list[CanonicalMessage]:
        result: list[CanonicalMessage] = []
        for idx, msg in enumerate(messages):
            msg_id = id(msg)
            if msg_id in self._cache:
                cached = self._cache[msg_id]
                result.append(CanonicalMessage(
                    role=cached.role,
                    content=cached.content,
                    tool_calls=list(cached.tool_calls) if cached.tool_calls else [],
                    tool_call_id=cached.tool_call_id,
                    source_index=idx,
                    name=cached.name
                ))
                continue

            canonical = self._convert_single(msg, idx)
            self._cache[msg_id] = canonical
            result.append(canonical)
        return result

    def _convert_single(self, msg: Any, index: int) -> CanonicalMessage:
        """Convert a single message (LangChain object or dict) to CanonicalMessage."""
        if isinstance(msg, dict):
            return self._from_dict(msg, index)
        return self._from_langchain(msg, index)

    def _from_dict(self, msg: dict[str, Any], index: int) -> CanonicalMessage:
        """Convert an OpenAI-format dict to CanonicalMessage."""
        role_str = str(msg.get("role", "ai")).lower()
        role = _ROLE_MAP.get(role_str, CanonicalRole.AI)

        content = msg.get("content", "") or ""
        if isinstance(content, list):
            # multimodal content blocks — extract text
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        tool_calls = self._extract_tool_calls_dict(msg)
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name")

        return CanonicalMessage(
            role=role,
            content=str(content),
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            source_index=index,
            name=name,
        )

    def _from_langchain(self, msg: Any, index: int) -> CanonicalMessage:
        """Convert a LangChain BaseMessage object to CanonicalMessage."""
        # Determine role from class name or type attribute
        msg_type = getattr(msg, "type", "") or ""
        class_name = type(msg).__name__.lower()

        if "human" in class_name or msg_type == "human":
            role = CanonicalRole.HUMAN
        elif "system" in class_name or msg_type == "system":
            role = CanonicalRole.SYSTEM
        elif "tool" in class_name or msg_type == "tool":
            role = CanonicalRole.TOOL
        else:
            role = CanonicalRole.AI

        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        # Extract tool_calls
        tool_calls: list[dict[str, Any]] = []
        raw_calls = getattr(msg, "tool_calls", None)
        if raw_calls:
            for tc in raw_calls:
                if isinstance(tc, dict):
                    tool_calls.append(self._normalize_tool_call(tc))
                elif hasattr(tc, "model_dump"):
                    tool_calls.append(self._normalize_tool_call(tc.model_dump()))
                elif hasattr(tc, "dict"):
                    tool_calls.append(self._normalize_tool_call(tc.dict()))

        tool_call_id = getattr(msg, "tool_call_id", None)
        name = getattr(msg, "name", None)

        return CanonicalMessage(
            role=role,
            content=str(content),
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            source_index=index,
            name=name,
        )

    def _extract_tool_calls_dict(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract and normalize tool_calls from a dict message."""
        raw = msg.get("tool_calls")
        if not raw:
            return []
        result: list[dict[str, Any]] = []
        for tc in raw:
            if isinstance(tc, dict):
                result.append(self._normalize_tool_call(tc))
        return result

    def _normalize_tool_call(self, tc: dict[str, Any]) -> dict[str, Any]:
        """Normalize a tool call dict to a consistent schema."""
        call_id = tc.get("id") or tc.get("call_id") or ""
        name = tc.get("name") or tc.get("function", {}).get("name", "unknown")

        # args can be in "args", "arguments", or "function.arguments"
        args = tc.get("args")
        if args is None:
            args = tc.get("arguments")
        if args is None:
            func = tc.get("function", {})
            args = func.get("arguments", {})

        # args might be a JSON string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"_raw": args}

        if not isinstance(args, dict):
            args = {"_raw": str(args)}

        return {"id": str(call_id), "name": str(name), "args": args}

    @staticmethod
    def to_openai_format(
        messages: list[CanonicalMessage]
    ) -> list[dict[str, Any]]:
        """Convert canonical messages back to OpenAI API format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            d: dict[str, Any] = {"role": MessageCanonicalizer._role_to_openai(msg.role)}

            if msg.content:
                d["content"] = msg.content
            elif msg.role != CanonicalRole.AI:
                d["content"] = ""

            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else str(tc["args"])
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                # OpenAI requires content to be present (can be null)
                if "content" not in d:
                    d["content"] = None

            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id

            if msg.name and msg.role == CanonicalRole.TOOL:
                d["name"] = msg.name

            result.append(d)
        return result

    @staticmethod
    def _role_to_openai(role: CanonicalRole) -> str:
        """Map CanonicalRole to OpenAI role string."""
        mapping = {
            CanonicalRole.SYSTEM: "system",
            CanonicalRole.HUMAN: "user",
            CanonicalRole.AI: "assistant",
            CanonicalRole.TOOL: "tool",
        }
        return mapping[role]
