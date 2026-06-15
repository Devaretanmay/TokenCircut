import hashlib
import json
from typing import Any, Optional

EXCLUDED_KEYS = frozenset({"timestamp", "trace_id", "_meta", "_tc_"})


def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def compute_state_hash(state: dict[str, Any]) -> str:
    filtered = {
        k: v
        for k, v in state.items()
        if not any(x in k.lower() for x in EXCLUDED_KEYS)
    }
    serialized = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        d = _to_dict(obj) if hasattr(obj, "dict") or hasattr(obj, "model_dump") else obj
        return {k: _serializable(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_serializable(item) for item in obj]
    if hasattr(obj, "dict"):
        return _serializable(obj.dict())
    if hasattr(obj, "model_dump"):
        return _serializable(obj.model_dump())
    return str(obj)


def extract_tool_type_signature(tool_call: Optional[dict[str, Any]]) -> str:
    if tool_call is None:
        return "NO_TOOL_CALL"
    name = tool_call.get("name", "unknown")
    args = tool_call.get("args", {})
    if hasattr(args, "dict") or hasattr(args, "model_dump"):
        d = _serializable(args)
    else:
        d = args if isinstance(args, dict) else {}
    arg_types = ",".join(type(v).__name__ for v in d.values())
    return f"{name}({arg_types})"


def compute_action_hash(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    if not messages:
        return compute_state_hash(state)

    tool_call = None
    tool_content = None
    tool_call_id = None

    for msg in reversed(messages):
        d = _to_dict(msg) if not isinstance(msg, dict) else msg
        if d.get("role") == "tool" or "tool_call_id" in d:
            if tool_content is None and "content" in d:
                tool_content = str(d["content"])
            if tool_call_id is None and "tool_call_id" in d:
                tool_call_id = d["tool_call_id"]
        if d.get("type") == "ai" or "tool_calls" in d:
            tcs = d.get("tool_calls", [])
            if tcs:
                tc = tcs[-1]
                if tool_call_id is None or tc.get("id") == tool_call_id:
                    tool_call = tc
                    break

    fingerprint = {}
    if tool_call:
        tc_d = _to_dict(tool_call) if not isinstance(tool_call, dict) else tool_call
        fingerprint["tool_name"] = tc_d.get("name", "unknown")
        fingerprint["tool_args"] = _serializable(tc_d.get("args", {}))
    if tool_content is not None:
        stable = tool_content[-200:]
        stable = "".join(ch for ch in stable if not ch.isdigit())
        fingerprint["tool_result"] = stable

    if not fingerprint:
        return compute_state_hash(state)

    serialized = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


__all__ = [
    "compute_state_hash",
    "compute_action_hash",
    "extract_tool_type_signature",
]
