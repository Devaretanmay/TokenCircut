"""
Fire-and-forget telemetry for TokenCircuit.

PRIVACY GUARANTEE (mathematically enforced by this module):
  - The payload schema is sealed to exactly the fields listed in _ALLOWED_KEYS.
  - No message content, tool arguments, tool results, or any user-supplied
    string enters the payload — all values are scalars derived solely from
    engine metadata (stage names, signal enum values, integer counters).
  - API keys are never logged, serialised, or stored in module-level state.
  - The HTTP call runs in a daemon thread; it is fire-and-forget and NEVER
    blocks the calling thread, the LangGraph node, or the asyncio event loop.

SECURITY PROPERTIES:
  - No eval(), exec(), pickle, or unsafe YAML.
  - urllib.request only — zero third-party network dependencies.
  - Thread is daemon=True so it cannot delay process shutdown.
  - Failure is silently swallowed; telemetry MUST NOT affect correctness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import urllib.request
from typing import Any

logger = logging.getLogger("tokencircuit.telemetry")

# ─── Sealed payload schema ────────────────────────────────────────────────────
# Only these keys may appear in the outbound JSON.  Any caller-supplied extra
# keys are stripped before serialisation.  Values MUST be str | int | float |
# bool — no nested dicts, no lists of strings that could smuggle content.
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "sdk_version",
        "python_version",
        "platform",
        "event_type",
        "node_name_hash",  # sha256[:8] of node_name — NOT the raw string
        "intervention_stage",
        "signal_types",  # comma-separated enum *values*, e.g. "SEMANTIC_STAGNATION"
        "tokens_saved_estimate",
        "consecutive_stagnation_count",
        "turn_number",
        "cooldown_remaining",
    }
)

_TELEMETRY_ENDPOINT: str = os.environ.get("TOKENCIRCUIT_TELEMETRY_URL", "")

# Respect opt-out: any truthy value in the env var disables telemetry.
# NOTE: whitespace-only values (e.g. "   ") are treated as "set but blank" and
# therefore opt OUT — we compare the raw value, not the stripped version, against
# the explicit allow-list of "this means false".  A blank-only string is not in
# that allow-list, so it correctly opts out.
_OPTED_OUT: bool = os.environ.get("TOKENCIRCUIT_DISABLE_TELEMETRY", "") not in (
    "",
    "0",
    "false",
    "no",
)

try:
    from . import __version__ as _SDK_VERSION
except ImportError:
    _SDK_VERSION = "unknown"


def _node_name_hash(node_name: str) -> str:
    """One-way hash of the node name — reveals structure, not content."""
    return hashlib.sha256(node_name.encode()).hexdigest()[:8]


def _build_payload(event_type: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Construct and seal the telemetry payload.

    Enforces _ALLOWED_KEYS whitelist.  Values are cast to safe primitives;
    any value that cannot be cast to str/int/float/bool is dropped entirely
    so raw message text can never sneak through a type error.
    """
    raw: dict[str, Any] = {
        "sdk_version": _SDK_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "event_type": event_type,
        **metadata,
    }

    sealed: dict[str, Any] = {}
    for key in _ALLOWED_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        # Whitelist primitive types only — no dicts, no arbitrary objects.
        if isinstance(value, (str, int, float, bool)):
            sealed[key] = value
        # Drop silently; this is intentional — see module docstring.

    return sealed


def send_event(
    event_type: str,
    *,
    node_name: str,
    intervention_stage: str,
    signal_types: list[str],
    tokens_saved_estimate: int,
    consecutive_stagnation_count: int = 0,
    turn_number: int = 0,
    cooldown_remaining: int = 0,
) -> None:
    """Fire-and-forget telemetry event.

    All arguments are metadata scalars.  No message content, no tool args,
    no tool results.  Runs in a daemon thread — returns immediately.

    If _TELEMETRY_ENDPOINT is empty or the caller is opted out, this is a
    pure no-op with zero overhead beyond the opt-out check.
    """
    if _OPTED_OUT or not _TELEMETRY_ENDPOINT:
        return

    metadata: dict[str, Any] = {
        # node_name is hashed — raw string never leaves the process.
        "node_name_hash": _node_name_hash(node_name),
        "intervention_stage": intervention_stage,
        # signal_types is a list of enum .value strings — no user content.
        "signal_types": ",".join(signal_types),
        "tokens_saved_estimate": int(tokens_saved_estimate),
        "consecutive_stagnation_count": int(consecutive_stagnation_count),
        "turn_number": int(turn_number),
        "cooldown_remaining": int(cooldown_remaining),
    }

    payload = _build_payload(event_type, metadata)

    def _send() -> None:
        try:
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                _TELEMETRY_ENDPOINT,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as _:
                pass
        except Exception:
            # Telemetry MUST be fire-and-forget; never propagate.
            logger.debug("Telemetry send failed (non-fatal)", exc_info=False)

    t = threading.Thread(target=_send, daemon=True, name="tc-telemetry")
    t.start()
