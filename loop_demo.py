"""
TokenCircuit Loop Demo — shows the Progressive Intervention Protocol in action.

Records a terminal animation: agent loops on a 403 error → TokenCircuit
detects stagnation → NUDGE → OVERRIDE → graceful agent pivot.

Usage:
    python loop_demo.py
"""

import logging
from time import sleep
from typing import Any

from tokencircuit import InterventionConfig
from tokencircuit.engine import InterventionEngine
from tokencircuit.types import InterventionStage

logging.basicConfig(
    level=logging.INFO,
    format=" %(message)s",
    force=True,
)
logging.getLogger("tokencircuit.engine").setLevel(logging.INFO)


AGENT_CALLS = [
    {"role": "user", "content": "Fetch the user list from /admin/users."},
    {"role": "assistant", "content": "Let me try fetching that data.",
     "tool_calls": [{"name": "fetch_sensitive_data", "args": {"endpoint": "/admin/users"}, "id": "call_0", "type": "tool_call"}]},
    {"role": "tool", "content": "Error 403: API access forbidden. Your API key does not have access.", "tool_call_id": "call_0"},
    {"role": "assistant", "content": "Let me try fetching that data.",
     "tool_calls": [{"name": "fetch_sensitive_data", "args": {"endpoint": "/admin/users"}, "id": "call_1", "type": "tool_call"}]},
    {"role": "tool", "content": "Error 403: API access forbidden. Your API key does not have access.", "tool_call_id": "call_1"},
    {"role": "assistant", "content": "Let me try fetching that data.",
     "tool_calls": [{"name": "fetch_sensitive_data", "args": {"endpoint": "/admin/users"}, "id": "call_2", "type": "tool_call"}]},
    {"role": "tool", "content": "Error 403: API access forbidden. Your API key does not have access.", "tool_call_id": "call_2"},
    {"role": "assistant", "content": "Let me try fetching that data.",
     "tool_calls": [{"name": "fetch_sensitive_data", "args": {"endpoint": "/admin/users"}, "id": "call_3", "type": "tool_call"}]},
    {"role": "tool", "content": "Error 403: API access forbidden. Your API key does not have access.", "tool_call_id": "call_3"},
    {"role": "assistant", "content": "Let me try fetching that data.",
     "tool_calls": [{"name": "fetch_sensitive_data", "args": {"endpoint": "/admin/users"}, "id": "call_4", "type": "tool_call"}]},
    {"role": "tool", "content": "Error 403: API access forbidden. Your API key does not have access.", "tool_call_id": "call_4"},
    {"role": "assistant", "content": "Let me try fetching that data.",
     "tool_calls": [{"name": "fetch_sensitive_data", "args": {"endpoint": "/admin/users"}, "id": "call_5", "type": "tool_call"}]},
    {"role": "tool", "content": "Error 403: API access forbidden. Your API key does not have access.", "tool_call_id": "call_5"},
]


STAGE_LABELS = {
    InterventionStage.PASS: " PASS ",
    InterventionStage.NUDGE: "NUDGE",
    InterventionStage.OVERRIDE: "OVERRIDE",
    InterventionStage.HARD_STOP: "HARD_STOP",
}


def main():
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║   TokenCircuit  —  Pre-Model Intervention Engine          ║")
    print("  ║   Agent hits 403 → loops → OVERRIDE → graceful summary     ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  ●  User: Fetch the user list from /admin/users.")
    print()

    config = InterventionConfig(
        nudge_threshold=2,
        override_threshold=4,
        hard_stop_threshold=8,
        window_size=5,
        similarity_threshold=0.50,
        enable_semantic_detection=True,
        enable_transcript_validation=False,
    )
    engine = InterventionEngine(config=config)
    tc_state: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    turn = 0

    # Run through the pre-recorded agent turns
    for entry in AGENT_CALLS:
        messages.append(entry)
        if entry["role"] != "tool":
            continue

        turn += 1
        decision = engine.process(
            messages=messages,
            state={"_tc_intervention": tc_state},
            thread_id="demo_1",
            node_name="llm",
        )
        tc_state = dict(decision.state_patch)

        label = STAGE_LABELS[decision.stage]
        coach = decision.coaching_message or ""
        if coach:
            coach = " " + coach[:80]

        print(f"  ── Turn {turn} ─────────────────────────────────────")
        if decision.stage == InterventionStage.PASS:
            print(f"     Agent called fetch_sensitive_data...")
            print(f"     → Error 403: Forbidden.")
            print(f"     [{label}]  building evidence ({'no signals' if not decision.signals else decision.signals[0].value})")
        elif decision.stage == InterventionStage.NUDGE:
            print(f"     Agent called fetch_sensitive_data...")
            print(f"     → Error 403: Forbidden.")
            print(f"     ⚡ [{label}]  {coach}")
        elif decision.stage == InterventionStage.OVERRIDE:
            print(f"     Agent called fetch_sensitive_data...")
            print(f"     → Error 403: Forbidden.")
            print(f"     🔴 [{label}]  {coach}")
            print(f"     → Agent reads the override directive and pivots...")
            print(f"     → AI: I was unable to access the admin API due to")
            print(f"            insufficient permissions (HTTP 403). Based on")
            print(f"            the errors, the endpoint requires elevated")
            print(f"            credentials. Recommended action: request")
            print(f"            read-only access from your administrator.")
            break
        elif decision.stage == InterventionStage.HARD_STOP:
            print(f"     ✖ [{label}]  {coach}")
            break

        sleep(0.15)

    print()
    print("  ✓  Agent completed gracefully — TokenCircuit prevented runaway.")
    print()


if __name__ == "__main__":
    main()
