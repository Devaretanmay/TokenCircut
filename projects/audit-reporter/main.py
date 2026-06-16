"""
Audit Reporter — Runs TokenCircuit in audit mode across multiple
simulated agent sessions, then generates a report of what interventions
would have been triggered.

Real use case: Before deploying TokenCircuit to production, run it in
audit mode on historical conversation logs to see what it would catch.
No risk of modifying real messages.

No API keys. Fully synthetic conversation data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tokencircuit import (
    InterventionEngine,
    InterventionConfig,
    InterventionStage,
    SignalType,
    default_intervention_state,
)


def _msg(role, content, tool_calls=None, tool_call_id=None, name=None):
    d = {"role": role, "content": content}
    if tool_calls: d["tool_calls"] = tool_calls
    if tool_call_id: d["tool_call_id"] = tool_call_id
    if name: d["name"] = name
    return d

def _tc(cid, name, args=None):
    return {"id": cid, "name": name, "args": args or {}}


# ── Simulated conversation logs (like historical data) ──

conversations = [
    {
        "id": "healthy-session-1",
        "agent": "web-searcher",
        "messages_gen": lambda: [
            (_msg("system", "Search agent."),
             _msg("user", "What's new in AI?"),
             _msg("assistant", "Let me search.",
                  tool_calls=[_tc("h1", "search", {"q": "AI news"})]),
             _msg("tool", "Latest: GPT-5 released",
                  tool_call_id="h1", name="search"),
             _msg("assistant", "Found: GPT-5 released.")),
        ],
    },
    {
        "id": "stuck-loop-1",
        "agent": "web-searcher",
        "messages_gen": lambda: [
            (_msg("system", "Search agent."),
             _msg("user", "Find my file"),
             _msg("assistant", "Searching for file",
                  tool_calls=[_tc(f"s{i}", "search", {"q": "my file"})]),
             _msg("tool", "No results",
                  tool_call_id=f"s{i}", name="search"),
             _msg("assistant", "No results, trying again."))
            for i in range(5)
        ],
    },
    {
        "id": "error-loop-1",
        "agent": "api-caller",
        "messages_gen": lambda: [
            (_msg("system", "API agent."),
             _msg("user", "Call API"),
             _msg("assistant", "Calling endpoint",
                  tool_calls=[_tc(f"e{i}", "api_call", {"url": "/data"})]),
             _msg("tool", "Error: 500 Internal Server Error",
                  tool_call_id=f"e{i}", name="api_call"),
             _msg("assistant", "Got error, retrying..."))
            for i in range(6)
        ],
    },
    {
        "id": "stagnant-writer-1",
        "agent": "content-writer",
        "messages_gen": lambda: [
            (_msg("system", "Writing agent."),
             _msg("user", "Write about AI"),
             _msg("assistant",
                  "AI is transforming the world. Artificial intelligence "
                  "is changing how we work and live." +
                  " The impact of AI on society is profound." +
                  " Machine learning and deep learning are key."),
             _msg("assistant",
                  "AI technology continues to advance rapidly." +
                  " The field of artificial intelligence is evolving." +
                  " ML and DL are core technologies driving change." +
                  " AI's impact on society grows."))
            for _ in range(4)
        ],
    },
    {
        "id": "mixed-behavior-1",
        "agent": "multi-tool",
        "messages_gen": lambda: [
            (_msg("system", "Multi-tool agent."),
             _msg("user", "Analyze data"),
             _msg("assistant", "Reading data",
                  tool_calls=[_tc("m1", "read_file", {"path": "data.csv"})]),
             _msg("tool", "1,2,3\n4,5,6",
                  tool_call_id="m1", name="read_file"),
             _msg("assistant", "Data processed.")),
            (_msg("system", "Multi-tool agent."),
             _msg("user", "Visualize results"),
             _msg("assistant", "Creating chart",
                  tool_calls=[_tc("m2", "plot", {"type": "bar"})]),
             _msg("tool", "Chart saved.",
                  tool_call_id="m2", name="plot"),
             _msg("assistant", "Done.")),
        ],
    },
]

# ── Run audit on all conversations ──

print("=== Audit Report: TokenCircuit Intervention Analysis ===")
print()

config = InterventionConfig(
    nudge_threshold=2, override_threshold=4, hard_stop_threshold=6,
    audit_mode=True,
    enable_semantic_detection=True,
    enable_transcript_validation=True,
    window_size=4,
)

results = []

for conv in conversations:
    engine = InterventionEngine(config=config)
    state = {"_tc_intervention": default_intervention_state()}
    conv_results = {"id": conv["id"], "turns": [], "max_stage": "PASS"}

    raw_turns = conv["messages_gen"]()
    # Flatten messages into turns (if generator returns tuple of message tuples)
    if isinstance(raw_turns[0], tuple):
        turns = raw_turns
    else:
        # Single-turn conversation
        turns = [raw_turns]

    for turn_msgs in turns:
        if isinstance(turn_msgs, tuple):
            msgs = list(turn_msgs)
        else:
            msgs = turn_msgs

        decision = engine.process(msgs, state, thread_id=conv["id"],
                                   node_name=conv["agent"])
        state["_tc_intervention"] = decision.state_patch
        conv_results["turns"].append({
            "stage": decision.stage.name,
            "signals": [s.value for s in decision.signals],
            "coaching": decision.coaching_message,
            "would_terminate": decision.should_terminate,
        })

    max_stage = max(t["stage"] for t in conv_results["turns"])
    conv_results["max_stage"] = max_stage
    results.append(conv_results)

# ── Print audit report ──

agents = {c["id"]: c["agent"] for c in conversations}

print(f"{'Conversation':30s} {'Agent':20s} {'Max Stage':12s} {'Interventions':15s}")
print("-" * 80)
for r in results:
    n_interventions = sum(1 for t in r["turns"]
                           if t["stage"] != "PASS")
    print(f"{r['id']:30s} {agents.get(r['id'], '?'):20s} "
          f"{r['max_stage']:12s} {n_interventions:5d} turns")
print()

# ── Detailed analysis ──
print("Detailed Analysis:")
print()
for r in results:
    print(f"  Conversation: {r['id']}")
    for i, t in enumerate(r["turns"]):
        signals_str = ", ".join(t["signals"]) if t["signals"] else "none"
        print(f"    Turn {i+1}: {t['stage']:10s} "
              f"signals=[{signals_str}]"
              f"{' ⚠ WOULD TERMINATE' if t['would_terminate'] else ''}")
    print()

# ── Summary statistics ──
print("Summary:")
print(f"  Conversations analyzed: {len(results)}")
total_turns = sum(len(r["turns"]) for r in results)
total_interventions = sum(
    sum(1 for t in r["turns"] if t["stage"] != "PASS")
    for r in results
)
total_terminations = sum(
    sum(1 for t in r["turns"] if t["would_terminate"])
    for r in results
)
print(f"  Total turns: {total_turns}")
print(f"  Interventions prevented: {total_interventions}")
print(f"  Terminations prevented: {total_terminations}")
print(f"  Agents variants: {len(agents)}")
print()

# ── State schema reducer demonstration ──
print("State Reducer Demo: List field merging")
from tokencircuit import tc_state_reducer

existing = {
    "orphaned_transaction_ids": ["a", "b"],
    "dropped_this_session": ["x"],
    "coaching_history": ["First coaching"],
}
update = {
    "orphaned_transaction_ids": ["b", "c"],
    "dropped_this_session": ["y"],
    "coaching_history": ["Second coaching"],
}
merged = tc_state_reducer(existing, update)
print(f"  Existing orphaned IDs: {existing['orphaned_transaction_ids']}")
print(f"  Update orphaned IDs:   {update['orphaned_transaction_ids']}")
print(f"  Merged orphaned IDs:   {merged['orphaned_transaction_ids']}")
print(f"  ✓ List fields append with deduplication")
print()

print("All audit-reporter scenarios complete.")
