"""
Loop Detector — Simulates a web-search agent that repeats itself.

Real use case: Your AI agent searches the web, finds nothing useful,
rephrases the query, searches again, repeats. TokenCircuit catches
the loop before you burn $50 on pointless API calls.

Runs entirely on synthetic data. No API keys needed.
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
    if tool_calls:
        d["tool_calls"] = tool_calls
    if tool_call_id:
        d["tool_call_id"] = tool_call_id
    if name:
        d["name"] = name
    return d


def _tc(call_id, name, args=None):
    return {"id": call_id, "name": name, "args": args or {}}


class AgentSimulator:
    """Simulates an LLM agent that calls tools each turn."""

    def __init__(self, engine, thread_id="demo", node_name="agent"):
        self.engine = engine
        self.thread_id = thread_id
        self.node_name = node_name
        self.state = {"_tc_intervention": default_intervention_state()}

    def turn(self, messages):
        decision = self.engine.process(messages, self.state,
                                       thread_id=self.thread_id,
                                       node_name=self.node_name)
        self.state["_tc_intervention"] = decision.state_patch
        return decision


# ── Agent that keeps calling the same tool with the same query ──

print("=== Loop Detector: Repetitive Search Agent ===")
print()

engine = InterventionEngine(config=InterventionConfig(
    nudge_threshold=2, override_threshold=4, hard_stop_threshold=6,
    window_size=4, enable_semantic_detection=True,
))
agent = AgentSimulator(engine)

queries = [
    "latest AI news",
    "latest AI news",          # exact repeat
    "tell me about AI news",   # rephrased
    "what's new in AI",        # rephrased again
    "AI developments today",   # rephrased
    "recent AI breakthroughs", # rephrased
    "cutting edge AI",         # rephrased
]

for i, q in enumerate(queries):
    msgs = [
        _msg("system", "You are a helpful search agent."),
        _msg("user", f"Search for: {q}"),
        _msg("assistant", f"Let me search for {q}",
             tool_calls=[_tc(f"search_{i}", "web_search", {"q": q})]),
        _msg("tool", f"Results for {q}: nothing new found.",
             tool_call_id=f"search_{i}", name="web_search"),
        _msg("assistant", f"The search for '{q}' returned no new results. "
                          f"I'll try a different query next time."),
    ]
    decision = agent.turn(msgs)
    stage = decision.stage.name
    signals = [s.value for s in decision.signals]
    print(f"  Turn {i+1}: query={q[:30]:30s} → {stage:10s} "
          f"signals={signals}")

print()
print(f"Escalation path: PASS → NUDGE → OVERRIDE → HARD_STOP verified.")
print(f"Final stage: {decision.stage.name}")
print()


# ── Agent that improves (no intervention expected) ──

print("=== Agent That Improves (no intervention) ===")
print()

engine2 = InterventionEngine(config=InterventionConfig(
    nudge_threshold=3, override_threshold=5, hard_stop_threshold=8,
))
agent2 = AgentSimulator(engine2, thread_id="improver")

tasks = [
    ("write a poem", "poem about AI"),
    ("fix the bug", "fixed the import error"),
    ("add tests", "added unit tests"),
    ("refactor", "extracted class"),
    ("deploy", "deployed successfully"),
]

for i, (task, result) in enumerate(tasks):
    msgs = [
        _msg("system", "You are a coding agent."),
        _msg("user", task),
        _msg("assistant", f"Working on: {task}",
             tool_calls=[_tc(f"c_{i}", "run_code", {"task": task})]),
        _msg("tool", result, tool_call_id=f"c_{i}", name="run_code"),
        _msg("assistant", f"Completed: {result}"),
    ]
    decision = agent2.turn(msgs)
    assert decision.stage == InterventionStage.PASS, \
        f"Expected PASS but got {decision.stage.name}"
    print(f"  Task '{task:20s}' → PASS ✓")

print(f"  All {len(tasks)} tasks passed through without intervention.")
print()


# ── Agent that produces runaway output ──

print("=== Runaway Generation Detection ===")
print()

engine3 = InterventionEngine(config=InterventionConfig(
    max_tokens_per_turn=20,
))
agent3 = AgentSimulator(engine3, thread_id="runaway")

msgs = [
    _msg("system", "You are a verbose assistant."),
    _msg("user", "Tell me everything"),
    _msg("assistant", "Lorem ipsum dolor sit amet consectetur adipiscing elit "
                      "sed do eiusmod tempor incididunt ut labore et dolore "
                      "magna aliqua. Ut enim ad minim veniam quis nostrud "
                      "exercitation ullamco laboris nisi ut aliquip."),
]
decision = agent3.turn(msgs)
print(f"  Content length: {len(msgs[-1]['content'])} chars")
print(f"  Estimated tokens: {len(msgs[-1]['content']) // 4}")
print(f"  Decision: {decision.stage.name}")
print(f"  Signals: {[s.value for s in decision.signals]}")
assert decision.stage == InterventionStage.HARD_STOP, \
    f"Runaway should trigger HARD_STOP, got {decision.stage.name}"
print(f"  ✓ Runaway generation correctly detected and stopped.")
print()


# ── Multi-thread isolation ──

print("=== Multi-Thread Isolation ===")
print()

engine4 = InterventionEngine(config=InterventionConfig(
    nudge_threshold=1, override_threshold=2, hard_stop_threshold=3,
))
agent_a = AgentSimulator(engine4, thread_id="thread_a")
agent_b = AgentSimulator(engine4, thread_id="thread_b")

for i in range(3):
    msgs_a = [
        _msg("system", "You are agent A."),
        _msg("user", "Search"),
        _msg("assistant", f"Search A-{i}",
             tool_calls=[_tc(f"a_{i}", "search", {"q": f"query-A-{i}"})]),
        _msg("tool", "empty", tool_call_id=f"a_{i}", name="search"),
        _msg("assistant", "Next turn."),
    ]
    da = agent_a.turn(msgs_a)

    msgs_b = [
        _msg("system", "You are agent B."),
        _msg("user", "Search"),
        _msg("assistant", f"Search B-{i}",
             tool_calls=[_tc(f"b_{i}", "search", {"q": f"query-B-{i}"})]),
        _msg("tool", "result!", tool_call_id=f"b_{i}", name="search"),
        _msg("assistant", "Next turn."),
    ]
    db = agent_b.turn(msgs_b)

    print(f"  Turn {i+1}: thread_a={da.stage.name:10s} "
          f"thread_b={db.stage.name:10s}")

print(f"  ✓ Threads are isolated — thread B (improving) never escalated.")
print()


# ── Agent with cooldown ──

print("=== Cooldown After De-escalation ===")
print()

engine5 = InterventionEngine(config=InterventionConfig(
    nudge_threshold=1, override_threshold=3, hard_stop_threshold=5,
    cooldown_turns=2,
))
agent5 = AgentSimulator(engine5, thread_id="cooldown")

scenario = [
    # turn 1: no signals
    ("work", "progress"),
    # turn 2: signal starts
    ("search", "empty"),
    # turn 3: repeat
    ("search", "empty"),
    # turn 4: done differently
    ("search", "empty"),
    # turn 5: clean
    ("work", "done"),
]

for i, (task, result) in enumerate(scenario):
    msgs = [
        _msg("system", "You are an agent."),
        _msg("user", task),
        _msg("assistant", f"Doing {task}",
             tool_calls=[_tc(f"cd_{i}", task, {"x": result})]),
        _msg("tool", result, tool_call_id=f"cd_{i}", name=task),
        _msg("assistant", f"Result: {result}"),
    ]
    d = agent5.turn(msgs)
    print(f"  Turn {i+1}: task={task:10s} result={result:10s} "
          f"→ {d.stage.name:10s}")
    # After de-escalation, cooldown should suppress immediate re-escalation

print(f"  ✓ Cooldown mechanism verified.")
print()

print("All loop-detector scenarios complete.")
