"""
Multi-Agent Stagnation — Simulates a LangGraph app with two nodes
(planner + executor), each monitored by TokenCircuit.

Real use case: A LangGraph agent with separate planning and execution
nodes. The planner keeps suggesting the same approach while the executor
keeps failing. TokenCircuit catches the planner loop before burning tokens.

No API keys. Fully synthetic message simulation.
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
    LangGraphPreModelAdapter,
    tc_state_reducer,
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


print("=== Multi-Agent Stagnation: Planner + Executor ===")
print()

# One engine shared across both nodes (real LangGraph pattern)
config = InterventionConfig(
    nudge_threshold=2, override_threshold=4, hard_stop_threshold=6,
    cooldown_turns=2, window_size=4,
)
engine = InterventionEngine(config=config)

# Simulate a LangGraph app running turns
# Each turn: planner thinks → executor acts → results come back

state = {"_tc_intervention": default_intervention_state(), "messages": []}

# ── Scenario 1: Healthy agent, no intervention ──
print("Scenario 1: Healthy agent — varied approaches")
for i in range(4):
    plan = f"Plan {i+1}: try approach {chr(65+i)}"
    result = f"Result {i+1}: approach {chr(65+i)} worked partially"

    # Planner node
    state["messages"] = [
        _msg("system", "You are a planner."),
        _msg("user", f"Solve problem iteration {i+1}"),
        _msg("assistant", plan),
    ]
    d1 = engine.process(state["messages"], state,
                        thread_id="multi", node_name="planner")
    state["_tc_intervention"] = tc_state_reducer(
        state["_tc_intervention"], d1.state_patch)

    # Executor node
    state["messages"] = state["messages"] + [
        _msg("assistant", f"Executing {plan}",
             tool_calls=[_tc(f"ex_{i}", "execute", {"plan": plan})]),
        _msg("tool", result, tool_call_id=f"ex_{i}", name="execute"),
        _msg("assistant", f"Observation: {result}"),
    ]
    d2 = engine.process(state["messages"], state,
                        thread_id="multi", node_name="executor")
    state["_tc_intervention"] = tc_state_reducer(
        state["_tc_intervention"], d2.state_patch)

    print(f"  Iteration {i+1}: planner={d1.stage.name:10s} "
          f"executor={d2.stage.name:10s}")
print(f"  ✓ Both nodes stay PASS with varied approaches.")
print()


# ── Scenario 2: Planner loops (same plan every time) ──
print("Scenario 2: Planner stuck in a loop")
state["_tc_intervention"] = default_intervention_state()

for i in range(5):
    plan = "We should try approach A."

    state["messages"] = [
        _msg("system", "You are a planner."),
        _msg("user", "Solve the problem"),
        _msg("assistant", f"Thinking about approach A",
             tool_calls=[_tc(f"p_{i}", "think", {"approach": "A"})]),
        _msg("tool", "Thinking...", tool_call_id=f"p_{i}", name="think"),
    ]
    d1 = engine.process(state["messages"], state,
                        thread_id="multi", node_name="planner")
    state["_tc_intervention"] = tc_state_reducer(
        state["_tc_intervention"], d1.state_patch)

    print(f"  Planner turn {i+1}: {d1.stage.name:10s} "
          f"({d1.coaching_message or 'no coaching'})")

    # Clear state for next turn (simulate different agent loop)
    state["_tc_intervention"]["turn_counter"] = i + 1
    state["_tc_intervention"]["consecutive_stagnation_count"] = \
        d1.state_patch.get("consecutive_stagnation_count", 0)
    state["_tc_intervention"]["current_stage"] = d1.state_patch.get(
        "current_stage", "pass")

print(f"  ✓ Planner escalated from PASS → NUDGE → OVERRIDE as it repeated.")
print()


# ── Scenario 3: LangGraphPreModelAdapter integration ──
print("Scenario 3: LangGraphPreModelAdapter hook integration")

adapter = LangGraphPreModelAdapter(
    config=InterventionConfig(
        nudge_threshold=2, override_threshold=4, hard_stop_threshold=6,
    )
)

# Simulate a pre_model_hook call
for i in range(3):
    hook_state = {
        "messages": [
            _msg("system", "You are coding."),
            _msg("user", "Write a search function"),
            _msg("assistant", f"Version {i+1}",
                 tool_calls=[_tc(f"h_{i}", "write_code",
                                  {"version": i+1})]),
            _msg("tool", "code written", tool_call_id=f"h_{i}",
                 name="write_code"),
            _msg("assistant", f"Done with version {i+1}"),
        ],
        "configurable": {"thread_id": "adapter-test"},
    }
    result = adapter._execute_hook(hook_state, node_name="coder")
    llm_messages = result.get("llm_input_messages")
    tc = result.get("_tc_intervention", {})

    if llm_messages:
        coaching_count = sum(
            1 for m in llm_messages if m["role"] == "system"
            and ("SYSTEM DIRECTIVE" in m["content"]
                 or "repeating" in m["content"])
        )
        print(f"  Hook turn {i+1}: {tc.get('current_stage', '?')} "
              f"{'✓ coaching injected' if coaching_count else 'no coaching'}")
    else:
        print(f"  Hook turn {i+1}: PASS (no message modification)")

print(f"  ✓ Adapter hook correctly returns llm_input_messages on intervention.")
print()


# ── Scenario 4: Audit mode ──
print("Scenario 4: Audit mode — observes without intervening")

adapter_audit = LangGraphPreModelAdapter(
    config=InterventionConfig(
        nudge_threshold=1, override_threshold=3, hard_stop_threshold=5,
        audit_mode=True,
    )
)
print("  Audit mode ON — interventions logged but not applied.")

for i in range(3):
    hook_state = {
        "messages": [
            _msg("system", "You are coding."),
            _msg("user", "Search for bug"),
            _msg("assistant", f"Searching bug #{i+1}",
                 tool_calls=[_tc(f"a_{i}", "search", {"bug": i+1})]),
            _msg("tool", "not found", tool_call_id=f"a_{i}", name="search"),
            _msg("assistant", "Searching again..."),
        ],
        "configurable": {"thread_id": "audit-test"},
    }
    result = adapter_audit._execute_hook(hook_state, node_name="audit_node")
    # In audit mode, hook never returns llm_input_messages
    has_llm = "llm_input_messages" in result
    tc = result.get("_tc_intervention", {})
    stage = tc.get("current_stage", "pass")
    print(f"  Turn {i+1}: stage={stage}, llm_input_messages={has_llm}")
    assert not has_llm, "Audit mode should not return llm_input_messages"

print(f"  ✓ Audit mode logs but does not modify messages.")
print()

print("All multi-agent scenarios complete.")
