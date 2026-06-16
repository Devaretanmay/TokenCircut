"""
Loop Simulation Tests — Deer-Flow (repeated bash) and Hermes (repeated grep) harnesses.

These integration-level tests simulate real-world agent loops observed in production:
1. Deer-Flow pattern: Agent repeatedly calls bash with the same command
2. Hermes pattern: Agent repeatedly calls grep with slight variations

Asserts correct escalation PASS → NUDGE → OVERRIDE → HARD_STOP, and that
OVERRIDE successfully strips looping transactions while preserving valid context.
"""

import pytest
from typing import Any

from tokencircuit.engine import InterventionEngine, InterventionConfig
from tokencircuit.state_schema import default_intervention_state
from tokencircuit.types import InterventionStage, SignalType
from tokencircuit.engine import TokenCircuitError


def _make_tool_call_msg(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Helper to create a valid assistant tool_call message."""
    import json
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def _make_tool_result_msg(call_id: str, content: str, name: str) -> dict[str, Any]:
    """Helper to create a valid tool result message."""
    return {"role": "tool", "content": content, "tool_call_id": call_id, "name": name}


class TestDeerFlowBashLoop:
    """
    Simulates the Deer-Flow pattern: an agent stuck calling bash with the same
    command ("ls /nonexistent") repeatedly, receiving "No such file or directory"
    each time, then trying again identically.
    """

    def _build_deer_flow_transcript(self, iterations: int) -> list[dict[str, Any]]:
        """Build a transcript of N iterations of the bash loop."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Find the configuration file for the project."},
        ]
        for i in range(iterations):
            call_id = f"call_bash_{i}"
            messages.append(_make_tool_call_msg(call_id, "bash", {"command": "ls /nonexistent/config"}))
            messages.append(_make_tool_result_msg(
                call_id,
                "ls: /nonexistent/config: No such file or directory",
                "bash",
            ))
        return messages

    def test_escalation_through_all_stages(self):
        """Engine correctly escalates through PASS → NUDGE → OVERRIDE → HARD_STOP."""
        config = InterventionConfig(
            nudge_threshold=2,
            override_threshold=4,
            hard_stop_threshold=6,
            window_size=5,
            similarity_threshold=0.8,
        )
        engine = InterventionEngine(config=config)

        stages_seen: list[InterventionStage] = []
        tc_state = default_intervention_state()

        for turn in range(1, 10):
            messages = self._build_deer_flow_transcript(turn)
            state = {
                "messages": messages,
                "_tc_intervention": dict(tc_state),
                "configurable": {"thread_id": "deer-flow-test"},
            }

            decision = engine.process(messages, state, thread_id="deer-flow-test", node_name="agent")
            stages_seen.append(decision.stage)

            # Update state
            for k, v in decision.state_patch.items():
                tc_state[k] = v

            if decision.should_terminate:
                break

        # Verify we hit all stages
        assert InterventionStage.PASS in stages_seen, "Must start with PASS"
        assert InterventionStage.NUDGE in stages_seen, "Must hit NUDGE"
        assert InterventionStage.OVERRIDE in stages_seen, "Must hit OVERRIDE"
        assert InterventionStage.HARD_STOP in stages_seen, "Must reach HARD_STOP"

        # Verify monotonic ordering
        first_nudge = next(i for i, s in enumerate(stages_seen) if s == InterventionStage.NUDGE)
        first_override = next(i for i, s in enumerate(stages_seen) if s == InterventionStage.OVERRIDE)
        first_stop = next(i for i, s in enumerate(stages_seen) if s == InterventionStage.HARD_STOP)
        assert first_nudge < first_override < first_stop, "Stages must escalate monotonically"

    def test_nudge_contains_coaching_for_bash(self):
        """NUDGE coaching message references the stuck tool and suggests alternatives."""
        config = InterventionConfig(nudge_threshold=2, override_threshold=4, hard_stop_threshold=6)
        engine = InterventionEngine(config=config)
        tc_state = default_intervention_state()

        # Run through to NUDGE
        for turn in range(1, 5):
            messages = self._build_deer_flow_transcript(turn)
            state = {"messages": messages, "_tc_intervention": dict(tc_state), "configurable": {"thread_id": "t1"}}
            decision = engine.process(messages, state, thread_id="t1", node_name="agent")
            for k, v in decision.state_patch.items():
                tc_state[k] = v
            if decision.stage == InterventionStage.NUDGE:
                assert decision.coaching_message is not None
                assert decision.llm_input_messages is not None
                # Coaching should be appended as last message
                last_msg = decision.llm_input_messages[-1]
                assert last_msg["role"] == "system"
                assert "approach" in last_msg["content"].lower() or "different" in last_msg["content"].lower()
                return

        pytest.fail("Should have reached NUDGE within 4 turns")

    def test_override_strips_repeated_bash_calls(self):
        """OVERRIDE compacts the transcript, removing redundant bash iterations."""
        config = InterventionConfig(nudge_threshold=2, override_threshold=3, hard_stop_threshold=6)
        engine = InterventionEngine(config=config)
        tc_state = default_intervention_state()

        messages = self._build_deer_flow_transcript(6)  # 6 identical iterations

        # Build up stagnation naturally through repeated calls
        for turn in range(1, 8):
            state = {"messages": messages, "_tc_intervention": dict(tc_state), "configurable": {"thread_id": "t2"}}
            decision = engine.process(messages, state, thread_id="t2", node_name="agent")
            for k, v in decision.state_patch.items():
                tc_state[k] = v
            if decision.stage >= InterventionStage.OVERRIDE:
                # Verify compaction: output should be shorter than input
                if decision.llm_input_messages:
                    assert len(decision.llm_input_messages) < len(messages), (
                        f"OVERRIDE should compact: {len(decision.llm_input_messages)} >= {len(messages)}"
                    )
                    # Verify the system/user context is preserved
                    roles = [m["role"] for m in decision.llm_input_messages]
                    assert "user" in roles, "Original user message must be preserved"
                    assert "system" in roles, "System/coaching message must be present"
                return

        pytest.fail(f"Should have reached OVERRIDE within 7 turns, last stage: {decision.stage.name}")

    def test_hard_stop_terminates(self):
        """HARD_STOP correctly signals termination after sustained stagnation."""
        config = InterventionConfig(nudge_threshold=1, override_threshold=2, hard_stop_threshold=4)
        engine = InterventionEngine(config=config)
        tc_state = default_intervention_state()

        messages = self._build_deer_flow_transcript(8)

        # Build up stagnation naturally until HARD_STOP
        for turn in range(1, 15):
            state = {"messages": messages, "_tc_intervention": dict(tc_state), "configurable": {"thread_id": "t3"}}
            decision = engine.process(messages, state, thread_id="t3", node_name="agent")
            for k, v in decision.state_patch.items():
                tc_state[k] = v
            if decision.should_terminate:
                assert decision.termination_reason is not None
                assert "HARD_STOP" in decision.termination_reason
                return

        pytest.fail(f"Should have reached HARD_STOP within 14 turns, last stage: {decision.stage.name}")


class TestHermesGrepLoop:
    """
    Simulates the Hermes pattern: an agent stuck calling grep with slight
    variations of the same query, receiving empty results.
    """

    def _build_hermes_transcript(self, iterations: int) -> list[dict[str, Any]]:
        """Build transcript of grep loop with slight variations."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a code search assistant."},
            {"role": "user", "content": "Find all usages of the deprecated API."},
        ]
        # Slight variations in grep pattern (paraphrased loop)
        variations = [
            "deprecated_api",
            "deprecated_API",
            "deprecatedApi",
            "deprecated.api",
            "DEPRECATED_API",
            "deprecated api",
            "deprecated-api",
            "deprecated_api_v1",
        ]
        for i in range(iterations):
            call_id = f"call_grep_{i}"
            pattern = variations[i % len(variations)]
            messages.append(_make_tool_call_msg(call_id, "grep", {"pattern": pattern, "path": "./src"}))
            messages.append(_make_tool_result_msg(call_id, "No matches found.", "grep"))
        return messages

    def test_hermes_loop_detected_despite_variations(self):
        """
        Even though the grep patterns vary slightly, the engine should
        detect FUTILE_ACTION (same tool, different results) or SEMANTIC_STAGNATION.
        """
        config = InterventionConfig(
            nudge_threshold=3,
            override_threshold=5,
            hard_stop_threshold=7,
            window_size=5,
            similarity_threshold=0.7,
        )
        engine = InterventionEngine(config=config)
        tc_state = default_intervention_state()

        intervention_triggered = False
        for turn in range(1, 10):
            messages = self._build_hermes_transcript(turn)
            state = {"messages": messages, "_tc_intervention": dict(tc_state), "configurable": {"thread_id": "hermes"}}
            decision = engine.process(messages, state, thread_id="hermes", node_name="agent")
            for k, v in decision.state_patch.items():
                tc_state[k] = v

            if decision.stage > InterventionStage.PASS:
                intervention_triggered = True
                break

        assert intervention_triggered, "Engine should intervene on Hermes grep loop"

    def test_hermes_override_preserves_original_task(self):
        """OVERRIDE must preserve the original user message (the task)."""
        config = InterventionConfig(nudge_threshold=2, override_threshold=3, hard_stop_threshold=6)
        engine = InterventionEngine(config=config)
        tc_state = default_intervention_state()
        tc_state["consecutive_stagnation_count"] = 4
        tc_state["current_stage"] = "nudge"
        tc_state["turn_counter"] = 5

        messages = self._build_hermes_transcript(6)
        state = {"messages": messages, "_tc_intervention": tc_state, "configurable": {"thread_id": "h2"}}
        decision = engine.process(messages, state, thread_id="h2", node_name="agent")

        if decision.llm_input_messages:
            contents = " ".join(m.get("content", "") or "" for m in decision.llm_input_messages)
            assert "deprecated" in contents.lower() or "api" in contents.lower(), (
                "Original task context must be preserved in OVERRIDE output"
            )

    def test_override_coaching_mentions_grep(self):
        """OVERRIDE directive should reference the futile tool."""
        config = InterventionConfig(nudge_threshold=1, override_threshold=2, hard_stop_threshold=5)
        engine = InterventionEngine(config=config)
        tc_state = default_intervention_state()
        tc_state["consecutive_stagnation_count"] = 3
        tc_state["current_stage"] = "nudge"
        tc_state["turn_counter"] = 4

        messages = self._build_hermes_transcript(5)
        state = {"messages": messages, "_tc_intervention": tc_state, "configurable": {"thread_id": "h3"}}
        decision = engine.process(messages, state, thread_id="h3", node_name="agent")

        if decision.coaching_message:
            # Coaching should acknowledge the repetition
            msg = decision.coaching_message.lower()
            assert "attempt" in msg or "strateg" in msg or "different" in msg


class TestMixedToolLoop:
    """
    Tests a more complex loop pattern: agent alternates between two tools
    but makes no progress overall.
    """

    def _build_mixed_loop(self, iterations: int) -> list[dict[str, Any]]:
        """Agent alternates search → fetch → search → fetch with same results."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Research assistant"},
            {"role": "user", "content": "Find the latest release notes for library X."},
        ]
        for i in range(iterations):
            if i % 2 == 0:
                call_id = f"call_search_{i}"
                messages.append(_make_tool_call_msg(call_id, "web_search", {"query": "library X release notes"}))
                messages.append(_make_tool_result_msg(call_id, "No relevant results found.", "web_search"))
            else:
                call_id = f"call_fetch_{i}"
                messages.append(_make_tool_call_msg(call_id, "url_fetch", {"url": "https://lib-x.io/releases"}))
                messages.append(_make_tool_result_msg(call_id, "Error 404: Page not found", "url_fetch"))
        return messages

    def test_mixed_tool_loop_eventually_intervenes(self):
        """Even alternating tools should be caught if results are consistently failing."""
        config = InterventionConfig(
            nudge_threshold=3,
            override_threshold=5,
            hard_stop_threshold=8,
            window_size=6,
        )
        engine = InterventionEngine(config=config)
        tc_state = default_intervention_state()

        max_stage = InterventionStage.PASS
        for turn in range(1, 12):
            messages = self._build_mixed_loop(turn)
            state = {"messages": messages, "_tc_intervention": dict(tc_state), "configurable": {"thread_id": "mixed"}}
            decision = engine.process(messages, state, thread_id="mixed", node_name="agent")
            for k, v in decision.state_patch.items():
                tc_state[k] = v
            max_stage = max(max_stage, decision.stage)
            if decision.should_terminate:
                break

        assert max_stage >= InterventionStage.NUDGE, (
            "Mixed tool loop with consistent failures should trigger at least NUDGE"
        )

    def test_progress_resets_intervention(self):
        """
        If the agent makes genuine progress (new tool family with success),
        the intervention state should reset.
        """
        config = InterventionConfig(nudge_threshold=2, override_threshold=4, hard_stop_threshold=6)
        engine = InterventionEngine(config=config)
        tc_state = default_intervention_state()

        # Build up stagnation
        stagnant_messages = self._build_mixed_loop(4)
        state = {"messages": stagnant_messages, "_tc_intervention": dict(tc_state), "configurable": {"thread_id": "prog"}}
        for turn in range(1, 4):
            decision = engine.process(stagnant_messages, state, thread_id="prog", node_name="agent")
            for k, v in decision.state_patch.items():
                tc_state[k] = v
            state["_tc_intervention"] = dict(tc_state)

        # Now inject genuine progress: new tool with success
        progress_messages = list(stagnant_messages) + [
            _make_tool_call_msg("call_new", "read_file", {"path": "/docs/RELEASE.md"}),
            _make_tool_result_msg("call_new", "# Release Notes v2.0\n\n- Feature A\n- Feature B", "read_file"),
            {"role": "assistant", "content": "I found the release notes! Version 2.0 includes Feature A and Feature B."},
        ]
        state_progress = {"messages": progress_messages, "_tc_intervention": dict(tc_state), "configurable": {"thread_id": "prog"}}
        decision = engine.process(progress_messages, state_progress, thread_id="prog", node_name="agent")

        # After genuine progress, should de-escalate
        assert decision.stage == InterventionStage.PASS, (
            f"Genuine progress should reset to PASS, got {decision.stage.name}"
        )
