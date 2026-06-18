"""
State Isolation Tests — verify _tc_intervention is strictly isolated per
(thread_id, node_name).

Critical invariants:
1. Different thread_ids MUST have completely independent intervention state.
2. Different node_names within the same thread MUST have independent state.
3. Concurrent execution MUST NOT cause state corruption.
4. State must survive across multiple process() calls for the same thread+node.
5. reset() for one key must not affect other keys.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tokencircuit.engine import InterventionConfig, InterventionEngine
from tokencircuit.state_schema import default_intervention_state
from tokencircuit.types import InterventionStage


def _stagnant_messages() -> list[dict[str, Any]]:
    """Messages that will trigger stagnation detection."""
    return [
        {"role": "user", "content": "Find X"},
        {
            "role": "assistant",
            "content": "Searching",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "X"}'},
                },  # noqa: E501
            ],
        },
        {
            "role": "tool",
            "content": "Not found",
            "tool_call_id": "call_1",
            "name": "search",
        },  # noqa: E501
    ]


def _progress_messages() -> list[dict[str, Any]]:
    """Messages that represent genuine progress (novel content)."""
    return [
        {"role": "user", "content": "Summarize findings"},
        {
            "role": "assistant",
            "content": "Based on my research, I've found the following key points about the quantum entanglement phenomenon.",
        },  # noqa: E501
    ]


class TestThreadIdIsolation:
    """Different thread_ids must have completely independent state."""

    def test_independent_stagnation_counters(self):
        """Thread A's stagnation count must not affect thread B."""
        config = InterventionConfig(
            nudge_threshold=2, override_threshold=4, hard_stop_threshold=6
        )  # noqa: E501
        engine = InterventionEngine(config=config)

        messages = _stagnant_messages()

        # Drive thread_a into stagnation
        tc_state_a = default_intervention_state()
        for turn in range(1, 5):
            state_a = {
                "messages": messages,
                "_tc_intervention": dict(tc_state_a),
                "configurable": {"thread_id": "thread_a"},
            }  # noqa: E501
            decision_a = engine.process(
                messages, state_a, thread_id="thread_a", node_name="agent"
            )  # noqa: E501
            for k, v in decision_a.state_patch.items():
                tc_state_a[k] = v

        # thread_b should start fresh
        tc_state_b = default_intervention_state()
        state_b = {
            "messages": messages,
            "_tc_intervention": tc_state_b,
            "configurable": {"thread_id": "thread_b"},
        }  # noqa: E501
        decision_b = engine.process(
            messages, state_b, thread_id="thread_b", node_name="agent"
        )  # noqa: E501

        assert decision_b.stage == InterventionStage.PASS, (
            f"Thread B should start at PASS, got {decision_b.stage.name} "
            f"(leaked from thread A which is at stagnation count {tc_state_a.get('consecutive_stagnation_count')})"  # noqa: E501
        )

    def test_100_independent_threads(self):
        """100 concurrent threads should have completely independent states."""
        config = InterventionConfig(
            nudge_threshold=3, override_threshold=5, hard_stop_threshold=8
        )  # noqa: E501
        engine = InterventionEngine(config=config)

        messages = _stagnant_messages()
        results: dict[str, list[InterventionStage]] = {}

        for thread_idx in range(100):
            thread_id = f"thread_{thread_idx:03d}"
            tc_state = default_intervention_state()
            stages: list[InterventionStage] = []

            # Run different number of turns per thread
            num_turns = (thread_idx % 5) + 1
            for turn in range(num_turns):
                state = {
                    "messages": messages,
                    "_tc_intervention": dict(tc_state),
                    "configurable": {"thread_id": thread_id},
                }  # noqa: E501
                decision = engine.process(
                    messages, state, thread_id=thread_id, node_name="agent"
                )  # noqa: E501
                stages.append(decision.stage)
                for k, v in decision.state_patch.items():
                    tc_state[k] = v

            results[thread_id] = stages

        # Threads with 1 turn should all be PASS
        for tid, stages in results.items():
            if len(stages) == 1:
                assert stages[0] == InterventionStage.PASS, (
                    f"{tid}: first turn should always be PASS"
                )  # noqa: E501

    def test_reset_one_thread_does_not_affect_others(self):
        """Resetting thread_a's state must not affect thread_b."""
        config = InterventionConfig(
            nudge_threshold=2, override_threshold=4, hard_stop_threshold=6
        )  # noqa: E501
        engine = InterventionEngine(config=config)

        messages = _stagnant_messages()

        # Build up state in both threads
        for thread_id in ["thread_a", "thread_b"]:
            tc_state = default_intervention_state()
            for turn in range(1, 4):
                state = {
                    "messages": messages,
                    "_tc_intervention": dict(tc_state),
                    "configurable": {"thread_id": thread_id},
                }  # noqa: E501
                engine.process(messages, state, thread_id=thread_id, node_name="agent")

        # Verify both have state
        assert "thread_a:agent" in engine._thread_states
        assert "thread_b:agent" in engine._thread_states

        # Reset only thread_a
        engine.reset("thread_a", "agent")

        # thread_a should be gone, thread_b unaffected
        assert "thread_a:agent" not in engine._thread_states, "thread_a should be reset"
        assert "thread_b:agent" in engine._thread_states, "thread_b must be unaffected"


class TestNodeNameIsolation:
    """Different node_names within the same thread must have independent state."""

    def test_agent_and_reviewer_independent(self):
        """Two nodes in the same thread should track stagnation independently."""
        config = InterventionConfig(
            nudge_threshold=2, override_threshold=4, hard_stop_threshold=6
        )  # noqa: E501
        engine = InterventionEngine(config=config)

        stagnant = _stagnant_messages()
        progress = _progress_messages()

        tc_state = default_intervention_state()

        # Drive "agent" into stagnation
        for turn in range(1, 5):
            state = {
                "messages": stagnant,
                "_tc_intervention": dict(tc_state),
                "configurable": {"thread_id": "t1"},
            }  # noqa: E501
            engine.process(stagnant, state, thread_id="t1", node_name="agent")  # noqa: E501

        # "reviewer" in same thread should be independent
        state_r = {
            "messages": progress,
            "_tc_intervention": dict(tc_state),
            "configurable": {"thread_id": "t1"},
        }  # noqa: E501
        decision_r = engine.process(
            progress, state_r, thread_id="t1", node_name="reviewer"
        )  # noqa: E501

        assert decision_r.stage == InterventionStage.PASS, (
            f"reviewer should be PASS (independent of agent), got {decision_r.stage.name}"  # noqa: E501
        )

    def test_three_nodes_independent_escalation(self):
        """Three nodes can be at different escalation stages simultaneously."""
        config = InterventionConfig(
            nudge_threshold=2, override_threshold=3, hard_stop_threshold=5
        )  # noqa: E501
        engine = InterventionEngine(config=config)
        messages = _stagnant_messages()

        node_stages: dict[str, InterventionStage] = {}

        for node_name, num_turns in [("agent", 5), ("planner", 3), ("critic", 1)]:
            tc_state = default_intervention_state()
            for turn in range(num_turns):
                state = {
                    "messages": messages,
                    "_tc_intervention": dict(tc_state),
                    "configurable": {"thread_id": "shared"},
                }  # noqa: E501
                decision = engine.process(
                    messages, state, thread_id="shared", node_name=node_name
                )  # noqa: E501
                for k, v in decision.state_patch.items():
                    tc_state[k] = v
            node_stages[node_name] = decision.stage

        # All three should be at different stages
        assert node_stages["critic"] == InterventionStage.PASS  # Only 1 turn
        assert (
            node_stages["planner"] >= InterventionStage.NUDGE
        )  # 3 turns >= nudge_threshold  # noqa: E501
        assert (
            node_stages["agent"] >= InterventionStage.OVERRIDE
        )  # 5 turns >= override_threshold  # noqa: E501


class TestConcurrentExecution:
    """Verify no state corruption under concurrent access."""

    def test_concurrent_threads_no_corruption(self):
        """Multiple OS threads processing different graph threads simultaneously."""
        config = InterventionConfig(
            nudge_threshold=3, override_threshold=5, hard_stop_threshold=8
        )  # noqa: E501
        engine = InterventionEngine(config=config)
        messages = _stagnant_messages()

        errors: list[str] = []

        def worker(
            thread_id: str, num_turns: int
        ) -> tuple[str, list[InterventionStage]]:  # noqa: E501
            stages: list[InterventionStage] = []
            tc_state = default_intervention_state()
            try:
                for turn in range(num_turns):
                    state = {
                        "messages": messages,
                        "_tc_intervention": dict(tc_state),
                        "configurable": {"thread_id": thread_id},
                    }  # noqa: E501
                    decision = engine.process(
                        messages, state, thread_id=thread_id, node_name="agent"
                    )  # noqa: E501
                    stages.append(decision.stage)
                    for k, v in decision.state_patch.items():
                        tc_state[k] = v
            except Exception as e:
                errors.append(f"{thread_id}: {e}")
            return thread_id, stages

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(20):
                futures.append(executor.submit(worker, f"concurrent_{i}", (i % 6) + 2))  # noqa: E501

            results: dict[str, list[InterventionStage]] = {}
            for future in as_completed(futures):
                tid, stages = future.result()
                results[tid] = stages

        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        assert len(results) == 20, "All 20 threads should complete"

        # Verify first turn is always PASS (no cross-contamination)
        for tid, stages in results.items():
            assert stages[0] == InterventionStage.PASS, (
                f"{tid}: first turn should be PASS, got {stages[0].name}"
            )

    def test_concurrent_same_thread_different_nodes(self):
        """Multiple nodes in the same thread processed concurrently."""
        config = InterventionConfig(
            nudge_threshold=2, override_threshold=4, hard_stop_threshold=6
        )  # noqa: E501
        engine = InterventionEngine(config=config)
        messages = _stagnant_messages()

        def process_node(node_name: str, turns: int) -> InterventionStage:
            tc_state = default_intervention_state()
            last_stage = InterventionStage.PASS
            for turn in range(turns):
                state = {
                    "messages": messages,
                    "_tc_intervention": dict(tc_state),
                    "configurable": {"thread_id": "shared_thread"},
                }  # noqa: E501
                decision = engine.process(
                    messages, state, thread_id="shared_thread", node_name=node_name
                )  # noqa: E501
                last_stage = decision.stage
                for k, v in decision.state_patch.items():
                    tc_state[k] = v
            return last_stage

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_node, f"node_{i}", (i + 1) * 2): f"node_{i}"
                for i in range(4)
            }
            results = {}
            for future in as_completed(futures):
                node = futures[future]
                results[node] = future.result()

        # node_0 (2 turns) should be at a lower stage than node_3 (8 turns)
        # At minimum, they shouldn't have corrupted each other
        assert results["node_0"] <= results["node_3"], (
            f"More turns should escalate further: node_0={results['node_0'].name}, node_3={results['node_3'].name}"  # noqa: E501
        )


class TestStatePersistedAcrossCalls:
    """Verify state accumulates correctly across multiple process() calls."""

    def test_stagnation_counter_accumulates(self):
        """Consecutive stagnation count should increase across calls."""
        config = InterventionConfig(
            nudge_threshold=3, override_threshold=5, hard_stop_threshold=8
        )  # noqa: E501
        engine = InterventionEngine(config=config)
        messages = _stagnant_messages()
        tc_state = default_intervention_state()

        counts: list[int] = []
        for turn in range(1, 6):
            state = {
                "messages": messages,
                "_tc_intervention": dict(tc_state),
                "configurable": {"thread_id": "acc"},
            }  # noqa: E501
            decision = engine.process(
                messages, state, thread_id="acc", node_name="agent"
            )  # noqa: E501
            for k, v in decision.state_patch.items():
                tc_state[k] = v
            counts.append(tc_state.get("consecutive_stagnation_count", 0))

        # Count should be monotonically non-decreasing while stagnating
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], (
                f"Stagnation count should not decrease: {counts}"
            )

    def test_stage_persists_in_state_patch(self):
        """The current_stage should be correctly reflected in state patches."""
        config = InterventionConfig(
            nudge_threshold=2, override_threshold=3, hard_stop_threshold=5
        )  # noqa: E501
        engine = InterventionEngine(config=config)
        messages = _stagnant_messages()
        tc_state = default_intervention_state()

        for turn in range(1, 5):
            state = {
                "messages": messages,
                "_tc_intervention": dict(tc_state),
                "configurable": {"thread_id": "stage"},
            }  # noqa: E501
            decision = engine.process(
                messages, state, thread_id="stage", node_name="agent"
            )  # noqa: E501
            for k, v in decision.state_patch.items():
                tc_state[k] = v

            # The state patch should record the current stage
            assert "current_stage" in decision.state_patch
            assert decision.state_patch["current_stage"] == decision.stage.name.lower()
