from __future__ import annotations

import pytest

from tokencircuit.engine import InterventionConfig, InterventionEngine
from tokencircuit.types import (
    InterventionContext,
    InterventionStage,
    SignalType,
)

pytestmark = pytest.mark.benchmark(min_rounds=100, warmup=True)


@pytest.fixture
def engine() -> InterventionEngine:
    return InterventionEngine(
        config=InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3,
        )
    )


@pytest.fixture
def pass_context() -> InterventionContext:
    return InterventionContext(
        thread_id="bench",
        node_name="agent",
        turn_number=1,
    )


@pytest.fixture
def nudge_context() -> InterventionContext:
    return InterventionContext(
        thread_id="bench",
        node_name="agent",
        turn_number=5,
        active_signals=[SignalType.STATE_STAGNATION],
        consecutive_stagnation_count=3,
        current_stage=InterventionStage.PASS,
    )


@pytest.fixture
def hard_stop_context() -> InterventionContext:
    return InterventionContext(
        thread_id="bench",
        node_name="agent",
        turn_number=10,
        active_signals=[SignalType.STATE_STAGNATION, SignalType.FUTILE_ACTION],
        consecutive_stagnation_count=8,
        current_stage=InterventionStage.OVERRIDE,
    )


class TestDecideHotPath:
    def test_decide_pass(self, benchmark, engine, pass_context):
        benchmark(engine.decide, pass_context, None)

    def test_decide_nudge(self, benchmark, engine, nudge_context):
        benchmark(engine.decide, nudge_context, None)

    def test_decide_hard_stop(self, benchmark, engine, hard_stop_context):
        benchmark(engine.decide, hard_stop_context, None)

    def test_decide_many_contexts(self, benchmark, engine):
        contexts = [
            InterventionContext(
                thread_id=f"thread_{i}",
                node_name="agent",
                turn_number=i,
                active_signals=[SignalType.STATE_STAGNATION] if i % 3 == 0 else [],
                consecutive_stagnation_count=i % 10,
                current_stage=(
                    InterventionStage.PASS if i < 3 else InterventionStage.NUDGE
                ),
            )
            for i in range(100)
        ]
        idx = [0]

        def cycle():
            c = contexts[idx[0] % len(contexts)]
            idx[0] += 1
            return engine.decide(c, None)

        benchmark(cycle)


class TestProcessHotPath:
    def test_process_simple_messages(self, benchmark):
        engine = InterventionEngine()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in San Francisco?"},
        ]
        state = {"_tc_intervention": {}, "messages": messages}

        def run():
            return engine.process(
                messages=messages,
                state=state,
                thread_id="bench",
                node_name="agent",
            )

        benchmark(run)

    def test_process_with_tool_calls(self, benchmark):
        engine = InterventionEngine()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"location": "SF"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Sunny, 72°F",
                "tool_call_id": "call_1",
            },
        ]
        state = {"_tc_intervention": {}, "messages": messages}

        def run():
            return engine.process(
                messages=messages,
                state=state,
                thread_id="bench_tools",
                node_name="agent",
            )

        benchmark(run)
