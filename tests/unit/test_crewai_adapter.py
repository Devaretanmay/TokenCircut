from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from tokencircuit.adapters.crewai import CrewAIAdapter
from tokencircuit.engine import (
    InterventionConfig,
    InterventionEngine,
)
from tokencircuit.types import InterventionStage


@dataclass
class FakeAgent:
    role: str = "test_agent"


@dataclass
class FakeTask:
    pass


@dataclass
class FakeLLMCallHookContext:
    messages: list[dict[str, Any]] = field(default_factory=list)
    agent: FakeAgent = field(default_factory=FakeAgent)
    task: Optional[FakeTask] = None


class TestCrewAIAdapterInit:
    def test_default_config(self):
        adapter = CrewAIAdapter()
        assert adapter._config.nudge_threshold == 3
        assert adapter._engine is not None

    def test_custom_config(self):
        config = InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3,
        )
        adapter = CrewAIAdapter(config=config)
        assert adapter._config.nudge_threshold == 1

    def test_custom_engine(self):
        engine = InterventionEngine()
        adapter = CrewAIAdapter(engine=engine)
        assert adapter._engine is engine


class TestCrewAIAdapterHook:
    def test_empty_messages_returns_true(self):
        adapter = CrewAIAdapter()
        ctx = FakeLLMCallHookContext(messages=[])
        assert adapter.hook(ctx) is True

    def test_pass_stage_returns_true(self):
        adapter = CrewAIAdapter()
        ctx = FakeLLMCallHookContext(messages=[{"role": "user", "content": "hello"}])
        assert adapter.hook(ctx) is True

    def test_hard_stop_via_direct_engine(self):
        config = InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3,
        )
        engine = InterventionEngine(config=config)
        state: dict[str, Any] = {}

        for i in range(1, 6):
            decision = engine.process(
                messages=[{"role": "user", "content": f"turn {i}"}],
                state=state,
                thread_id="crewai_test",
                node_name="test_agent",
            )
            state = {"_tc_intervention": decision.state_patch}
            if decision.stage == InterventionStage.HARD_STOP:
                return

        pytest.fail("Expected HARD_STOP after escalation via engine.process()")

    def test_hard_stop_via_adapter_with_state_accumulation(self):
        config = InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3,
        )
        adapter = CrewAIAdapter(config=config)
        task = FakeTask()
        state: dict[str, Any] = {}

        for i in range(1, 6):
            ctx = FakeLLMCallHookContext(
                messages=[{"role": "user", "content": f"turn {i}"}],
                agent=FakeAgent(role="test_agent"),
                task=task,
            )
            decision = adapter._engine.process(
                messages=ctx.messages,
                state=state,
                thread_id="crewai_test_agent_" + str(id(task)),
                node_name="test_agent",
            )
            state = {"_tc_intervention": decision.state_patch}
            if decision.stage == InterventionStage.HARD_STOP:
                return

        pytest.fail("Expected HARD_STOP via adapter engine with accumulated state")

    def test_nudge_appends_coaching(self):
        config = InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3,
        )
        engine = InterventionEngine(config=config)
        state: dict[str, Any] = {}

        decision1 = engine.process(
            messages=[{"role": "user", "content": "hello"}],
            state=state,
            thread_id="crewai_test",
            node_name="test_agent",
        )
        state = {"_tc_intervention": decision1.state_patch}

        decision2 = engine.process(
            messages=[{"role": "user", "content": "hello again"}],
            state=state,
            thread_id="crewai_test",
            node_name="test_agent",
        )
        assert decision2.stage == InterventionStage.NUDGE
        assert decision2.llm_input_messages is not None

    def test_hook_error_failsafe_returns_true(self, caplog):
        class ExplodingConfig:
            @property
            def max_threads(self):
                raise RuntimeError("boom")

            nudge_threshold = 3
            override_threshold = 5
            hard_stop_threshold = 8
            audit_mode = False
            window_size = 5
            max_tokens_per_turn = 4000
            similarity_threshold = 0.92
            enable_semantic_detection = False
            enable_transcript_validation = False
            auto_recovery = True
            max_budget_usd = 0.0
            token_pricing = {}
            nudge_template = ""
            override_template = ""
            cooldown_turns = 2
            max_orphan_tolerance = 2

        adapter = CrewAIAdapter(config=ExplodingConfig())
        ctx = FakeLLMCallHookContext(messages=[{"role": "user", "content": "hi"}])
        assert adapter.hook(ctx) is True
        assert "CrewAIAdapter hook failed" in caplog.text
