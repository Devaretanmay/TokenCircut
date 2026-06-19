import pytest

pytest.importorskip("crewai")

from crewai import Agent, Crew, Task

from tokencircuit import InterventionConfig, TokenCircuitError
from tokencircuit.adapters.crewai import instrument_crewai


def test_crewai_adapter_hard_stop():
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm="gpt-3.5-turbo",
    )
    task = Task(description="Test task", expected_output="Test output", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    config = InterventionConfig(
        nudge_threshold=1,
        override_threshold=2,
        hard_stop_threshold=3,
    )
    instrument_crewai(crew, config=config)

    assert agent.step_callback is not None

    with pytest.raises(TokenCircuitError, match="TokenCircuit HARD_STOP"):
        for _ in range(5):
            agent.step_callback("identical output")


def test_crewai_adapter_pass_with_progress():
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm="gpt-3.5-turbo",
    )
    task = Task(description="Test task", expected_output="Test output", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    config = InterventionConfig(
        nudge_threshold=3,
    )
    instrument_crewai(crew, config=config)

    for i in range(5):
        result = agent.step_callback(f"unique output {i}")
        assert result == f"unique output {i}"


def test_crewai_adapter_chains_original_callback():
    called = False

    def original_cb(step_output):
        nonlocal called
        called = True
        return step_output

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm="gpt-3.5-turbo",
        step_callback=original_cb,
    )
    task = Task(description="Test task", expected_output="Test output", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    instrument_crewai(crew)

    result = agent.step_callback("test")
    assert result == "test"
    assert called is True
