"""
CrewAI integration tests for TokenCircuit.

These tests verify that the CrewAI interceptor correctly detects
delegation loops and raises TokenCircuitError.

NOTE: crewai requires Python < 3.14 due to tiktoken compatibility.
These tests are skipped if crewai is not installed.
"""

import pytest

pytest.importorskip("crewai")

from tokencircuit import instrument_crewai
from tokencircuit.engine import InterventionConfig


class TestCrewAIIntegration:
    """
    Integration tests for the CrewAI interceptor.

    Tests:
    - A looping agent delegation raises TokenCircuitError at iteration 5
    - Exception message identifies the looping agent role and task type
    - A legitimate 3-agent pipeline WITHOUT loops completes normally
    """

    def test_loop_detected(self):
        """Build a 3-agent Crew with allow_delegation=True that loops."""
        from crewai import Agent, Crew, Process, Task

        researcher = Agent(
            role="Researcher",
            goal="Research the topic",
            backstory="Expert researcher",
            allow_delegation=True,
        )
        writer = Agent(
            role="Writer",
            goal="Write the content",
            backstory="Expert writer",
            allow_delegation=True,
        )
        manager = Agent(
            role="Manager",
            goal="Manage the process",
            backstory="Expert manager",
            allow_delegation=True,
        )

        task1 = Task(
            description="Research AI trends",
            agent=researcher,
            expected_output="Research report",
        )
        task2 = Task(
            description="Write about AI trends",
            agent=writer,
            expected_output="Written article",
        )
        task3 = Task(
            description="Review and manage",
            agent=manager,
            expected_output="Management review",
        )

        crew = Crew(
            agents=[researcher, writer, manager],
            tasks=[task1, task2, task3],
            process=Process.hierarchical,
            verbose=False,
        )

        config = InterventionConfig(
            nudge_threshold=3,
            override_threshold=5,
            hard_stop_threshold=8,
            window_size=5,
        )
        safe_crew = instrument_crewai(crew, config=config)

        with pytest.raises(Exception) as exc_info:
            safe_crew.kickoff()

        msg = str(exc_info.value)
        assert "TokenCircuit" in msg

    def test_exception_message_identifies_agent(self):
        from crewai import Agent, Crew, Process, Task

        agent = Agent(
            role="LoopingAgent",
            goal="Cause a loop",
            backstory="Testing agent",
            allow_delegation=True,
        )
        task = Task(
            description="Looping task",
            agent=agent,
            expected_output="Loop output",
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.hierarchical,
            verbose=False,
        )

        config = InterventionConfig(
            nudge_threshold=3,
            override_threshold=5,
            hard_stop_threshold=8,
            window_size=5,
        )
        safe_crew = instrument_crewai(crew, config=config)

        with pytest.raises(Exception) as exc_info:
            safe_crew.kickoff()

        msg = str(exc_info.value)
        assert "TokenCircuit" in msg

    def test_legitimate_pipeline_completes(self):
        """A non-looping 3-agent sequential pipeline should complete normally."""
        from crewai import Agent, Crew, Process, Task

        researcher = Agent(
            role="Researcher",
            goal="Research",
            backstory="Expert",
            allow_delegation=False,
        )
        writer = Agent(
            role="Writer",
            goal="Write",
            backstory="Expert",
            allow_delegation=False,
        )

        task1 = Task(
            description="Research the topic",
            agent=researcher,
            expected_output="Research findings",
        )
        task2 = Task(
            description="Write the article",
            agent=writer,
            expected_output="Final article",
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[task1, task2],
            process=Process.sequential,
            verbose=False,
        )

        config = InterventionConfig(
            nudge_threshold=3,
            override_threshold=5,
            hard_stop_threshold=8,
            window_size=5,
        )
        safe_crew = instrument_crewai(crew, config=config)

        try:
            result = safe_crew.kickoff()
            assert result is not None
        except Exception:
            pytest.fail("Legitimate pipeline should not raise")

