import pytest
from unittest.mock import Mock, MagicMock

from tokencircuit.adapters.crewai import CrewAIInterventionAdapter
from tokencircuit.engine import InterventionConfig

def test_crewai_adapter_closure_fix():
    # Create mock agents
    agent1 = MagicMock()
    agent1.role = "agent1_role"
    agent1.step_callback = None

    agent2 = MagicMock()
    agent2.role = "agent2_role"
    agent2.step_callback = None

    agent3 = MagicMock()
    agent3.role = "agent3_role"
    agent3.step_callback = None

    # Create mock crew
    mock_crew = MagicMock()
    mock_crew.agents = [agent1, agent2, agent3]
    mock_crew.id = "test_crew_id"

    # Create adapter
    config = InterventionConfig(agency_id="test_agency")
    adapter = CrewAIInterventionAdapter(mock_crew, config=config)

    # Apply adapter
    adapter.apply()

    # Get the assigned callbacks
    cb1 = agent1.step_callback
    cb2 = agent2.step_callback
    cb3 = agent3.step_callback

    assert cb1 is not None
    assert cb2 is not None
    assert cb3 is not None

    # Call callbacks to see which role they pass to _intercept_step
    # We mock _intercept_step to avoid triggering the full pipeline with mock data
    adapter._intercept_step = Mock()

    step1 = MagicMock()
    cb1(step1)
    adapter._intercept_step.assert_called_with(step1, "agent1_role")

    adapter._intercept_step.reset_mock()
    step2 = MagicMock()
    cb2(step2)
    adapter._intercept_step.assert_called_with(step2, "agent2_role")

    adapter._intercept_step.reset_mock()
    step3 = MagicMock()
    cb3(step3)
    adapter._intercept_step.assert_called_with(step3, "agent3_role")
