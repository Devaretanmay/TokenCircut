import pytest
from unittest.mock import MagicMock

from tokencircuit import instrument_crewai, InterventionConfig
from tokencircuit.exceptions import TokenCircuitError

def test_instrument_crewai_hooks_agents():
    # Mock CrewAI agent
    class MockAgent:
        def __init__(self, role):
            self.role = role
            self.step_callback = None
            self.called_original = False
            
        def original_callback(self, step):
            self.called_original = True

    class MockCrew:
        def __init__(self, agents):
            self.agents = agents

    agent1 = MockAgent("Researcher")
    agent2 = MockAgent("Writer")
    agent2.step_callback = agent2.original_callback
    
    crew = MockCrew([agent1, agent2])
    
    # Instrument
    config = InterventionConfig(audit_mode=True)
    instrumented_crew = instrument_crewai(crew, config=config)
    
    assert instrumented_crew is crew
    assert crew.agents[0].step_callback is not None
    assert crew.agents[1].step_callback is not None
    
    # Trigger callback to verify original is called
    crew.agents[1].step_callback({"log": "testing"})
    assert crew.agents[1].called_original is True

def test_crewai_adapter_hard_stops_on_loop():
    class MockAgent:
        def __init__(self, role):
            self.role = role
            self.step_callback = None
            
    crew = MockCrew([MockAgent("Tester")])
    
    config = InterventionConfig(
        nudge_threshold=1,
        override_threshold=2,
        hard_stop_threshold=3,
        window_size=5,
    )
    instrument_crewai(crew, config=config)
    
    agent = crew.agents[0]
    
    # Simulate a loop: Agent repeatedly tries to call a tool that fails
    class MockStep:
        def __init__(self, tool, tool_input, result):
            self.tool = tool
            self.tool_input = tool_input
            self.result = result
            self.log = f"Calling {tool}"
            
    step = MockStep("fetch_data", {"q": "test"}, "Error: timeout")
    
    # Turn 1: Pass
    agent.step_callback(step)
    
    # Turn 2: Nudge (logs warning, doesn't throw)
    agent.step_callback(step)
    
    # Turn 3: Override (logs warning, doesn't throw)
    agent.step_callback(step)
    
    # Turn 4: Hard Stop
    with pytest.raises(TokenCircuitError) as excinfo:
        agent.step_callback(step)
        
    assert "HARD_STOP" in str(excinfo.value)

class MockCrew:
    def __init__(self, agents):
        self.agents = agents