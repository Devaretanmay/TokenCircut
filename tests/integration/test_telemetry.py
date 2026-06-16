import pytest
import sys
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from tokencircuit import InterventionEngine, InterventionConfig
from tokencircuit.state_schema import default_intervention_state
from tokencircuit.types import SignalType

def test_engine_emits_otel_spans():
    # Mock tracer components
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    with patch("tokencircuit.engine._get_tracer", return_value=mock_tracer):
        config = InterventionConfig(max_tokens_per_turn=10) # Set low to force RUNAWAY_GENERATION
        engine = InterventionEngine(config=config)
        
        messages = [
            HumanMessage(content="Test"),
            AIMessage(content="A" * 100), # > 10 tokens -> RUNAWAY_GENERATION -> HARD_STOP
        ]
        state = {"_tc_intervention": default_intervention_state()}
        
        engine.process(messages, state, thread_id="otel_test", node_name="agent")
        
        # Verify span was created
        mock_tracer.start_as_current_span.assert_called_once()
        args, kwargs = mock_tracer.start_as_current_span.call_args
        assert args[0] == "TokenCircuit.Intervention"
        assert kwargs["attributes"]["thread_id"] == "otel_test"
        assert kwargs["attributes"]["node_name"] == "agent"
        
        # Verify span events
        mock_span.set_attribute.assert_called_with("intervention.stage", "HARD_STOP")
        mock_span.add_event.assert_any_call("SignalDetected", {"signal.type": SignalType.RUNAWAY_GENERATION.value})
