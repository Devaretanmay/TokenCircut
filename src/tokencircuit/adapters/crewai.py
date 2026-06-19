"""CrewAI native adapter."""

from __future__ import annotations

import logging
from typing import Any

from ..engine import InterventionConfig, InterventionEngine, TokenCircuitError
from ..state_schema import default_intervention_state, tc_state_reducer
from ..types import InterventionStage

logger = logging.getLogger("tokencircuit.adapters.crewai")


def instrument_crewai(crew: Any, config: InterventionConfig | None = None) -> Any:
    """Instrument a CrewAI Crew instance with TokenCircuit."""
    engine = InterventionEngine(config=config or InterventionConfig())
    crew_cb = getattr(crew, "step_callback", None)

    crew_id = str(id(crew))
    transcript: list[dict[str, Any]] = []
    tc_state = default_intervention_state()

    def _process_step(step_output: Any) -> None:
        nonlocal tc_state
        text = str(step_output)
        transcript.append({"role": "assistant", "content": text})

        decision = engine.process(
            messages=transcript,
            state={"_tc_intervention": tc_state},
            thread_id=crew_id,
            node_name="crewai",
        )
        tc_state = tc_state_reducer(tc_state, decision.state_patch)

        if decision.stage == InterventionStage.HARD_STOP:
            raise TokenCircuitError(
                decision.termination_reason or "TokenCircuit HARD_STOP"
            )

    def tc_crew_callback(step_output: Any) -> Any:
        _process_step(step_output)
        if crew_cb:
            return crew_cb(step_output)
        return step_output

    crew.step_callback = tc_crew_callback

    if hasattr(crew, "agents"):
        for agent in crew.agents:
            agent_cb = getattr(agent, "step_callback", None)

            def _make_agent_cb(orig_cb: Any) -> Any:
                def _agent_tc_cb(step_output: Any) -> Any:
                    _process_step(step_output)
                    if orig_cb:
                        return orig_cb(step_output)
                    return step_output

                return _agent_tc_cb

            agent.step_callback = _make_agent_cb(agent_cb)

    return crew
