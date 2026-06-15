import logging
from typing import Any, AsyncIterator, Optional

from ..config import TokenCircuitConfig
from ..detectors.composite import CompositeDetector, DetectionResult
from ..exceptions import TokenCircuitError
from ..otel.hash_utils import compute_action_hash, extract_tool_type_signature
from ..ring_buffer import RingBuffer
from ..telemetry import TelemetryEvent, compute_cost_estimate, emit_event_async

logger = logging.getLogger("tokencircuit")


class LangGraphInterceptor:
    def __init__(
        self,
        graph: Any,
        config: Optional[TokenCircuitConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._graph = graph
        if config is not None:
            self._config = config
        else:
            from ..config import load_config
            self._config = load_config(api_key)
        self._detector = CompositeDetector(self._config.window_size)
        self._buffers: dict[str, RingBuffer] = {}
        self._iteration: dict[str, int] = {}
        self._api_key = api_key
        self._discover_nodes()

    def _discover_nodes(self) -> None:
        self._node_names: set[str] = set()
        try:
            g = self._graph.get_graph()
            for node_id, node_data in g.nodes.items():
                name = node_data.name if hasattr(node_data, "name") else node_id
                if name and not name.startswith("__"):
                    self._node_names.add(name)
        except Exception:
            logger.warning(
                "TokenCircuit: could not discover node names from compiled graph"
            )

    def _get_buffer(self, agent_id: str, node_name: str) -> RingBuffer:
        key = f"{agent_id}:{node_name}"
        if key not in self._buffers:
            self._buffers[key] = RingBuffer(maxlen=self._config.window_size)
        return self._buffers[key]

    def _increment_iteration(self, agent_id: str, node_name: str) -> int:
        key = f"{agent_id}:{node_name}"
        self._iteration[key] = self._iteration.get(key, 0) + 1
        return self._iteration[key]

    def _extract_tool_call(
        self, state: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1] if isinstance(messages, list) else messages
        if isinstance(last, dict):
            tool_calls = last.get("tool_calls", [])
            if tool_calls:
                return tool_calls[-1]
            return None
        if hasattr(last, "tool_calls") and last.tool_calls:
            return last.tool_calls[-1]
        return None

    def _extract_model_name(self, state: dict[str, Any]) -> str:
        model = state.get("model_name") or state.get("model") or "unknown"
        if isinstance(model, str):
            return model
        if isinstance(model, dict):
            return model.get("model", "unknown")
        return "unknown"

    def _on_detection(
        self, result: DetectionResult, agent_id: str, model_name: str
    ) -> str:
        iterations_saved = self._config.max_repeats - result.iteration
        tokens_saved, cost_saved = compute_cost_estimate(
            model_name, max(iterations_saved, 1)
        )

        if (
            self._config.telemetry_enabled
            and self._config.agency_id
            and self._config.client_id
            and self._api_key
        ):
            event = TelemetryEvent(
                agency_id=self._config.agency_id,
                client_id=self._config.client_id,
                agent_framework="langgraph",
                signal_type=result.signal_type or "UNKNOWN",
                node_name=result.node_name,
                iterations_at_detection=result.iteration,
                model_name=model_name,
                estimated_tokens_saved=tokens_saved,
                estimated_cost_saved_usd=cost_saved,
            )
            emit_event_async(event, api_key=self._api_key)

        msg = (
            f"TokenCircuit [{result.signal_type}]: "
            f"node='{result.node_name}' at iteration {result.iteration} "
            f"(est. {tokens_saved} tokens saved, ~${cost_saved:.4f})"
        )
        logger.warning(msg)
        return msg

    async def astream(
        self,
        input: Any,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        if config is None:
            config = {}
        agent_id = config.get("configurable", {}).get(
            "thread_id", "default_agent"
        )

        stream = self._graph.astream(input, config, **kwargs)

        async for step_output in stream:
            node_name = None
            state = None

            if isinstance(step_output, dict) and self._node_names:
                for n in self._node_names:
                    if n in step_output:
                        node_name = n
                        state = step_output[n]
                        break

            if node_name is None:
                yield step_output
                continue

            action_hash = compute_action_hash(
                state if isinstance(state, dict) else {}
            )
            tool_call = self._extract_tool_call(
                state if isinstance(state, dict) else {}
            )
            tool_sig = extract_tool_type_signature(tool_call)
            iteration = self._increment_iteration(agent_id, node_name)

            buffer = self._get_buffer(agent_id, node_name)
            buffer.push({
                "state_hash": action_hash,
                "tool_type_signature": tool_sig,
                "iteration": iteration,
            })

            result = self._detector.evaluate(agent_id, node_name, buffer)
            if result is not None:
                model_name = self._extract_model_name(
                    state if isinstance(state, dict) else {}
                )
                msg = self._on_detection(result, agent_id, model_name)
                raise TokenCircuitError(msg)

            yield step_output

    async def astream_events(
        self, input: Any, config: dict[str, Any], **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        async for event in self._graph.astream_events(input, config, **kwargs):
            yield event

    def ainvoke(
        self, input: Any, config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._async_stream_collect(input, config, **kwargs)
        )

    async def _async_stream_collect(
        self, input: Any, config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        result = None
        async for output in self.astream(input, config, **kwargs):
            result = output
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)
