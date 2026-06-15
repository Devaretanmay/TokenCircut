import logging
from typing import Any, AsyncIterator, Optional

from ..config import TokenCircuitConfig, load_config
from ..detectors.pipeline import DetectionPipeline
from ..exceptions import TokenCircuitError
from ..otel.hash_utils import compute_action_hash, extract_tool_type_signature
from ..telemetry import compute_cost_estimate

logger = logging.getLogger("tokencircuit")


class LangGraphInterceptor:
    def __init__(
        self,
        graph: Any,
        config: Optional[TokenCircuitConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._graph = graph
        if config is None:
            config = load_config(api_key)
        self._config = config
        self._pipeline = DetectionPipeline(config, "langgraph", api_key=api_key)
        self._node_names = self._discover_nodes()

    def _discover_nodes(self) -> set[str]:
        try:
            g = self._graph.get_graph()
            names: set[str] = set()
            for node_id, node_data in g.nodes.items():
                name = node_data.name if hasattr(node_data, "name") else node_id
                if name and not name.startswith("__"):
                    names.add(name)
            return names
        except Exception:
            logger.warning(
                "TokenCircuit: could not discover node names from compiled graph"
            )
            return set()

    def _extract_tool_call(
        self, state: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1] if isinstance(messages, list) else messages
        if isinstance(last, dict):
            tool_calls = last.get("tool_calls", [])
            return tool_calls[-1] if tool_calls else None
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

            state = state if isinstance(state, dict) else {}
            action_hash = compute_action_hash(state)
            tool_call = self._extract_tool_call(state)
            tool_sig = extract_tool_type_signature(tool_call)

            result = self._pipeline.record_step(
                agent_id, node_name, action_hash, tool_sig
            )
            if result is not None:
                model_name = self._extract_model_name(state)
                iterations_saved = max(self._config.max_repeats - result.iteration, 1)
                tokens_saved, cost_saved = compute_cost_estimate(
                    model_name, iterations_saved
                )
                msg = (
                    f"TokenCircuit [{result.signal_type}]: "
                    f"node='{result.node_name}' at iteration {result.iteration} "
                    f"(est. {tokens_saved} tokens saved, ~${cost_saved:.4f})"
                )
                logger.warning(msg)
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
