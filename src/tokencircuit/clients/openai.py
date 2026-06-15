import hashlib
import json
import logging
from typing import Any, Optional

from ..config import TokenCircuitConfig, load_config
from ..detectors.pipeline import DetectionPipeline
from ..exceptions import TokenCircuitError

logger = logging.getLogger("tokencircuit")


class TokenCircuitClient:
    def __init__(
        self,
        client: Any,
        config: Optional[TokenCircuitConfig] = None,
        api_key: Optional[str] = None,
        session_id: str = "default",
    ) -> None:
        self._client = client
        if config is None:
            config = load_config(api_key)
        self._config = config
        self._pipeline = DetectionPipeline(config, "openai", api_key=api_key)
        self._session_id = session_id
        self._wrap_chat()

    def _wrap_chat(self) -> None:
        original_chat = self._client.chat
        original_completions = original_chat.completions
        original_create = original_completions.create

        def wrapped_create(*args: Any, **kwargs: Any) -> Any:
            response = original_create(*args, **kwargs)
            self._check_response(kwargs.get("model", "unknown"), response)
            return response

        wrapped_completions = _Proxy(original_completions, {"create": wrapped_create})
        wrapped_chat = _Proxy(original_chat, {"completions": wrapped_completions})
        self.chat = wrapped_chat

    def _check_response(self, model: str, response: Any) -> None:
        state_hash, tool_sig = self._hash_response(response)
        result = self._pipeline.record_step(
            self._session_id, model, state_hash, tool_sig
        )
        if result is not None:
            msg = (
                f"TokenCircuit [{result.signal_type}]: "
                f"model='{model}' at iteration {result.iteration}"
            )
            logger.warning(msg)
            raise TokenCircuitError(
                msg,
                signal_type=result.signal_type or "",
                node_name=result.node_name,
                iteration=result.iteration,
                state_hashes_window=result.state_hashes_window,
                tool_signatures_window=result.tool_signatures_window,
            )

    def _hash_response(self, response: Any) -> tuple[str, str]:
        hasher = hashlib.sha256()
        tool_sigs: list[str] = []

        for choice in response.choices:
            content = choice.message.content or ""
            hasher.update(content.encode())
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    args = json.dumps(tc.function.arguments, sort_keys=True)
                    tool_sigs.append(f"{tc.function.name}({args})")

        tool_sig = "|".join(tool_sigs) if tool_sigs else "NO_TOOL_CALL"
        return hasher.hexdigest(), tool_sig

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _Proxy:
    def __init__(self, original: Any, overrides: dict[str, Any]):
        self._original = original
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._original, name)
