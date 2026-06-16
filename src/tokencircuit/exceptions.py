"""Exception hierarchy for TokenCircuit."""


class TokenCircuitError(RuntimeError):
    def __init__(
        self,
        message: str,
        signal_type: str = "",
        node_name: str = "",
        iteration: int = 0,
        state_hashes_window: list[str] | None = None,
        tool_signatures_window: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.signal_type = signal_type
        self.node_name = node_name
        self.iteration = iteration
        self.state_hashes_window = state_hashes_window or []
        self.tool_signatures_window = tool_signatures_window or []


__all__ = [
    "TokenCircuitError",
]
