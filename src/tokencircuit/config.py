"""Configuration for TokenCircuit (deprecated — use InterventionConfig instead)."""

import warnings
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "TokenCircuitConfig",
]


@dataclass
class TokenCircuitConfig:
    max_repeats: int = 5
    window_size: int = 5
    agency_id: Optional[str] = None
    client_id: Optional[str] = None
    model_name: str = "unknown"
    telemetry_enabled: bool = field(default=True)

    def __post_init__(self) -> None:
        warnings.warn(
            "TokenCircuitConfig is deprecated. Use InterventionConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.max_repeats < 1:
            raise ValueError("max_repeats must be >= 1")
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
