import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("tokencircuit")

SUPABASE_CONFIG_URL = (
    "https://tokencircuit.supabase.co/rest/v1/agency_configs"
)


@dataclass
class TokenCircuitConfig:
    max_repeats: int = 5
    window_size: int = 5
    agency_id: Optional[str] = None
    client_id: Optional[str] = None
    model_name: str = "unknown"
    telemetry_enabled: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.max_repeats < 1:
            raise ValueError("max_repeats must be >= 1")
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")


def load_config(api_key: Optional[str] = None) -> TokenCircuitConfig:
    defaults = TokenCircuitConfig(max_repeats=5, window_size=5)

    if not api_key:
        return defaults

    try:
        import httpx

        resp = httpx.get(
            SUPABASE_CONFIG_URL,
            headers={
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
            },
            timeout=2.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            data = data[0] if len(data) > 0 else {}
        if not isinstance(data, dict):
            data = {}
        return TokenCircuitConfig(**{
            k: v for k, v in data.items()
            if k in TokenCircuitConfig.__dataclass_fields__
        })
    except Exception as exc:
        logger.warning(
            "TokenCircuit: config fetch failed (%s: %s), using defaults",
            type(exc).__name__, exc,
        )
        return defaults
