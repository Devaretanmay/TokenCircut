"""BudgetEnforcer — tracks cumulative token costs locally."""

from __future__ import annotations

from typing import Dict


class BudgetExceededError(RuntimeError):
    """Raised when the cumulative token budget is exceeded."""

    pass


class BudgetEnforcer:
    """
    Tracks token consumption and enforces dollar-denominated budgets.

    Zero-trust: tracks everything locally in RAM.
    """

    def __init__(
        self,
        max_budget_usd: float = 0.0,
        token_pricing: Dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            max_budget_usd: Maximum allowed spend in USD. 0.0 means unlimited.
            token_pricing: Map of model_name to price per 1M tokens.
        """
        self.max_budget_usd = max_budget_usd
        self.token_pricing = token_pricing or {
            "gpt-4o": 5.0,
            "gpt-4o-mini": 0.15,
            "claude-3-5-sonnet": 3.0,
        }
        self._current_spend_usd = 0.0

    def record_usage(self, model: str, tokens: int) -> float:
        """
        Record token usage and check budget.

        Returns:
            The total current spend in USD.
        """
        price_per_1m = self.token_pricing.get(model, 5.0)  # Default to gpt-4o price
        cost = (tokens / 1_000_000) * price_per_1m
        self._current_spend_usd += cost

        if self.max_budget_usd > 0 and self._current_spend_usd >= self.max_budget_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${self._current_spend_usd:.4f} > "
                f"${self.max_budget_usd:.4f}"
            )

        return self._current_spend_usd

    @property
    def current_spend(self) -> float:
        return self._current_spend_usd

    def reset(self) -> None:
        self._current_spend_usd = 0.0
