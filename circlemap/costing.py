"""Conservative pre-run token and monetary cost estimates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class CostEstimate:
    planned_calls: int
    max_attempts_per_call: int
    input_token_estimation_method: str
    estimated_input_tokens: int
    maximum_output_tokens: int
    input_price_per_million: float
    output_price_per_million: float
    estimated_base_cost_usd: float
    maximum_retry_cost_usd: float

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_tokens_from_characters(text: str) -> int:
    """Conservative heuristic for prompts dominated by decimal tables."""
    return max(1, math.ceil(len(text) / 2.5) + 16)


def estimate_cost(
    prompts: Sequence[str],
    *,
    max_output_tokens_per_call: int,
    max_attempts_per_call: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> CostEstimate:
    if not prompts:
        raise ValueError("at least one prompt is required")
    if max_output_tokens_per_call <= 0:
        raise ValueError("max_output_tokens_per_call must be positive")
    if max_attempts_per_call <= 0:
        raise ValueError("max_attempts_per_call must be positive")
    if input_price_per_million < 0 or output_price_per_million < 0:
        raise ValueError("token prices must be non-negative")

    input_tokens = sum(estimate_tokens_from_characters(p) for p in prompts)
    output_tokens = len(prompts) * max_output_tokens_per_call
    base_cost = (
        input_tokens * input_price_per_million
        + output_tokens * output_price_per_million
    ) / 1_000_000
    return CostEstimate(
        planned_calls=len(prompts),
        max_attempts_per_call=max_attempts_per_call,
        input_token_estimation_method="ceil(characters / 2.5) + 16",
        estimated_input_tokens=input_tokens,
        maximum_output_tokens=output_tokens,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        estimated_base_cost_usd=base_cost,
        maximum_retry_cost_usd=base_cost * max_attempts_per_call,
    )


def format_cost_estimate(estimate: CostEstimate) -> str:
    return "\n".join(
        [
            "Stage B pre-run cost estimate",
            f"  planned calls: {estimate.planned_calls:,}",
            (
                "  estimated input tokens: "
                f"{estimate.estimated_input_tokens:,}"
            ),
            (
                "  input estimate method: "
                f"{estimate.input_token_estimation_method}"
            ),
            (
                "  maximum output tokens: "
                f"{estimate.maximum_output_tokens:,}"
            ),
            (
                "  prices per 1M tokens: "
                f"input ${estimate.input_price_per_million:.4f}, "
                f"output ${estimate.output_price_per_million:.4f}"
            ),
            (
                "  estimated base cost: "
                f"${estimate.estimated_base_cost_usd:.4f}"
            ),
            (
                f"  retry ceiling ({estimate.max_attempts_per_call} attempts): "
                f"${estimate.maximum_retry_cost_usd:.4f}"
            ),
        ]
    )
