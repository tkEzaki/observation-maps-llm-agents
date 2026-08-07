"""Stage B fixed-field stimuli, prompt, and strict response contract."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .observation import RelativePhaseHistogram, serialize_histogram, wrap_phase

ACTION_VALUES = {"retard": -1, "stay": 0, "advance": 1}
PROMPT_VERSION = "response-law-v0.1"


@dataclass(frozen=True)
class ParsedSocialAction:
    label: str
    value: int


def fixed_field_histogram(
    offset_radians: float,
    concentration: float,
    *,
    n_bins: int = 24,
    synthetic_peer_count: int = 240,
) -> RelativePhaseHistogram:
    """Create a deterministic discretized von Mises field around an offset."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if not np.isfinite(offset_radians):
        raise ValueError("offset_radians must be finite")
    if not np.isfinite(concentration) or concentration < 0:
        raise ValueError("concentration must be finite and non-negative")
    if synthetic_peer_count <= 0:
        raise ValueError("synthetic_peer_count must be positive")

    edges = np.linspace(-np.pi, np.pi, n_bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    log_weights = concentration * np.cos(
        wrap_phase(centers - offset_radians)
    )
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    fractions = weights / np.sum(weights)
    return RelativePhaseHistogram(
        edges=edges,
        fractions=fractions,
        peer_count=synthetic_peer_count,
    )


def build_response_prompt(histogram: RelativePhaseHistogram) -> str:
    """Serialize the frozen v0.1 response-law prompt."""
    distribution = serialize_histogram(histogram, decimals=6)
    return (
        "You are an interaction operator for a phase on a circle.\n"
        "The table gives the normalized distribution of other phases relative "
        "to you in radians.\n"
        "Negative relative phase is behind you; positive relative phase is "
        "ahead of you.\n"
        "Choose one social correction in response to this distribution:\n"
        '- \"advance\": move in the positive phase direction\n'
        '- \"stay\": apply no social correction\n'
        '- \"retard\": move in the negative phase direction\n'
        "Use only the relative distribution below. Return exactly one JSON "
        "object and no other text.\n"
        'Required schema: {\"social_action\":\"advance|stay|retard\"}\n'
        "Relative phase distribution:\n"
        f"{distribution}"
    )


def parse_social_action(raw_text: str) -> ParsedSocialAction:
    """Parse only the frozen one-field JSON response schema."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("response is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    if set(payload) != {"social_action"}:
        raise ValueError("response must contain only social_action")
    label = payload["social_action"]
    if label not in ACTION_VALUES:
        raise ValueError(
            "social_action must be advance, stay, or retard"
        )
    return ParsedSocialAction(label=label, value=ACTION_VALUES[label])
