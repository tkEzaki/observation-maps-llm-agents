"""Same-information moments observation-map controls (serialization / length).

These variants keep circular-moment information identical to moments_m1_m3
and only change display format or prompt length.
"""

from __future__ import annotations

from circlemap.observation import RelativePhaseHistogram
from circlemap.representations import (
    _instruction,
    _serialize_moments_histogram,
    build_representation_prompt_from_histogram,
)

CONTROL_VARIANTS = (
    "moments_original",
    "moments_reformatted",
    "moments_length_matched",
)

# Midpoint of centers/intervals fixed lengths on the frozen 48-field panel.
LENGTH_MATCH_TARGET_CHARS = 1496


def _moment_values_block(histogram: RelativePhaseHistogram) -> str:
    return _serialize_moments_histogram(histogram, order=3, decimals=6)


def build_moments_original(histogram: RelativePhaseHistogram) -> str:
    return build_representation_prompt_from_histogram("moments_m1_m3", histogram)


def build_moments_reformatted(histogram: RelativePhaseHistogram) -> str:
    """Same moment numbers presented as a table (order: m1,m2,m3)."""
    raw = _moment_values_block(histogram)
    # Parse lines moment_k_cos / moment_k_sin
    vals: dict[tuple[int, str], str] = {}
    for line in raw.splitlines():
        # moment_1_cos: +0.123456
        name, value = line.split(":")
        name = name.strip()
        value = value.strip()
        parts = name.split("_")
        harm = int(parts[1])
        kind = parts[2]
        vals[(harm, kind)] = value
    rows = ["harmonic | cos | sin"]
    rows.append("---|---|---")
    for h in (1, 2, 3):
        rows.append(f"{h} | {vals[(h, 'cos')]} | {vals[(h, 'sin')]}")
    description = (
        "The table lists the first three normalized circular moments "
        "of the distribution of other phases relative to you "
        "(identical information to the line-list moments encoding)."
    )
    payload = "\n".join(rows)
    return (
        _instruction(description)
        + "Relative phase distribution as circular moments (table):\n"
        + payload
    )


def build_moments_length_matched(histogram: RelativePhaseHistogram) -> str:
    """Original moments payload plus neutral padding toward histogram length."""
    base = build_moments_original(histogram)
    filler = (
        "\nNote: length-matching placeholder only; ignore this line; "
        "it carries no phase information."
    )
    out = base
    # Deterministic: append filler until >= target, then truncate filler block
    # to exact target when possible without cutting the base.
    while len(out) < LENGTH_MATCH_TARGET_CHARS:
        out += filler
    if len(out) > LENGTH_MATCH_TARGET_CHARS:
        # Keep base intact; trim only padding.
        pad_needed = LENGTH_MATCH_TARGET_CHARS - len(base)
        if pad_needed <= 0:
            return base
        pad = (filler * ((pad_needed // len(filler)) + 2))[:pad_needed]
        out = base + pad
    return out


def build_control_prompt(variant: str, histogram: RelativePhaseHistogram) -> str:
    if variant == "moments_original":
        return build_moments_original(histogram)
    if variant == "moments_reformatted":
        return build_moments_reformatted(histogram)
    if variant == "moments_length_matched":
        return build_moments_length_matched(histogram)
    raise ValueError(f"unknown observation-map control variant: {variant}")
