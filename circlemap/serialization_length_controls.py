"""Serialization-length controls: prompt length varied at fixed information.

Motivation
----------
The three primary observation maps (moments, centers, intervals) differ both in
how much of the relative-phase field they retain and in how long the resulting
prompt is.  The same-information control already reported (Fig. 5) varies
*layout* and *neutral padding* at fixed moment values, but the padded condition
confounds length with context volume and with the position of the relevant
numbers.

This module supplies a cleaner, purely serialization-based manipulation.  Each
of the two content classes is written in a standard (long) and a compact
(short) form:

    content class          standard form           compact form
    ---------------------  ----------------------  ----------------------
    three circular moments moments_standard        moments_compact
    24 histogram masses    centers_standard        centers_compact

Within a content class the two forms are re-serializations of *the same
numbers at the same precision*: no filler text is added, no derived quantity is
introduced, and no digit is dropped.  Only the layout, the labelling and hence
the character count change.

Crucially, the resulting length ordering crosses the content ordering.  On the
frozen 48-field panel the prompts are 752, 789, 971 and 1429 characters for
``moments_compact``, ``moments_standard``, ``centers_compact`` and
``centers_standard``.  ``centers_compact`` therefore sits 458 characters from
its own content class partner but only 182 characters from ``moments_standard``:
on the length axis it is much closer to the moments arms, while on the content
axis it is identical to ``centers_standard``.  A pure prompt-length account of
the encoding effect predicts that response distances follow the length ordering;
a content account predicts they follow the content ordering.  The two
predictions are crossed by construction, so a single acquisition separates them.

Scope note (deliberate)
-----------------------
The moment order is held at three throughout.  Order is not a length knob: the
binned moments are a discrete Fourier transform of the 24 bin masses, so order
12 reproduces the histogram exactly (verified to 3e-16 on the frozen field
panel) and would no longer be a summary of the field but a re-notation of the
centers condition, which is already an arm of the primary design.  Varying the
order would therefore vary the retained information, which is precisely what
this control holds fixed.
"""

from __future__ import annotations

import numpy as np

from circlemap.observation import RelativePhaseHistogram
from circlemap.representations import (
    _instruction,
    _rebinned_centers,
    _serialize_moments_histogram,
)

SLC_VARIANTS = (
    "moments_standard",
    "moments_compact",
    "centers_standard",
    "centers_compact",
)

#: Which content class each variant serializes.
SLC_CONTENT = {
    "moments_standard": "moments3",
    "moments_compact": "moments3",
    "centers_standard": "histogram24",
    "centers_compact": "histogram24",
}

#: Which serialization form each variant uses.
SLC_FORM = {
    "moments_standard": "standard",
    "moments_compact": "compact",
    "centers_standard": "standard",
    "centers_compact": "compact",
}

N_BINS = 24
DECIMALS = 6
MOMENT_ORDER = 3


# --------------------------------------------------------------------------
# payload builders (content only, no instruction block)
# --------------------------------------------------------------------------


def moments_payload_standard(histogram: RelativePhaseHistogram) -> str:
    """One labelled line per moment component (the primary moments encoding)."""
    return _serialize_moments_histogram(
        histogram, order=MOMENT_ORDER, decimals=DECIMALS
    )


def moments_payload_compact(histogram: RelativePhaseHistogram) -> str:
    """The same six numbers on a single line, same precision."""
    raw = moments_payload_standard(histogram)
    values: dict[tuple[int, str], str] = {}
    for line in raw.splitlines():
        name, value = line.split(":")
        parts = name.strip().split("_")
        values[(int(parts[1]), parts[2])] = value.strip()
    return "; ".join(
        f"m{h}=({values[(h, 'cos')]},{values[(h, 'sin')]})"
        for h in range(1, MOMENT_ORDER + 1)
    )


def centers_payload_standard(histogram: RelativePhaseHistogram) -> str:
    """One labelled line per bin: index, centre and mass (primary encoding)."""
    centers, fractions = _rebinned_centers(
        histogram, n_bins=N_BINS, origin_shift_bins=0.0
    )
    return "\n".join(
        f"bin_{index:02d} center {center:+.{DECIMALS}f}: {value:.{DECIMALS}f}"
        for index, (center, value) in enumerate(zip(centers, fractions))
    )


def centers_payload_compact(histogram: RelativePhaseHistogram) -> str:
    """The same 24 masses as one comma-separated row, same precision.

    The bin centres are not listed because they are the same fixed grid in
    every field and are stated exactly in the header, so no field information
    is dropped relative to :func:`centers_payload_standard`.
    """
    _, fractions = _rebinned_centers(
        histogram, n_bins=N_BINS, origin_shift_bins=0.0
    )
    return ",".join(f"{v:.{DECIMALS}f}" for v in fractions)


# --------------------------------------------------------------------------
# prompt builders
# --------------------------------------------------------------------------

#: The standard arms reuse the primary encoding descriptions verbatim, so that
#: ``moments_standard`` and ``centers_standard`` are byte-identical to the
#: ``moments_m1_m3`` and ``centers_24_standard`` prompts of the main design.
_DESCRIPTIONS = {
    "moments3": (
        "The values are the first three normalized circular moments "
        "of the distribution of other phases relative to you."
    ),
    "histogram24": (
        f"The table gives normalized relative-phase mass at {N_BINS} circular "
        "bin centers in radians."
    ),
}

_HEADERS = {
    "moments_standard": "Relative phase distribution as circular moments:\n",
    "moments_compact": (
        "Relative phase distribution as circular moments, compact row "
        "(m<k>=(cos,sin)):\n"
    ),
    "centers_standard": "Relative phase distribution by bin center:\n",
    "centers_compact": (
        "Relative phase distribution by bin center, compact row. The 24 bins "
        "are equal and span -pi to +pi; the k-th value is the mass of the bin "
        "centered at -pi+(k-0.5)*2*pi/24, for k = 1..24:\n"
    ),
}

_PAYLOADS = {
    "moments_standard": moments_payload_standard,
    "moments_compact": moments_payload_compact,
    "centers_standard": centers_payload_standard,
    "centers_compact": centers_payload_compact,
}


def build_slc_prompt(variant: str, histogram: RelativePhaseHistogram) -> str:
    if variant not in SLC_VARIANTS:
        raise ValueError(f"unknown serialization-length control variant: {variant}")
    return (
        _instruction(_DESCRIPTIONS[SLC_CONTENT[variant]])
        + _HEADERS[variant]
        + _PAYLOADS[variant](histogram)
    )


# --------------------------------------------------------------------------
# decoders: used offline to prove that the compact and standard forms of a
# content class carry byte-identical information.
# --------------------------------------------------------------------------


def decode_moments(variant: str, prompt: str) -> np.ndarray:
    """Recover the six moment components from either moments serialization."""
    payload = prompt.split(_HEADERS[variant], 1)[1]
    out = np.zeros((MOMENT_ORDER, 2), dtype=np.float64)
    if SLC_FORM[variant] == "standard":
        for line in payload.splitlines():
            name, value = line.split(":")
            parts = name.strip().split("_")
            out[int(parts[1]) - 1, 0 if parts[2] == "cos" else 1] = float(value)
    else:
        for chunk in payload.split(";"):
            head, rest = chunk.strip().split("=", 1)
            cos_s, sin_s = rest.strip("()").split(",")
            out[int(head[1:]) - 1] = (float(cos_s), float(sin_s))
    return out


def decode_centers(variant: str, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    """Recover the 24 centres and masses from either centers serialization."""
    payload = prompt.split(_HEADERS[variant], 1)[1]
    if SLC_FORM[variant] == "standard":
        centers, masses = [], []
        for line in payload.splitlines():
            head, mass = line.split(":")
            centers.append(float(head.split("center")[1]))
            masses.append(float(mass))
        return np.asarray(centers), np.asarray(masses)
    masses = np.asarray([float(x) for x in payload.strip().split(",")])
    width = 2.0 * np.pi / N_BINS
    centers = -np.pi + width * (np.arange(N_BINS) + 0.5)
    return centers, masses


def decode_information(variant: str, prompt: str):
    """Return the field-dependent content of a prompt, form-independent."""
    if SLC_CONTENT[variant] == "moments3":
        return decode_moments(variant, prompt)
    return decode_centers(variant, prompt)
