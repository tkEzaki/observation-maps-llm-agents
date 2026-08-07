"""Serialization-binding ladder: four serializations of one histogram.

Why this exists
---------------
The serialization-length control (``serialization_length_controls``) refuted a
monotone prompt-length account of the encoding effect, but it left one question
open. Its compact histogram condition sits far from every other condition in
response, and that condition differs from the standard histogram condition in
two ways at once: it is 458 characters shorter, and it binds each mass to its
bin positionally rather than by an explicit label. Only the length half of that
was excluded.

This ladder separates the two. All four conditions serialize the *same* 24 bin
masses at the same precision, so the retained information is identical
throughout; they differ only in how each mass is bound to its bin, and their
prompt lengths are deliberately arranged so that length and binding make
opposite predictions:

    condition               binding of mass to bin        form
    ----------------------  ----------------------------  -------------------
    centers_compact         positional, grid stated by     one bare row
                            rule in the header
    centers_indexed_row     explicit bin index per mass    one row, b<k>=mass
    centers_centered_row    explicit bin centre per mass    one row, centre:mass
    centers_standard        explicit index and centre      one labelled line
                            per mass                       per bin

The two anchor conditions are imported unchanged from
``serialization_length_controls``, so ``centers_standard`` remains byte
identical to the primary ``centers_24_standard`` encoding of the main design
and ``centers_compact`` remains byte identical to the arm already acquired.

The crossing
------------
``centers_indexed_row`` sits only about a hundred characters from
``centers_compact`` but several hundred from ``centers_standard``. A prompt
length account therefore predicts that it responds like the compact condition,
while a binding account predicts it responds like the standard one, because it
is the shortest condition in which every mass carries an explicit label. The
prespecified contrast is the difference of those two distances, and its sign
alone separates the two accounts. The length gaps are arranged to favour the
length account a priori, so a positive sign is evidence against it that does
not depend on any calibration between characters and response distance.

Scope note (deliberate, carried over)
-------------------------------------
The moment order does not appear anywhere in this ladder. Every condition
carries the full 24-bin histogram: this is a control on serialization at fixed
information, not on how much of the field is retained.
"""

from __future__ import annotations

import numpy as np

from circlemap.observation import RelativePhaseHistogram
from circlemap.representations import _instruction, _rebinned_centers
from circlemap.serialization_length_controls import (
    DECIMALS,
    N_BINS,
    _DESCRIPTIONS,
    _HEADERS,
    centers_payload_compact,
    centers_payload_standard,
)

SBC_VARIANTS = (
    "centers_compact",
    "centers_indexed_row",
    "centers_centered_row",
    "centers_standard",
)

#: Every condition serializes the same content class.
SBC_CONTENT = {v: "histogram24" for v in SBC_VARIANTS}

#: How each mass is bound to its bin. This is the axis under test.
SBC_BINDING = {
    "centers_compact": "positional",
    "centers_indexed_row": "explicit_index",
    "centers_centered_row": "explicit_centre",
    "centers_standard": "explicit_index_and_centre",
}

#: Line structure, kept as a separate label so it is never conflated with
#: binding: three conditions are single-row, only the standard one is per-line.
SBC_FORM = {
    "centers_compact": "row",
    "centers_indexed_row": "row",
    "centers_centered_row": "row",
    "centers_standard": "lines",
}

#: The two conditions inherited from the serialization-length control, which
#: must stay byte identical to what was already acquired.
SBC_ANCHORS = ("centers_compact", "centers_standard")
SBC_NEW = ("centers_indexed_row", "centers_centered_row")


# --------------------------------------------------------------------------
# payload builders for the two new rungs
# --------------------------------------------------------------------------


def centers_payload_indexed_row(histogram: RelativePhaseHistogram) -> str:
    """One row, each mass carrying its bin index: ``b00=0.000000,...``.

    Same 24 masses at the same precision as every other condition. The bin
    grid is stated exactly in the header, as in the compact condition, so no
    field information is added; what is added is an explicit label binding
    each mass to a bin.
    """
    _, fractions = _rebinned_centers(
        histogram, n_bins=N_BINS, origin_shift_bins=0.0
    )
    return ",".join(
        f"b{index:02d}={value:.{DECIMALS}f}" for index, value in enumerate(fractions)
    )


def centers_payload_centered_row(histogram: RelativePhaseHistogram) -> str:
    """One row, each mass carrying its bin centre: ``-3.010693:0.000000,...``.

    The centres are the same fixed grid printed by the standard condition, at
    the same precision, so again no field information is added relative to the
    compact condition; the binding is by printed centre rather than by index.
    """
    centers, fractions = _rebinned_centers(
        histogram, n_bins=N_BINS, origin_shift_bins=0.0
    )
    return ",".join(
        f"{center:+.{DECIMALS}f}:{value:.{DECIMALS}f}"
        for center, value in zip(centers, fractions)
    )


_SBC_HEADERS = {
    "centers_compact": _HEADERS["centers_compact"],
    "centers_standard": _HEADERS["centers_standard"],
    "centers_indexed_row": (
        "Relative phase distribution by bin center, compact row with bin "
        "indices. The 24 bins are equal and span -pi to +pi; bin k is centered "
        "at -pi+(k+0.5)*2*pi/24, for k = 0..23:\n"
    ),
    "centers_centered_row": (
        "Relative phase distribution by bin center, compact row of "
        "center:mass pairs in radians:\n"
    ),
}

_SBC_PAYLOADS = {
    "centers_compact": centers_payload_compact,
    "centers_indexed_row": centers_payload_indexed_row,
    "centers_centered_row": centers_payload_centered_row,
    "centers_standard": centers_payload_standard,
}


def build_sbc_prompt(variant: str, histogram: RelativePhaseHistogram) -> str:
    if variant not in SBC_VARIANTS:
        raise ValueError(f"unknown serialization-binding variant: {variant}")
    return (
        _instruction(_DESCRIPTIONS["histogram24"])
        + _SBC_HEADERS[variant]
        + _SBC_PAYLOADS[variant](histogram)
    )


# --------------------------------------------------------------------------
# decoders: used offline to prove all four conditions carry the same masses
# --------------------------------------------------------------------------


def decode_masses(variant: str, prompt: str) -> np.ndarray:
    """Recover the 24 bin masses from any rung of the ladder."""
    payload = prompt.split(_SBC_HEADERS[variant], 1)[1]
    if variant == "centers_compact":
        return np.asarray([float(x) for x in payload.strip().split(",")])
    if variant == "centers_indexed_row":
        out = np.zeros(N_BINS, dtype=np.float64)
        for chunk in payload.strip().split(","):
            head, value = chunk.split("=")
            out[int(head[1:])] = float(value)
        return out
    if variant == "centers_centered_row":
        return np.asarray(
            [float(chunk.split(":")[1]) for chunk in payload.strip().split(",")]
        )
    return np.asarray(
        [float(line.split(":")[1]) for line in payload.splitlines() if line.strip()]
    )


def decode_centres(variant: str, prompt: str) -> np.ndarray:
    """Bin centres, printed where the condition prints them and implied otherwise."""
    payload = prompt.split(_SBC_HEADERS[variant], 1)[1]
    width = 2.0 * np.pi / N_BINS
    implied = -np.pi + width * (np.arange(N_BINS) + 0.5)
    if variant in ("centers_compact", "centers_indexed_row"):
        return implied
    if variant == "centers_centered_row":
        return np.asarray(
            [float(chunk.split(":")[0]) for chunk in payload.strip().split(",")]
        )
    return np.asarray(
        [
            float(line.split("center")[1].split(":")[0])
            for line in payload.splitlines()
            if line.strip()
        ]
    )
