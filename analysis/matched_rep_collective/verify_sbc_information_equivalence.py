"""Offline audit of the serialization-binding ladder.

Proves, without any model call, that

1. the two anchor conditions are byte identical to what was already acquired in
   the serialization-length control, and ``centers_standard`` is byte identical
   to the primary ``centers_24_standard`` encoding;
2. all four conditions decode to the same 24 bin masses, bit exact, so the
   retained information is identical along the whole ladder;
3. the length axis crosses the binding axis: ``centers_indexed_row`` is much
   closer in characters to ``centers_compact`` than to ``centers_standard``,
   while being the shortest condition that binds every mass to an explicit
   label. A prompt-length account and a binding account therefore predict
   opposite signs for the prespecified contrast.

Writes ``sbc_information_audit.json`` next to this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circlemap.costing import estimate_tokens_from_characters  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.representations import (  # noqa: E402
    build_representation_prompt_from_histogram,
)
from circlemap.serialization_binding_controls import (  # noqa: E402
    N_BINS,
    SBC_BINDING,
    SBC_FORM,
    SBC_VARIANTS,
    build_sbc_prompt,
    decode_centres,
    decode_masses,
)
from circlemap.serialization_length_controls import build_slc_prompt  # noqa: E402

FIELDS_PATH = ROOT / "analysis/matched_rep_collective/replay_fields_v0_1.json"
OUT_PATH = Path(__file__).with_name("sbc_information_audit.json")

#: The rung whose length and binding predictions disagree.
PIVOT = "centers_indexed_row"
POSITIONAL_ANCHOR = "centers_compact"
LABELLED_ANCHOR = "centers_standard"


def _histogram(field: dict) -> RelativePhaseHistogram:
    return RelativePhaseHistogram(
        edges=np.linspace(-np.pi, np.pi, N_BINS + 1),
        fractions=np.asarray(field["physical_histogram_24"], dtype=float),
        peer_count=int(field.get("peer_count", 16)),
    )


def main() -> int:
    fields = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))["fields"]

    lengths: dict[str, list[int]] = {v: [] for v in SBC_VARIANTS}
    tokens: dict[str, list[int]] = {v: [] for v in SBC_VARIANTS}
    anchors_identical = {
        "centers_compact_matches_slc": True,
        "centers_standard_matches_slc": True,
        "centers_standard_matches_primary": True,
    }
    max_mass_gap = 0.0
    max_centre_gap = 0.0

    for field in fields:
        hist = _histogram(field)
        prompts = {v: build_sbc_prompt(v, hist) for v in SBC_VARIANTS}
        for variant, prompt in prompts.items():
            lengths[variant].append(len(prompt))
            tokens[variant].append(int(estimate_tokens_from_characters(prompt)))

        anchors_identical["centers_compact_matches_slc"] &= prompts[
            "centers_compact"
        ] == build_slc_prompt("centers_compact", hist)
        anchors_identical["centers_standard_matches_slc"] &= prompts[
            "centers_standard"
        ] == build_slc_prompt("centers_standard", hist)
        anchors_identical["centers_standard_matches_primary"] &= prompts[
            "centers_standard"
        ] == build_representation_prompt_from_histogram("centers_24_standard", hist)

        ref_mass = decode_masses("centers_standard", prompts["centers_standard"])
        ref_centre = decode_centres("centers_standard", prompts["centers_standard"])
        for variant, prompt in prompts.items():
            max_mass_gap = max(
                max_mass_gap,
                float(np.max(np.abs(decode_masses(variant, prompt) - ref_mass))),
            )
            max_centre_gap = max(
                max_centre_gap,
                float(np.max(np.abs(decode_centres(variant, prompt) - ref_centre))),
            )

    def _stats(values: dict[str, list[int]]) -> dict[str, dict[str, float]]:
        return {
            v: {
                "min": int(np.min(values[v])),
                "median": float(np.median(values[v])),
                "max": int(np.max(values[v])),
            }
            for v in SBC_VARIANTS
        }

    med = {v: float(np.median(lengths[v])) for v in SBC_VARIANTS}
    to_positional = med[PIVOT] - med[POSITIONAL_ANCHOR]
    to_labelled = med[LABELLED_ANCHOR] - med[PIVOT]

    report = {
        "n_fields": len(fields),
        "fields_file": str(FIELDS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "variants": {
            v: {
                "content": "histogram24",
                "binding": SBC_BINDING[v],
                "form": SBC_FORM[v],
            }
            for v in SBC_VARIANTS
        },
        "prompt_chars": _stats(lengths),
        "prompt_tokens": _stats(tokens),
        "anchors_byte_identical": anchors_identical,
        "information_equivalence": {
            "max_abs_bin_mass_gap": max_mass_gap,
            "max_abs_bin_centre_gap_print_precision": max_centre_gap,
            "note": (
                "All four conditions carry the same 24 bin masses at the same "
                "precision. The centre gap is the six-decimal print precision of "
                "the conditions that list the centres, against the analytic grid "
                "the conditions that state it by rule imply."
            ),
        },
        "length_axis_crosses_binding_axis": {
            "pivot": PIVOT,
            "chars_pivot_to_positional_anchor": to_positional,
            "chars_pivot_to_labelled_anchor": to_labelled,
            "length_favours": POSITIONAL_ANCHOR if to_positional < to_labelled else LABELLED_ANCHOR,
            "ratio_labelled_over_positional": to_labelled / max(to_positional, 1e-9),
            "binding_favours": LABELLED_ANCHOR,
            "crossed": bool(to_positional < to_labelled),
            "note": (
                "The pivot is the shortest condition that binds every mass to an "
                "explicit label. It sits much closer in characters to the "
                "positional condition, so a prompt-length account predicts it "
                "responds like that one, while a binding account predicts it "
                "responds like the fully labelled one. The sign of the "
                "prespecified contrast separates them, and the length gaps favour "
                "the length account a priori."
            ),
        },
        "moment_order_not_involved": (
            "Every condition carries the full 24-bin histogram. This is a control "
            "on serialization at fixed information and says nothing about how much "
            "of the field is retained."
        ),
    }
    report["pass"] = bool(
        all(anchors_identical.values())
        and max_mass_gap == 0.0
        and report["length_axis_crosses_binding_axis"]["crossed"]
    )
    OUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
