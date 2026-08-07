"""Offline audit of the serialization-length control.

Proves, without any model call, that

1. the two standard arms are byte-identical to the primary ``moments_m1_m3``
   and ``centers_24_standard`` prompts;
2. within each content class the compact and standard arms decode to the same
   numbers (masses bit-exact; centres to print precision);
3. the length axis crosses the content axis on the frozen 48-field panel;
4. moment order is not a length knob: order-12 binned moments reconstruct the
   24-bin histogram exactly, so a higher-order moments arm would be the
   centers arm in another notation rather than a longer summary.

Writes ``slc_information_audit.json`` next to this script.
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
from circlemap.serialization_length_controls import (  # noqa: E402
    N_BINS,
    SLC_CONTENT,
    SLC_FORM,
    SLC_VARIANTS,
    build_slc_prompt,
    decode_centers,
    decode_moments,
)

FIELDS_PATH = ROOT / "analysis/matched_rep_collective/replay_fields_v0_1.json"
OUT_PATH = Path(__file__).with_name("slc_information_audit.json")


def _histogram(field: dict) -> RelativePhaseHistogram:
    return RelativePhaseHistogram(
        edges=np.linspace(-np.pi, np.pi, N_BINS + 1),
        fractions=np.asarray(field["physical_histogram_24"], dtype=float),
        peer_count=int(field.get("peer_count", 16)),
    )


def _binned_moments(masses: np.ndarray, order: int) -> np.ndarray:
    width = 2.0 * np.pi / masses.size
    centres = -np.pi + width * (np.arange(masses.size) + 0.5)
    weight = masses / masses.sum()
    return np.asarray(
        [np.sum(weight * np.exp(1j * m * centres)) for m in range(1, order + 1)]
    )


def _reconstruct_from_moments(moments: np.ndarray, n_bins: int) -> np.ndarray:
    """Invert the binned-moment DFT from harmonics 1..n_bins/2."""
    width = 2.0 * np.pi / n_bins
    centres = -np.pi + width * (np.arange(n_bins) + 0.5)
    out = np.full(n_bins, 1.0 / n_bins, dtype=np.float64)
    half = n_bins // 2
    for m in range(1, half + 1):
        term = moments[m - 1] * np.exp(-1j * m * centres)
        # harmonics m and n_bins-m are conjugate duplicates except at m=half
        out += (1.0 if m == half else 2.0) * np.real(term) / n_bins
    return out


def main() -> int:
    fields = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))["fields"]

    lengths: dict[str, list[int]] = {v: [] for v in SLC_VARIANTS}
    tokens: dict[str, list[int]] = {v: [] for v in SLC_VARIANTS}
    standard_identical = {"moments": True, "centers": True}
    max_moment_gap = 0.0
    max_mass_gap = 0.0
    max_centre_gap = 0.0
    max_reconstruction_error = 0.0
    min_moment_magnitude = np.inf

    for field in fields:
        hist = _histogram(field)
        prompts = {v: build_slc_prompt(v, hist) for v in SLC_VARIANTS}
        for variant, prompt in prompts.items():
            lengths[variant].append(len(prompt))
            tokens[variant].append(int(estimate_tokens_from_characters(prompt)))

        standard_identical["moments"] &= prompts[
            "moments_standard"
        ] == build_representation_prompt_from_histogram("moments_m1_m3", hist)
        standard_identical["centers"] &= prompts[
            "centers_standard"
        ] == build_representation_prompt_from_histogram("centers_24_standard", hist)

        a = decode_moments("moments_standard", prompts["moments_standard"])
        b = decode_moments("moments_compact", prompts["moments_compact"])
        max_moment_gap = max(max_moment_gap, float(np.max(np.abs(a - b))))

        c_std, m_std = decode_centers("centers_standard", prompts["centers_standard"])
        c_cmp, m_cmp = decode_centers("centers_compact", prompts["centers_compact"])
        max_mass_gap = max(max_mass_gap, float(np.max(np.abs(m_std - m_cmp))))
        max_centre_gap = max(max_centre_gap, float(np.max(np.abs(c_std - c_cmp))))

        masses = np.asarray(field["physical_histogram_24"], dtype=float)
        masses = masses / masses.sum()
        full = _binned_moments(masses, N_BINS // 2)
        max_reconstruction_error = max(
            max_reconstruction_error,
            float(np.max(np.abs(_reconstruct_from_moments(full, N_BINS) - masses))),
        )
        min_moment_magnitude = min(min_moment_magnitude, float(np.min(np.abs(full))))

    def _stats(values: dict[str, list[int]]) -> dict[str, dict[str, float]]:
        return {
            v: {
                "min": int(np.min(values[v])),
                "median": float(np.median(values[v])),
                "max": int(np.max(values[v])),
            }
            for v in SLC_VARIANTS
        }

    med = {v: float(np.median(lengths[v])) for v in SLC_VARIANTS}
    med_tok = {v: float(np.median(tokens[v])) for v in SLC_VARIANTS}

    report = {
        "n_fields": len(fields),
        "fields_file": str(FIELDS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "variants": {
            v: {"content": SLC_CONTENT[v], "form": SLC_FORM[v]} for v in SLC_VARIANTS
        },
        "prompt_chars": _stats(lengths),
        "prompt_tokens": _stats(tokens),
        "standard_arms_byte_identical_to_primary": standard_identical,
        "information_equivalence": {
            "max_abs_moment_component_gap": max_moment_gap,
            "max_abs_bin_mass_gap": max_mass_gap,
            "max_abs_bin_centre_gap_print_precision": max_centre_gap,
        },
        "length_axis_crosses_content_axis": {
            "chars_centers_compact_to_centers_standard": med["centers_standard"]
            - med["centers_compact"],
            "chars_centers_compact_to_moments_standard": med["centers_compact"]
            - med["moments_standard"],
            "tokens_centers_compact_to_centers_standard": med_tok["centers_standard"]
            - med_tok["centers_compact"],
            "tokens_centers_compact_to_moments_standard": med_tok["centers_compact"]
            - med_tok["moments_standard"],
            "crossed": (
                med["centers_standard"] - med["centers_compact"]
                > med["centers_compact"] - med["moments_standard"]
            ),
        },
        "moment_order_is_not_a_length_knob": {
            "order_used_in_all_arms": 3,
            "order_that_reconstructs_the_histogram": N_BINS // 2,
            "max_abs_reconstruction_error": max_reconstruction_error,
            "min_abs_moment_magnitude_orders_1_to_12": min_moment_magnitude,
            "note": (
                "Binned moments are a discrete Fourier transform of the 24 bin "
                "masses, so harmonics 1..12 invert exactly. A 12th-order moments "
                "arm is the centers arm in another notation, not a longer "
                "summary of it."
            ),
        },
    }
    report["pass"] = bool(
        all(standard_identical.values())
        and max_moment_gap == 0.0
        and max_mass_gap == 0.0
        and max_reconstruction_error < 1e-12
        and report["length_axis_crosses_content_axis"]["crossed"]
    )
    OUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
