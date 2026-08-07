"""Cost estimate for frozen 48×3×32 replay panel (no API calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circlemap.costing import estimate_cost, format_cost_estimate  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.representations import (  # noqa: E402
    build_representation_prompt_from_histogram,
)

OUT = ROOT / "analysis" / "matched_rep_collective"
FIELDS = OUT / "replay_fields_v0_1.json"
PROTOCOL = (
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_replay_v0_1.json"
)
TARGETS = (
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
)
CEILING = 12.0


def main() -> int:
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["fields"]
    edges = np.linspace(-np.pi, np.pi, 25)
    texts: list[str] = []
    for field in fields:
        hist = RelativePhaseHistogram(
            edges=edges,
            fractions=np.asarray(field["physical_histogram_24"], dtype=float),
            peer_count=16,
        )
        for target in TARGETS:
            prompt = build_representation_prompt_from_histogram(target, hist)
            texts.extend([prompt] * 32)

    assert len(texts) == 4608, len(texts)
    estimate = estimate_cost(
        texts,
        max_output_tokens_per_call=20,
        max_attempts_per_call=3,
        input_price_per_million=0.75,
        output_price_per_million=4.5,
    )
    under = estimate.estimated_base_cost_usd < CEILING
    report = {
        "n_calls": 4608,
        "n_unique_field_target_prompts": 144,
        "responses_per_condition": 32,
        "acquisition_blocks_preferred": "16x2",
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "estimate": {
            "base_usd": estimate.estimated_base_cost_usd,
            "retry_ceiling_usd": estimate.maximum_retry_cost_usd,
            "protocol_ceiling_usd": CEILING,
            "under_ceiling": under,
            "estimated_input_tokens": estimate.estimated_input_tokens,
        },
        "print": format_cost_estimate(estimate).replace(
            "Stage B", "Matched 3x3 replay"
        ),
    }
    out = OUT / "replay_estimate_only_v0_1.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Mark protocol paid_gates estimate flag for documentation (auth still false)
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    proto["paid_gates"]["estimate_only_under_ceiling"] = under
    PROTOCOL.write_text(json.dumps(proto, indent=2) + "\n", encoding="utf-8")

    print(report["print"])
    print(
        f"base=${estimate.estimated_base_cost_usd:.4f}  "
        f"x3=${estimate.maximum_retry_cost_usd:.4f}  ceiling=${CEILING:.2f}"
    )
    print("UNDER_CEILING" if under else "OVER_CEILING")
    print(f"Wrote {out}")
    return 0 if under else 1


if __name__ == "__main__":
    raise SystemExit(main())
