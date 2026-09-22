"""Estimate-only for same-information observation-map control (no API)."""

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
from circlemap.observation_map_controls import (  # noqa: E402
    CONTROL_VARIANTS,
    LENGTH_MATCH_TARGET_CHARS,
    build_control_prompt,
)

OUT = ROOT / "analysis" / "matched_rep_collective"
FIELDS = OUT / "replay_fields_v0_1.json"
PROTOCOL = (
    ROOT
    / "experiments/stage_c/protocol_matched_rep_collective_omap_control_v0_1.json"
)


def main() -> int:
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["fields"]
    edges = np.linspace(-np.pi, np.pi, 25)
    texts: list[str] = []
    lengths = {v: [] for v in CONTROL_VARIANTS}
    for field in fields:
        hist = RelativePhaseHistogram(
            edges=edges,
            fractions=np.asarray(field["physical_histogram_24"], dtype=float),
            peer_count=16,
        )
        for variant in CONTROL_VARIANTS:
            prompt = build_control_prompt(variant, hist)
            lengths[variant].append(len(prompt))
            texts.extend([prompt] * 16)
    assert len(texts) == 2304
    estimate = estimate_cost(
        texts,
        max_output_tokens_per_call=20,
        max_attempts_per_call=3,
        input_price_per_million=0.75,
        output_price_per_million=4.5,
    )
    report = {
        "n_calls": 2304,
        "length_match_target_chars": LENGTH_MATCH_TARGET_CHARS,
        "mean_prompt_chars": {k: float(np.mean(v)) for k, v in lengths.items()},
        "estimate": {
            "base_usd": estimate.estimated_base_cost_usd,
            "retry_ceiling_usd": estimate.maximum_retry_cost_usd,
            "protocol_ceiling_usd": 6.0,
            "under_ceiling": estimate.estimated_base_cost_usd < 6.0,
        },
        "print": format_cost_estimate(estimate).replace(
            "Stage B", "Observation-map control"
        ),
    }
    out = OUT / "omap_estimate_only_v0_1.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["print"])
    print(
        f"base=${estimate.estimated_base_cost_usd:.4f}  "
        f"x3=${estimate.maximum_retry_cost_usd:.4f}  ceiling=$6"
    )
    print("lengths", report["mean_prompt_chars"])
    print("UNDER_CEILING" if report["estimate"]["under_ceiling"] else "OVER")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
