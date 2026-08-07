"""Offline P3 full-operator gate calibration + P4 phenotype atlas."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFOLD = ROOT / "analysis" / "complex_kernel_stimulus_manifold"
OUT = ROOT / "analysis" / "stage_b_offline_p3_p4"


def _prob_matrix(condition: dict) -> np.ndarray:
    probs = condition["probabilities_by_offset"]
    return np.column_stack(
        [probs["retard"], probs["stay"], probs["advance"]]
    )


def mean_tv(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.mean(np.sum(np.abs(first - second), axis=1)))


def main() -> None:
    report = json.loads(
        (MANIFOLD / "complex_kernel_analysis.json").read_text(encoding="utf-8")
    )
    blocks = list(report["blocks"].keys())
    if len(blocks) != 2:
        raise SystemExit("expected exactly two blocks")
    representations = list(report["blocks"][blocks[0]]["representations"].keys())
    profiles = report["stimulus_profiles"]

    within_rows = []
    between_rows = []
    ratio_rows = []
    for profile in profiles:
        within_tvs = []
        for representation in representations:
            left = report["blocks"][blocks[0]]["representations"][representation][
                "conditions"
            ][profile]
            right = report["blocks"][blocks[1]]["representations"][representation][
                "conditions"
            ][profile]
            tv = mean_tv(_prob_matrix(left), _prob_matrix(right))
            within_tvs.append(tv)
            within_rows.append(
                {
                    "profile": profile,
                    "representation": representation,
                    "mean_tv_across_blocks": tv,
                    "comparison": "within_representation_across_blocks",
                }
            )
        within_mean = float(np.mean(within_tvs))
        between_tvs = []
        for i, left_rep in enumerate(representations):
            for right_rep in representations[i + 1 :]:
                tvs = []
                for block in blocks:
                    left = report["blocks"][block]["representations"][left_rep][
                        "conditions"
                    ][profile]
                    right = report["blocks"][block]["representations"][right_rep][
                        "conditions"
                    ][profile]
                    tvs.append(mean_tv(_prob_matrix(left), _prob_matrix(right)))
                tv = float(np.mean(tvs))
                between_tvs.append(tv)
                between_rows.append(
                    {
                        "profile": profile,
                        "representation_a": left_rep,
                        "representation_b": right_rep,
                        "mean_tv_across_blocks": tv,
                        "comparison": "between_representation",
                    }
                )
        between_min = float(np.min(between_tvs)) if between_tvs else float("nan")
        between_mean = float(np.mean(between_tvs)) if between_tvs else float("nan")
        ratio = between_min / within_mean if within_mean > 1e-12 else float("nan")
        ratio_rows.append(
            {
                "profile": profile,
                "within_tv_mean": within_mean,
                "between_tv_min": between_min,
                "between_tv_mean": between_mean,
                "between_min_over_within_mean": ratio,
                "passes_ratio_gt_2": int(ratio > 2.0) if ratio == ratio else 0,
                "classification": report["field_separations"][profile][
                    "classification"
                ],
                "minimum_pair_c1_chordal": report["field_separations"][profile][
                    "minimum_pair_c1_chordal"
                ],
            }
        )

    # Phenotype atlas from endpoints CSV (across-block means).
    endpoints = list(
        csv.DictReader((MANIFOLD / "complex_endpoints.csv").open(encoding="utf-8"))
    )
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in endpoints:
        short = row["representation"].split("_")[0]
        grouped[(short, row["profile"])].append(row)
    atlas_rows = []
    for (rep, profile), rows in sorted(grouped.items()):
        def avg(key: str) -> float:
            return float(np.mean([float(item[key]) for item in rows]))

        atlas_rows.append(
            {
                "representation": rep,
                "profile": profile,
                "a0": avg("a0"),
                "a1": avg("a1"),
                "b1": avg("b1"),
                "R1": avg("R1"),
                "R2": avg("R2"),
                "activity": avg("activity"),
                "mean_entropy": avg("mean_entropy"),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(OUT / "full_operator_within_tv.csv", within_rows)
    write_csv(OUT / "full_operator_between_tv.csv", between_rows)
    write_csv(OUT / "full_operator_tv_calibration.csv", ratio_rows)
    write_csv(OUT / "phenotype_atlas.csv", atlas_rows)

    summary = {
        "source": str(MANIFOLD),
        "polar_gate_historical": {
            "metric": "minimum_pairwise_C1_chordal",
            "threshold": 0.25,
            "status": "frozen_preregistered_do_not_rewrite_v0_1",
        },
        "full_operator_gate_proposed": {
            "primary_rule": "between_min_TV / within_mean_TV > 2",
            "companion": "bootstrap Δ distribution excludes 0 (when available)",
            "absolute_tv_0_20": {
                "status": "descriptive_only_not_frozen",
                "note": (
                    "Absolute 0.20 is scale-dependent; prefer between/within "
                    "ratio calibrated on this manifold."
                ),
            },
            "calibration_rows": ratio_rows,
        },
        "atlas_note": (
            "a0 and activity are field-dependent: a0=a0(R, rho). Unimodal "
            "interval+/center- signs do not globalize."
        ),
    }
    (OUT / "p3_p4_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"wrote": str(OUT), "profiles": len(ratio_rows)}, indent=2))


if __name__ == "__main__":
    main()
