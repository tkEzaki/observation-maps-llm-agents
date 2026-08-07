"""Analyze Step D nested sparse-peer imbalance × realization panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.analyze_complex_kernel import _write_csv
from analysis.analyze_transmutation import load_block
from analysis.complex_kernel import (
    mean_total_variation,
    polar,
    summarize_condition,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_RE = re.compile(
    r"sparse_peer(?P<peer>\d+)_antipodal_(?P<a>\d+)_(?P<b>\d+)_r(?P<r>\d+)_k(?P<k>\d+)"
)


def _parse_profile(profile: str) -> dict:
    match = PROFILE_RE.fullmatch(profile)
    if match is None:
        raise ValueError(f"unrecognized sparse profile id: {profile}")
    return {
        "peer_count": int(match.group("peer")),
        "count_a": int(match.group("a")),
        "count_b": int(match.group("b")),
        "realization": int(match.group("r")),
        "imbalance": f"{match.group('a')}_{match.group('b')}",
    }


def _prob_matrix(summary: dict) -> np.ndarray:
    probs = summary["probabilities_by_offset"]
    return np.column_stack(
        [probs["retard"], probs["stay"], probs["advance"]]
    )


def analyze_sparse_peer(
    run_dirs: list[Path],
    output_dir: Path,
) -> dict:
    if len(run_dirs) != 2:
        raise ValueError("exactly two run directories required")
    loaded = [load_block(path) for path in run_dirs]
    protocol = loaded[0][0]["protocol"]
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    profiles = list(protocol["stimulus_profiles"])
    representations = list(protocol["representations"])
    primary_order = int(protocol["primary_fourier_order"])

    endpoint_rows = []
    block_payload: dict = {}
    for config, counts in loaded:
        block_id = config["protocol"]["seed_block_id"]
        block_payload[block_id] = {"representations": {}}
        for representation in representations:
            conditions = {}
            for profile in profiles:
                meta = _parse_profile(profile)
                summary = summarize_condition(
                    counts[representation][profile],
                    offsets,
                    primary_order=primary_order,
                )
                harmonics = summary["harmonics"]
                polar1 = harmonics["polar"]["1"]
                conditions[profile] = {"meta": meta, **summary}
                endpoint_rows.append(
                    {
                        "block": block_id,
                        "representation": representation,
                        "profile": profile,
                        "imbalance": meta["imbalance"],
                        "realization": meta["realization"],
                        "count_a": meta["count_a"],
                        "count_b": meta["count_b"],
                        "a0": harmonics["a0"],
                        "a1": harmonics["a_sin"][1],
                        "b1": harmonics["b_cos"][1],
                        "R1": polar1["R"],
                        "phi1_degrees": math.degrees(polar1["phi"]),
                        "R2": harmonics["polar"]["2"]["R"],
                        "activity": summary["activity"]["mean_activity"],
                        "mean_entropy": summary["mean_entropy"],
                    }
                )
            block_payload[block_id]["representations"][representation] = {
                "conditions": conditions
            }

    # Across-block mean per (representation, imbalance, realization)
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in endpoint_rows:
        key = (row["representation"], row["imbalance"], int(row["realization"]))
        grouped[key].append(row)

    realization_rows = []
    for (representation, imbalance, realization), rows in sorted(grouped.items()):
        realization_rows.append(
            {
                "representation": representation,
                "imbalance": imbalance,
                "realization": realization,
                "a1": float(np.mean([row["a1"] for row in rows])),
                "b1": float(np.mean([row["b1"] for row in rows])),
                "R1": float(np.mean([row["R1"] for row in rows])),
                "R2": float(np.mean([row["R2"] for row in rows])),
                "a0": float(np.mean([row["a0"] for row in rows])),
                "activity": float(np.mean([row["activity"] for row in rows])),
            }
        )

    # Variance decomposition by imbalance.
    variance_rows = []
    distance_rows = []
    for imbalance in sorted({row["imbalance"] for row in realization_rows}):
        by_rep = {
            representation: [
                row
                for row in realization_rows
                if row["imbalance"] == imbalance
                and row["representation"] == representation
            ]
            for representation in representations
        }
        within_rms = {}
        for representation, rows in by_rep.items():
            a1 = np.asarray([row["a1"] for row in rows], dtype=np.float64)
            activity = np.asarray(
                [row["activity"] for row in rows], dtype=np.float64
            )
            r1 = np.asarray([row["R1"] for row in rows], dtype=np.float64)
            within_rms[representation] = {
                "a1_rms": float(np.std(a1, ddof=1)) if a1.size > 1 else 0.0,
                "activity_rms": float(np.std(activity, ddof=1))
                if activity.size > 1
                else 0.0,
                "R1_rms": float(np.std(r1, ddof=1)) if r1.size > 1 else 0.0,
                "a1_mean": float(np.mean(a1)),
                "activity_mean": float(np.mean(activity)),
                "R1_mean": float(np.mean(r1)),
            }
            variance_rows.append(
                {
                    "imbalance": imbalance,
                    "representation": representation,
                    **within_rms[representation],
                }
            )

        # Between-encoding distances on realization-mean (a1,b1) and activity,
        # plus mean TV across realizations×blocks using first realization only
        # as a quick operator distance — better: average TV over realizations.
        for i, left in enumerate(representations):
            for right in representations[i + 1 :]:
                left_pts = np.column_stack(
                    [
                        [row["a1"] for row in by_rep[left]],
                        [row["b1"] for row in by_rep[left]],
                    ]
                )
                right_pts = np.column_stack(
                    [
                        [row["a1"] for row in by_rep[right]],
                        [row["b1"] for row in by_rep[right]],
                    ]
                )
                # Mean C1 chordal between matched realization indices
                n = min(left_pts.shape[0], right_pts.shape[0])
                c1 = [
                    float(np.linalg.norm(left_pts[k] - right_pts[k]))
                    for k in range(n)
                ]
                within = 0.5 * (
                    within_rms[left]["a1_rms"] + within_rms[right]["a1_rms"]
                )
                # Also activity separation
                act_sep = abs(
                    within_rms[left]["activity_mean"]
                    - within_rms[right]["activity_mean"]
                )
                act_within = 0.5 * (
                    within_rms[left]["activity_rms"]
                    + within_rms[right]["activity_rms"]
                )
                # Mean TV over blocks×realizations
                tvs = []
                for block_id in block_payload:
                    left_conds = block_payload[block_id]["representations"][left][
                        "conditions"
                    ]
                    right_conds = block_payload[block_id]["representations"][right][
                        "conditions"
                    ]
                    for profile, left_summary in left_conds.items():
                        if left_summary["meta"]["imbalance"] != imbalance:
                            continue
                        right_summary = right_conds[profile]
                        tvs.append(
                            mean_total_variation(
                                _prob_matrix(left_summary),
                                _prob_matrix(right_summary),
                            )
                        )
                mean_tv = float(np.mean(tvs)) if tvs else float("nan")
                distance_rows.append(
                    {
                        "imbalance": imbalance,
                        "representation_a": left,
                        "representation_b": right,
                        "mean_c1_chordal_across_realizations": float(np.mean(c1)),
                        "c1_within_realization_rms_mean": within,
                        "c1_between_over_within": (
                            float(np.mean(c1)) / within if within > 1e-12 else float("nan")
                        ),
                        "activity_abs_mean_diff": act_sep,
                        "activity_within_rms_mean": act_within,
                        "activity_between_over_within": (
                            act_sep / act_within if act_within > 1e-12 else float("nan")
                        ),
                        "mean_tv": mean_tv,
                        "passes_c1_ratio_gt_2": int(
                            (float(np.mean(c1)) / within) > 2.0
                            if within > 1e-12
                            else 0
                        ),
                        "passes_activity_ratio_gt_2": int(
                            (act_sep / act_within) > 2.0
                            if act_within > 1e-12
                            else 0
                        ),
                    }
                )

    report = {
        "family": "sparse_peer_panel",
        "protocol_name": protocol["protocol_name"],
        "n_profiles": len(profiles),
        "imbalances": sorted({row["imbalance"] for row in realization_rows}),
        "n_realizations": len(
            {row["realization"] for row in realization_rows}
        ),
        "gate_note": (
            "Stage-3 sparse training is supported when between/within ratios "
            "exceed ~2 for C1 or activity on the relevant imbalance."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sparse_peer_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "complex_endpoints.csv", endpoint_rows)
    _write_csv(output_dir / "realization_means.csv", realization_rows)
    _write_csv(output_dir / "within_realization_variance.csv", variance_rows)
    _write_csv(output_dir / "between_within_distances.csv", distance_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "complex_kernel_sparse_peer",
    )
    args = parser.parse_args()
    report = analyze_sparse_peer(args.run_dir, args.out_dir)
    print(json.dumps({"wrote": str(args.out_dir), **{k: report[k] for k in ("n_profiles", "imbalances")}}, indent=2))


if __name__ == "__main__":
    main()
