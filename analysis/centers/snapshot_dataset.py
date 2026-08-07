"""CENT-0: hash-locked centers_24_standard training snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.dataset import (  # noqa: E402
    build_unified_dataset,
    post_pilot_source_runs,
    write_dataset_csv,
)
from circlemap.field_features import COMMON_FEATURE_NAMES  # noqa: E402

REP = "centers_24_standard"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "centers_branch",
    )
    args = parser.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Building post-pilot unified dataset...")
    all_rows, packed = build_unified_dataset(
        ROOT, source_runs=post_pilot_source_runs(ROOT)
    )
    rows = [r for r in all_rows if r.representation == REP]
    if not rows:
        raise SystemExit(f"no rows for {REP}")
    block = packed[REP]
    x = block["features"]
    counts = block["counts"]
    unresolved = block["unresolved_near_zero"]
    if len(rows) != x.shape[0]:
        raise RuntimeError("row/feature misalignment")

    write_dataset_csv(rows, out / "training_rows_centers_v0.csv")
    np.savez_compressed(
        out / "training_arrays_centers_v0.npz",
        features=x,
        counts=counts,
        unresolved_near_zero=unresolved,
        estimated_eps=block["estimated_eps"],
        common_feature_names=np.asarray(COMMON_FEATURE_NAMES),
        native_feature_names=np.asarray([f"bin_{i:02d}" for i in range(24)]),
    )

    # Phenotype summary
    totals = np.sum(counts, axis=1, keepdims=True)
    totals = np.maximum(totals, 1.0)
    p = counts / totals
    activity = p[:, 0] + p[:, 2]
    g = p[:, 2] - p[:, 0]
    a0 = p[:, 1]
    phenotype_rows = []
    for i, row in enumerate(rows):
        phenotype_rows.append(
            {
                "profile": row.profile,
                "profile_family": row.profile_family,
                "source_family": row.source_family,
                "acquisition_block": row.acquisition_block,
                "epsilon_level": row.epsilon_level,
                "sparse_realization": row.sparse_realization,
                "offset_group": row.offset_group,
                "offset_index": row.offset_index,
                "p_retard": float(p[i, 0]),
                "p_stay": float(p[i, 1]),
                "p_advance": float(p[i, 2]),
                "activity": float(activity[i]),
                "g": float(g[i]),
                "a0": float(a0[i]),
                "abs_z1": float(x[i, COMMON_FEATURE_NAMES.index("abs_z1")]),
                "antipodal_balance": float(
                    x[i, COMMON_FEATURE_NAMES.index("antipodal_balance")]
                ),
                "n_trials": int(np.sum(counts[i])),
            }
        )
    with (out / "source_profile_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phenotype_rows[0].keys()))
        writer.writeheader()
        writer.writerows(phenotype_rows)

    sources = Counter(r.source_family for r in rows)
    blocks = Counter(r.acquisition_block for r in rows)
    rollup = {
        "representation": REP,
        "n_rows": len(rows),
        "n_features": int(x.shape[1]),
        "n_common": len(COMMON_FEATURE_NAMES),
        "n_native": 24,
        "sources": dict(sources),
        "acquisition_blocks": dict(blocks),
        "mean_activity": float(np.mean(activity)),
        "mean_p_stay": float(np.mean(p[:, 1])),
        "mean_abs_g": float(np.mean(np.abs(g))),
        "exact_balance_mean_stay": float(
            np.mean(
                [
                    phenotype_rows[i]["p_stay"]
                    for i, r in enumerate(rows)
                    if r.epsilon_level == "+0.000"
                    or "antipodal_equal" in r.profile
                    or "_8_8_" in r.profile
                ]
                or [float("nan")]
            )
        ),
        "note": (
            "moments/intervals responses excluded; no cyclic-shift augmentation; "
            "collective_replay not included (moments-only)"
        ),
    }
    (out / "phenotype_rollup_v0.json").write_text(
        json.dumps(rollup, indent=2), encoding="utf-8"
    )

    file_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "dataset_manifest_v0.json"
    }
    manifest = {
        "branch": "centers",
        "version": "centers-dataset-v0",
        "representation": REP,
        "status": "hash_locked",
        "n_rows": len(rows),
        "sources": dict(sources),
        "file_sha256": file_hashes,
        "aggregate_sha256": _sha256_json(file_hashes),
        "exclusions": {
            "moments_responses": True,
            "intervals_responses": True,
            "collective_replay": True,
            "cyclic_shift_augmentation": True,
        },
        "scientific_question": (
            "Can a center-bin encoding of the same physical field yield a "
            "predictable microscopic response operator, and does it select "
            "different macroscopic dynamics from moments?"
        ),
    }
    (out / "dataset_manifest_v0.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "n_rows": len(rows),
        "sources": dict(sources),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "exact_balance_mean_stay": rollup["exact_balance_mean_stay"],
        "mean_p_stay": rollup["mean_p_stay"],
        "out": str(out.relative_to(ROOT).as_posix()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
