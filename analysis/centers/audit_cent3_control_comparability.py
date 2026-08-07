"""CENT-4C: compare CENT-3 controls to historical centers phenotypes (offline)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.centers.cent3_io import (  # noqa: E402
    BRANCH,
    REP,
    action_channels,
    histogram_from_field,
    latest_cent3_runs,
    load_catalog,
    pooled_and_block_counts,
)
from analysis.stage3.models import counts_to_probs  # noqa: E402
from circlemap.observation import serialize_histogram  # noqa: E402
from circlemap.stimuli import STIMULUS_CATALOG, build_stimulus_histogram  # noqa: E402
from circlemap.representations import (  # noqa: E402
    build_representation_prompt_from_histogram,
)

# Historical anchors from Stage B / manifold / sparse panels
HISTORICAL = [
    {
        "anchor_id": "transmutation_kappa_9_dense",
        "profile": "kappa_9",
        "source_family": "transmutation",
        "catalog_id": None,  # von Mises via protocol concentration
        "peer_count": 240,
        "kappa": 9.0,
        "note": "Stage B sign-reversing phenotype source (dense peer)",
    },
    {
        "anchor_id": "manifold_unimodal_k9_dense",
        "profile": "unimodal_k9",
        "source_family": "stimulus_manifold",
        "catalog_id": "unimodal_k9",
        "peer_count": 240,
        "kappa": 9.0,
        "note": "manifold unimodal dense",
    },
    {
        "anchor_id": "sparse_N8_unimodal_k6",
        "profile": "sparse_N8_unimodal_k6",
        "source_family": "stimulus_manifold",
        "catalog_id": "sparse_N8_unimodal_k6",
        "peer_count": 8,
        "kappa": 6.0,
        "note": "finite-peer unimodal already in training",
    },
    {
        "anchor_id": "sparse_N16_antipodal_k6",
        "profile": "sparse_N16_antipodal_k6",
        "source_family": "stimulus_manifold",
        "catalog_id": "sparse_N16_antipodal_k6",
        "peer_count": 16,
        "kappa": 6.0,
        "note": "finite-peer antipodal",
    },
]


def _load_training_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate_profile(
    rows: list[dict],
    *,
    profile: str,
    source_family: str | None = None,
) -> dict[float, np.ndarray]:
    by_off: dict[float, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for row in rows:
        if row["representation"] != REP:
            continue
        if row["profile"] != profile:
            continue
        if source_family is not None and row["source_family"] != source_family:
            continue
        off = round(float(row["offset_radians"]), 6)
        by_off[off][0] += int(row["count_retard"])
        by_off[off][1] += int(row["count_stay"])
        by_off[off][2] += int(row["count_advance"])
    return by_off


def _nearest(by_off: dict[float, np.ndarray], target: float) -> tuple[float, np.ndarray]:
    if not by_off:
        return float("nan"), np.zeros(3)
    off = min(by_off, key=lambda o: abs(o - target))
    return off, by_off[off]


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    pa = counts_to_probs(a[None, :], alpha=0.5)[0]
    pb = counts_to_probs(b[None, :], alpha=0.5)[0]
    return float(0.5 * np.sum(np.abs(pa - pb)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir

    catalog_path = branch / "cent3_stimulus_catalog_v0.json"
    load_catalog(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    fields = json.loads((branch / "cent3_frozen_fields_v0.json").read_text(encoding="utf-8"))[
        "fields"
    ]
    controls = [f for f in fields if f["bucket"] == "control"]
    pooled, _ = pooled_and_block_counts(latest_cent3_runs())
    train_rows = _load_training_rows(branch / "training_rows_centers_v0.csv")

    # Historical offset-0 / π summaries
    hist_summaries = []
    for anchor in HISTORICAL:
        by_off = _aggregate_profile(
            train_rows,
            profile=anchor["profile"],
            source_family=anchor["source_family"],
        )
        off0, c0 = _nearest(by_off, 0.0)
        off_pi, c_pi = _nearest(by_off, math.pi)
        # also try -pi
        off_npi, c_npi = _nearest(by_off, -math.pi)
        if abs(off_npi + math.pi) < abs(off_pi - math.pi):
            off_pi, c_pi = off_npi, c_npi
        # Local polarity pair around the mode (historical multi-offset design)
        off_neg, c_neg = _nearest(by_off, -0.13)
        off_pos, c_pos = _nearest(by_off, +0.13)
        ch0 = action_channels(counts_to_probs(c0[None, :], alpha=0.0)[0])
        chpi = action_channels(counts_to_probs(c_pi[None, :], alpha=0.0)[0])
        ch_neg = action_channels(counts_to_probs(c_neg[None, :], alpha=0.0)[0])
        ch_pos = action_channels(counts_to_probs(c_pos[None, :], alpha=0.0)[0])
        # Serialized vector at offset 0 if catalog available
        ser = None
        if anchor["catalog_id"] and anchor["catalog_id"] in STIMULUS_CATALOG:
            hist = build_stimulus_histogram(anchor["catalog_id"], 0.0)
            ser = serialize_histogram(hist, decimals=6)
            peer = int(hist.peer_count)
            # bin origin: edges always [-pi, pi) canonical
            bin_origin = float(hist.edges[0])
        else:
            # dense kappa via fixed_field path
            from circlemap.response import fixed_field_histogram

            hist = fixed_field_histogram(0.0, float(anchor["kappa"]), synthetic_peer_count=240)
            ser = serialize_histogram(hist, decimals=6)
            peer = 240
            bin_origin = float(hist.edges[0])

        hist_summaries.append(
            {
                **anchor,
                "resolved_peer_count": peer,
                "bin_origin": bin_origin,
                "focal_offset_nearest_0": off0,
                "focal_offset_nearest_pi": off_pi,
                "n_valid_near_0": int(np.sum(c0)),
                "n_valid_near_pi": int(np.sum(c_pi)),
                "stay_near_0": ch0["p_stay"],
                "a0_near_0": ch0["a0"],
                "stay_near_pi": chpi["p_stay"],
                "a0_near_pi": chpi["a0"],
                "polarity_sign_flip_near_0_vs_pi": bool(ch0["a0"] * chpi["a0"] < 0),
                "offset_neg_of_mode": off_neg,
                "offset_pos_of_mode": off_pos,
                "a0_neg_of_mode": ch_neg["a0"],
                "a0_pos_of_mode": ch_pos["a0"],
                "stay_neg_of_mode": ch_neg["p_stay"],
                "stay_pos_of_mode": ch_pos["p_stay"],
                "local_polarity_sign_flip_across_mode": bool(
                    ch_neg["a0"] * ch_pos["a0"] < 0
                ),
                "serialized_24bin_sha16": __import__("hashlib")
                .sha256(ser.encode())
                .hexdigest()[:16],
            }
        )

    # CENT-3 controls
    ctrl_rows = []
    for field in controls:
        fid = field["field_id"]
        hist = histogram_from_field(field)
        ser = serialize_histogram(hist, decimals=6)
        prompt = build_representation_prompt_from_histogram(REP, hist)
        counts = pooled[fid]
        ch = action_channels(counts_to_probs(counts[None, :], alpha=0.0)[0])
        stim = field["stimulus"]
        ctrl_rows.append(
            {
                "field_id": fid,
                "role": field["role"],
                "peer_count": int(stim["peer_count"]),
                "kappa_nominal": float(stim.get("concentration", 0.0)),
                "derived_from_catalog": catalog.get(fid, {}).get("derived_from_catalog"),
                "bin_origin": float(hist.edges[0]),
                "focal_offset": 0.0,
                "n_bins": int(stim["n_bins"]),
                "field_normalization": "fractions_sum_to_1_integerized_to_peer",
                "n_nonzero_bins": int(np.sum(hist.fractions > 0)),
                "bin_sparsity": float(np.mean(hist.fractions > 0)),
                "n_valid": int(np.sum(counts)),
                "p_stay_obs": ch["p_stay"],
                "a0_obs": ch["a0"],
                "A_obs": ch["A"],
                "serialized_24bin_sha16": __import__("hashlib")
                .sha256(ser.encode())
                .hexdigest()[:16],
                "prompt_sha16": __import__("hashlib")
                .sha256(prompt.encode())
                .hexdigest()[:16],
            }
        )

    # Pairwise comparability judgments
    pos = next(r for r in ctrl_rows if r["field_id"] == "cent_ctrl_unimodal_positive_polar")
    rev = next(r for r in ctrl_rows if r["field_id"] == "cent_ctrl_unimodal_sign_reversed")
    sparse_u = next(r for r in ctrl_rows if r["field_id"] == "cent_ctrl_sparse_unimodal")
    dense_sign = next(
        h for h in hist_summaries if h["anchor_id"] == "transmutation_kappa_9_dense"
    )
    sparse_hist = next(
        h for h in hist_summaries if h["anchor_id"] == "sparse_N8_unimodal_k6"
    )

    same_condition_as_dense_sign_reversing = (
        pos["peer_count"] == dense_sign["resolved_peer_count"]
        and abs(pos["focal_offset"] - 0.0) < 1e-12
        # dense phenotype used many offsets; control uses single offset 0
    )

    judgment = {
        "safe_wording": (
            "At peer count 16 and the frozen focal-frame serialization, both "
            "polarity-reversed unimodal controls collapsed to abstention rather "
            "than expressing opposite directional responses."
        ),
        "not_claimed": (
            "Do not claim failure of the historical dense sign-reversing phenotype "
            "gate; peer/offset conditions differ."
        ),
        "comparability": {
            "cent3_unimodal_vs_dense_kappa9": {
                "same_peer": bool(pos["peer_count"] == dense_sign["resolved_peer_count"]),
                "same_focal_offset_design": False,
                "historical_used_multi_offset_fourier": True,
                "cent3_offset": 0.0,
                "historical_peer": dense_sign["resolved_peer_count"],
                "cent3_peer": pos["peer_count"],
                "classification": "new_peer_regime_phenotype",
            },
            "cent3_sparse_unimodal_vs_historical_sparse_N8": {
                "same_peer": bool(sparse_u["peer_count"] == sparse_hist["resolved_peer_count"]),
                "historical_stay_near_0": sparse_hist["stay_near_0"],
                "cent3_stay": sparse_u["p_stay_obs"],
                "classification": (
                    "consistent_with_finite_peer_abstention"
                    if sparse_u["p_stay_obs"] > 0.7
                    and sparse_hist["stay_near_0"] > 0.7
                    else "partially_comparable"
                ),
            },
            "polarity_pair_cent3": {
                "positive_a0": pos["a0_obs"],
                "reversed_a0": rev["a0_obs"],
                "both_all_stay": bool(pos["p_stay_obs"] == 1.0 and rev["p_stay_obs"] == 1.0),
                "sign_flip_observed": bool(pos["a0_obs"] * rev["a0_obs"] < 0),
            },
        },
        "dense_sign_reversing_reference": {
            "anchor": dense_sign["anchor_id"],
            "stay_near_0": dense_sign["stay_near_0"],
            "a0_near_0": dense_sign["a0_near_0"],
            "a0_neg_of_mode": dense_sign["a0_neg_of_mode"],
            "a0_pos_of_mode": dense_sign["a0_pos_of_mode"],
            "local_polarity_sign_flip_across_mode": dense_sign[
                "local_polarity_sign_flip_across_mode"
            ],
            "note": (
                "Historical sign-reversing claim is a multi-offset Fourier a1(κ) "
                "trajectory under peer=240; not a single offset=0 stay/action test."
            ),
        },
        "implication": (
            "centers response depends on finite-peer regime, not field shape alone; "
            "peer-stratified experts are scientifically motivated"
        ),
    }

    # TV between cent3 control empirical and nearest historical offset-0 cell
    match_rows = []
    mapping = [
        ("cent_ctrl_unimodal_positive_polar", "manifold_unimodal_k9_dense", 0.0),
        ("cent_ctrl_unimodal_positive_polar", "transmutation_kappa_9_dense", 0.0),
        ("cent_ctrl_sparse_unimodal", "sparse_N8_unimodal_k6", 0.0),
        ("cent_ctrl_sparse_antipodal", "sparse_N16_antipodal_k6", 0.0),
    ]
    hist_by = {h["anchor_id"]: h for h in hist_summaries}
    ctrl_by = {c["field_id"]: c for c in ctrl_rows}
    for fid, anchor_id, target_off in mapping:
        anchor = next(a for a in HISTORICAL if a["anchor_id"] == anchor_id)
        by_off = _aggregate_profile(
            train_rows,
            profile=anchor["profile"],
            source_family=anchor["source_family"],
        )
        off, counts_h = _nearest(by_off, target_off)
        counts_c = pooled[fid]
        match_rows.append(
            {
                "cent3_field": fid,
                "historical_anchor": anchor_id,
                "historical_offset_used": off,
                "tv_cent3_vs_historical": _tv(counts_c, counts_h),
                "cent3_stay": ctrl_by[fid]["p_stay_obs"],
                "historical_stay": hist_by[anchor_id]["stay_near_0"],
                "peer_cent3": ctrl_by[fid]["peer_count"],
                "peer_historical": hist_by[anchor_id]["resolved_peer_count"],
            }
        )

    out = {
        "status": "cent4c_control_comparability",
        "representation": REP,
        "historical_anchors": hist_summaries,
        "cent3_controls": ctrl_rows,
        "distribution_matches": match_rows,
        "judgment": judgment,
        "same_condition_as_dense_sign_reversing": bool(
            same_condition_as_dense_sign_reversing
        ),
    }
    (branch / "cent4c_control_comparability.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    with (branch / "cent4c_control_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ctrl_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ctrl_rows)
    print(json.dumps({"judgment": judgment, "matches": match_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
