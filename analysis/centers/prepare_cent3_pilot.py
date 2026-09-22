"""CENT-3 prep: circular holdout compare, 36-field selection, protocol freeze.

Does NOT launch paid API calls. Writes review table + prospective lock.
"""

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

from analysis.centers.models import HybridNativeKernel  # noqa: E402
from analysis.stage3.applicability_domain import ApplicabilityDomain  # noqa: E402
from analysis.stage3.coverage_utils import N_COMMON, feature_builder  # noqa: E402
from analysis.stage3.freeze_pilot_protocol import (  # noqa: E402
    _integerize,
    construct_sibling_field,
)
from analysis.stage3.models import (  # noqa: E402
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    SoftmaxStumpBoost,
)
from analysis.stage3.risk_calibrator import RiskCalibrator  # noqa: E402
from analysis.stage3.validation import action_metrics  # noqa: E402
from circlemap.field_features import extract_field_descriptors  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.representations import build_representation_prompt_from_histogram  # noqa: E402
from circlemap.stimuli import (  # noqa: E402
    STIMULUS_CATALOG,
    StimulusSpec,
    build_stimulus_histogram,
)

REP = "centers_24_standard"
BRANCH = ROOT / "analysis" / "centers_branch"
BANK_PATH = ROOT / "analysis" / "stage3a_artifacts" / "candidate_field_bank.npz"


# Peer-matched anchors only (N∈{9,17} ⇒ peer∈{8,16}). Dense peer=240
# unimodal_k9 is excluded from the paid pilot AD peer gate.
CONTROLS = (
    {
        "field_id": "cent_ctrl_unimodal_positive_polar",
        "role": "centers_unimodal_positive_polar",
        "catalog_id": "cent_tmp_peer16_unimodal_k9",
        "notes": "Peer-16 unimodal positive-polar anchor (κ=9)",
        "register": {
            "kind": "unimodal",
            "concentration": 9.0,
            "peer_count": 16,
        },
    },
    {
        "field_id": "cent_ctrl_unimodal_sign_reversed",
        "role": "centers_sign_reversed_anchor",
        "catalog_id": "cent_tmp_peer16_unimodal_k9",
        "shift_bins": 12,
        "notes": "Same peer-16 unimodal shifted by π (sign-reversed first harmonic)",
    },
    {
        "field_id": "cent_ctrl_exact_antipodal",
        "role": "exact_antipodal",
        "catalog_id": "sparse_peer16_antipodal_8_8_r00_k6",
        "notes": "Exact antipodal balance",
    },
    {
        "field_id": "cent_ctrl_small_imbalance",
        "role": "small_imbalance",
        "catalog_id": "sparse_peer16_antipodal_9_7_r00_k6",
        "notes": "Small count imbalance",
    },
    {
        "field_id": "cent_ctrl_sparse_unimodal",
        "role": "sparse_unimodal",
        "catalog_id": "sparse_N8_unimodal_k6",
        "notes": "Sparse unimodal (peer 8)",
    },
    {
        "field_id": "cent_ctrl_sparse_antipodal",
        "role": "sparse_antipodal",
        "catalog_id": "sparse_N16_antipodal_k6",
        "notes": "Sparse antipodal equal-weight",
    },
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: object) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _quantize_key(abs_z1: float, abs_z2: float, peer: int, eps: float) -> tuple:
    return (
        int(np.clip(abs_z1, 0, 1) * 10),
        int(np.clip(abs_z2, 0, 1) * 10),
        int(peer),
        int(np.clip(abs(eps), 0, 0.5) * 20),
    )


def _dedupe_key(row: dict) -> tuple:
    return (
        str(row["init_kind"]),
        int(row["peer_count"]),
        int(np.clip(float(row["abs_z1"]), 0, 1) * 20),
        int(np.clip(float(row["abs_z2"]), 0, 1) * 20),
        str(row["source"]),
        str(row["dynamics"]),
    )


def compare_circular_holdout(branch: Path) -> dict:
    """Acquisition-block holdout: Hellinger vs circular-EMD hybrid kernels."""
    arrays = np.load(branch / "training_arrays_centers_v0.npz", allow_pickle=True)
    x = arrays["features"]
    counts = arrays["counts"]
    with (branch / "training_rows_centers_v0.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    blocks = sorted({r["acquisition_block"] for r in rows})
    test_block = blocks[-1]
    test_mask = np.asarray(
        [r["acquisition_block"] == test_block for r in rows], dtype=bool
    )
    x_tr, c_tr = x[~test_mask], counts[~test_mask]
    x_te, c_te = x[test_mask], counts[test_mask]
    out = {}
    for metric in ("hellinger", "circular_emd"):
        model = HybridNativeKernel.fit(
            x_tr, c_tr, native_metric=metric, k_neighbors=24
        )
        pred = model.predict_proba(x_te)
        out[metric] = action_metrics(pred, c_te)
    winner = min(out.items(), key=lambda kv: kv[1]["log_loss"])[0]
    payload = {
        "holdout_block": test_block,
        "metrics": out,
        "preferred_native_metric_by_holdout_log_loss": winner,
        "note": (
            "Full LOGO CV skipped circular for runtime; holdout compare is the "
            "native-distance gate. Production model remains acquisition-block "
            "CV winner (activity_direction_hurdle), not hybrid kernel."
        ),
    }
    (branch / "cent1_native_distance_holdout.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _histogram_from_fractions(
    fractions: tuple[float, ...] | list[float], peer_count: int
) -> RelativePhaseHistogram:
    edges = np.linspace(-np.pi, np.pi, 25, dtype=np.float64)
    return RelativePhaseHistogram(
        edges=edges,
        fractions=np.asarray(fractions, dtype=np.float64),
        peer_count=int(peer_count),
    )


def _ensure_tmp_catalog(control: dict) -> None:
    reg = control.get("register")
    if not reg:
        return
    catalog_id = control["catalog_id"]
    if catalog_id in STIMULUS_CATALOG:
        return
    STIMULUS_CATALOG[catalog_id] = StimulusSpec(
        kind=reg["kind"],
        concentration=float(reg["concentration"]),
        peer_count=int(reg["peer_count"]),
    )


def _control_fractions(control: dict) -> tuple[tuple[float, ...], int]:
    _ensure_tmp_catalog(control)
    hist = build_stimulus_histogram(control["catalog_id"], 0.0)
    frac = np.asarray(hist.fractions, dtype=np.float64)
    shift_bins = int(control.get("shift_bins", 0))
    if shift_bins:
        frac = np.roll(frac, shift_bins)
    peer = int(hist.peer_count)
    if peer not in {8, 16}:
        raise ValueError(f"control peer_count {peer} not in {{8,16}}")
    return _integerize(frac, peer), peer


def _score_candidate(
    fractions: tuple[float, ...],
    peer_count: int,
    *,
    ad: ApplicabilityDomain,
    calibrator: RiskCalibrator,
    models: dict,
) -> dict:
    hist = _histogram_from_fractions(fractions, peer_count)
    x, unresolved, desc = feature_builder(REP, hist)
    x1 = x[None, :]
    sc = ad.score(x1, unresolved_near_zero=np.asarray([unresolved], dtype=bool))
    risk = float(
        calibrator.predict_arrays(
            d_shape=sc["d_shape"],
            d_native=sc["d_native"],
            local_density=sc["local_density"],
            u_ensemble=np.zeros(1),
            d_orientation=sc["d_orientation"],
            policy_a=sc["policy_a"],
        )[0]
    )
    preds = [m.predict_proba(x1)[0] for m in models.values()]
    disagreement = 0.0
    pairs = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            disagreement += 0.5 * float(np.sum(np.abs(preds[i] - preds[j])))
            pairs += 1
    disagreement /= max(pairs, 1)
    prod = models["activity_direction_hurdle"].predict_proba(x1)[0]
    return {
        "predicted_risk": risk,
        "disagreement": disagreement,
        "g_peer": int(bool(sc["g_peer"][0])),
        "d_shape": float(sc["d_shape"][0]),
        "d_native": float(sc["d_native"][0]),
        "abs_z1": float(desc.common[2]),
        "abs_z2": float(desc.common[5]),
        "abs_z3": float(desc.common[8]),
        "estimated_eps": float(desc.estimated_eps),
        "p_minus": float(prod[0]),
        "p_stay": float(prod[1]),
        "p_plus": float(prod[2]),
        "features": x.tolist(),
    }


def select_and_freeze(branch: Path) -> dict:
    arrays = np.load(branch / "training_arrays_centers_v0.npz", allow_pickle=True)
    x_train = arrays["features"]
    counts = arrays["counts"]
    ad = ApplicabilityDomain.load(branch / "applicability_domain_centers_v0")
    calibrator = RiskCalibrator.load(branch / "risk_calibrator_centers_v0.json")
    models = {
        "activity_direction_hurdle": KernelHurdleMultinomial.load(
            branch / "production_model_activity_direction_hurdle"
        ),
        "kernel_nn_euclidean": KernelNeighborBaseline.fit(x_train, counts),
        "hybrid_hellinger": HybridNativeKernel.fit(
            x_train, counts, native_metric="hellinger"
        ),
        "softmax_stump_boost": SoftmaxStumpBoost.fit(
            x_train, counts, n_estimators=30
        ),
    }

    bank = np.load(BANK_PATH, allow_pickle=True)
    rep_mask = bank["representation"].astype(str) == REP
    peer_ok = np.isin(bank["peer_count"], [8, 16])
    mask = rep_mask & peer_ok
    idx = np.where(mask)[0]
    print(f"Centers peer-matched bank rows: {idx.size}")

    keys = [
        _quantize_key(
            float(bank["abs_z1"][i]),
            float(bank["abs_z2"][i]),
            int(bank["peer_count"][i]),
            float(bank["estimated_eps"][i]),
        )
        for i in idx
    ]
    occ = Counter(keys)
    max_occ = max(occ.values()) if occ else 1

    # Coarse priority on bank descriptors (full-feature risk after construct).
    common = bank["features_common"][idx]
    preds = []
    for name in ("kernel_nn_euclidean", "activity_direction_hurdle", "softmax_stump_boost"):
        # Common-only disagreement proxy for ranking pool.
        m = {
            "kernel_nn_euclidean": KernelNeighborBaseline.fit(
                x_train[:, :N_COMMON], counts
            ),
            "activity_direction_hurdle": KernelHurdleMultinomial.fit(
                x_train[:, :N_COMMON], counts
            ),
            "softmax_stump_boost": SoftmaxStumpBoost.fit(
                x_train[:, :N_COMMON], counts, n_estimators=20
            ),
        }[name]
        preds.append(m.predict_proba(common))
    disagreement = np.zeros(idx.size, dtype=np.float64)
    pairs = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            disagreement += 0.5 * np.sum(np.abs(preds[i] - preds[j]), axis=1)
            pairs += 1
    disagreement /= max(pairs, 1)

    ood = np.nan_to_num(bank["d_combined"][idx].astype(np.float64), nan=0.0)
    ood_n = (ood - ood.min()) / max(float(ood.max() - ood.min()), 1e-12)
    dis_n = (disagreement - disagreement.min()) / max(
        float(disagreement.max() - disagreement.min()), 1e-12
    )
    occ_n = np.asarray([occ[k] / max_occ for k in keys], dtype=np.float64)
    priority = occ_n + ood_n + dis_n
    priority = priority + 0.15 * bank["ood_combined"][idx].astype(np.float64)
    priority = priority - 0.25 * bank["ood_peer"][idx].astype(np.float64)

    # Build candidate pools by source/bucket intent.
    sources = bank["source"][idx].astype(str)
    dynamics = bank["dynamics"][idx].astype(str)
    pool_meta = []
    for local, bank_i in enumerate(idx):
        pool_meta.append(
            {
                "local": local,
                "bank_index": int(bank_i),
                "priority": float(priority[local]),
                "occupancy_n": float(occ_n[local]),
                "disagreement_proxy": float(disagreement[local]),
                "ood_n": float(ood_n[local]),
                "n_agents": int(bank["n_agents"][bank_i]),
                "peer_count": int(bank["peer_count"][bank_i]),
                "abs_z1": float(bank["abs_z1"][bank_i]),
                "abs_z2": float(bank["abs_z2"][bank_i]),
                "estimated_eps": float(bank["estimated_eps"][bank_i]),
                "source": str(sources[local]),
                "init_kind": str(bank["init_kind"][bank_i]),
                "dynamics": str(dynamics[local]),
                "t": int(bank["t"][bank_i]),
            }
        )

    def _take_bucket(
        candidates: list[dict], *, n: int, sort_key: str
    ) -> list[dict]:
        ordered = sorted(candidates, key=lambda r: -float(r[sort_key]))
        kept: list[dict] = []
        seen: set = set()
        for row in ordered:
            key = _dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            kept.append(row)
            if len(kept) >= n:
                break
        return kept

    high_occ_risk_pool = [
        r
        for r in pool_meta
        if r["occupancy_n"] >= 0.35 and r["ood_n"] >= 0.35
    ]
    disagree_pool = [r for r in pool_meta if r["disagreement_proxy"] > 0]
    drift_pool = [r for r in pool_meta if r["source"] == "B_k0_drift"]

    al_high = _take_bucket(high_occ_risk_pool, n=16, sort_key="priority")
    used = {_dedupe_key(r) for r in al_high}
    al_dis = [
        r
        for r in _take_bucket(disagree_pool, n=40, sort_key="disagreement_proxy")
        if _dedupe_key(r) not in used
    ][:8]
    used |= {_dedupe_key(r) for r in al_dis}
    al_drift = [
        r
        for r in _take_bucket(drift_pool, n=40, sort_key="priority")
        if _dedupe_key(r) not in used
    ][:6]

    al_rows = al_high + al_dis + al_drift
    # Backfill if short
    if len(al_rows) < 30:
        for r in sorted(pool_meta, key=lambda x: -x["priority"]):
            if _dedupe_key(r) in used:
                continue
            al_rows.append(r)
            used.add(_dedupe_key(r))
            if len(al_rows) >= 30:
                break
    assert len(al_rows) == 30, len(al_rows)

    buckets = (
        ["high_occupancy_high_risk"] * 16
        + ["model_disagreement"] * 8
        + ["k0_or_drift"] * 6
    )

    frozen_fields = []
    review_rows = []
    prospective = []

    print("Constructing 30 AL fields...")
    for rank, (row, bucket) in enumerate(zip(al_rows, buckets)):
        print(f"  AL {rank:02d} bucket={bucket} bank={row['bank_index']}")
        snap = construct_sibling_field(row, row["bank_index"])
        field_id = f"cent_al_f{rank:02d}_n{snap['n_agents']}_p{snap['peer_count']}"
        scored = _score_candidate(
            snap["fractions"],
            snap["peer_count"],
            ad=ad,
            calibrator=calibrator,
            models=models,
        )
        physical_hash = _sha256_json(
            {
                "fractions": snap["fractions"],
                "peer_count": snap["peer_count"],
                "n_bins": 24,
            }
        )
        spec = {
            "id": field_id,
            "kind": "fixed",
            "concentration": 0.0,
            "peer_count": snap["peer_count"],
            "n_bins": 24,
            "fixed_fractions": list(snap["fractions"]),
        }
        STIMULUS_CATALOG[field_id] = StimulusSpec(
            kind="fixed",
            concentration=0.0,
            peer_count=snap["peer_count"],
            n_bins=24,
            fixed_fractions=snap["fractions"],
        )
        hist = _histogram_from_fractions(snap["fractions"], snap["peer_count"])
        prompt = build_representation_prompt_from_histogram(REP, hist)
        frozen_fields.append(
            {
                "field_id": field_id,
                "role": "active_learning",
                "bucket": bucket,
                "bank_index": row["bank_index"],
                "physical_hash": physical_hash,
                "stimulus": spec,
                "meta": snap,
                "predicted_risk": scored["predicted_risk"],
                "disagreement": scored["disagreement"],
                "occupancy_n": row["occupancy_n"],
                "prompt_sha256": {REP: _sha256_text(prompt)},
                "p_hat_pre": {
                    "p_minus": scored["p_minus"],
                    "p_stay": scored["p_stay"],
                    "p_plus": scored["p_plus"],
                },
            }
        )
        review_rows.append(
            {
                "field_id": field_id,
                "role": "active_learning",
                "bucket": bucket,
                "constructible": "yes_fixed_histogram",
                "peer_count": snap["peer_count"],
                "peer_ok": snap["peer_count"] in {8, 16},
                "physical_hash": physical_hash[:16],
                "source": snap["source"],
                "init_kind": snap["init_kind"],
                "dynamics": snap["dynamics"],
                "occupancy_n": row["occupancy_n"],
                "predicted_risk": scored["predicted_risk"],
                "disagreement": scored["disagreement"],
                "abs_z1": scored["abs_z1"],
                "abs_z2": scored["abs_z2"],
                "match_score_vs_bank": snap["match_score"],
                "pathology_flag": (
                    "warn_loose_match" if snap["match_score"] > 0.15 else "ok"
                ),
                "review_status": "pending_human_review",
            }
        )
        prospective.append(
            {
                "field_id": field_id,
                "bucket": bucket,
                "R_pre": scored["predicted_risk"],
                "p_minus_pre": scored["p_minus"],
                "p_stay_pre": scored["p_stay"],
                "p_plus_pre": scored["p_plus"],
                "disagreement": scored["disagreement"],
                "g_peer": scored["g_peer"],
                "d_shape": scored["d_shape"],
                "d_native": scored["d_native"],
                "abs_z1": scored["abs_z1"],
                "abs_z2": scored["abs_z2"],
                "physical_hash": physical_hash,
            }
        )

    print("Adding 6 controls...")
    for control in CONTROLS:
        shift = int(control.get("shift_bins", 0))
        fractions, peer = _control_fractions(control)
        field_id = control["field_id"]
        scored = _score_candidate(
            fractions, peer, ad=ad, calibrator=calibrator, models=models
        )
        physical_hash = _sha256_json(
            {"fractions": fractions, "peer_count": peer, "n_bins": 24}
        )
        spec = {
            "id": field_id,
            "kind": "fixed",
            "concentration": 0.0,
            "peer_count": peer,
            "n_bins": 24,
            "fixed_fractions": list(fractions),
            "derived_from_catalog": control["catalog_id"],
            "shift_bins": shift,
        }
        STIMULUS_CATALOG[field_id] = StimulusSpec(
            kind="fixed",
            concentration=0.0,
            peer_count=peer,
            n_bins=24,
            fixed_fractions=fractions,
        )
        hist = _histogram_from_fractions(fractions, peer)
        prompt = build_representation_prompt_from_histogram(REP, hist)
        frozen_fields.append(
            {
                "field_id": field_id,
                "role": control["role"],
                "bucket": "control",
                "bank_index": None,
                "physical_hash": physical_hash,
                "stimulus": spec,
                "meta": {
                    "source": "catalog_control",
                    "catalog_id": control["catalog_id"],
                    "peer_count": peer,
                    "notes": control["notes"],
                    "shift_bins": shift,
                    "construction": "catalog_snapshot_fixed",
                },
                "predicted_risk": scored["predicted_risk"],
                "disagreement": scored["disagreement"],
                "occupancy_n": None,
                "prompt_sha256": {REP: _sha256_text(prompt)},
                "p_hat_pre": {
                    "p_minus": scored["p_minus"],
                    "p_stay": scored["p_stay"],
                    "p_plus": scored["p_plus"],
                },
            }
        )
        review_rows.append(
            {
                "field_id": field_id,
                "role": control["role"],
                "bucket": "control",
                "constructible": "yes_catalog_fixed_snapshot",
                "peer_count": peer,
                "peer_ok": peer in {8, 16},
                "physical_hash": physical_hash[:16],
                "source": "catalog_control",
                "init_kind": control["catalog_id"],
                "dynamics": "none",
                "occupancy_n": "",
                "predicted_risk": scored["predicted_risk"],
                "disagreement": scored["disagreement"],
                "abs_z1": scored["abs_z1"],
                "abs_z2": scored["abs_z2"],
                "match_score_vs_bank": 0.0,
                "pathology_flag": "ok",
                "review_status": "pending_human_review",
            }
        )
        prospective.append(
            {
                "field_id": field_id,
                "bucket": "control",
                "R_pre": scored["predicted_risk"],
                "p_minus_pre": scored["p_minus"],
                "p_stay_pre": scored["p_stay"],
                "p_plus_pre": scored["p_plus"],
                "disagreement": scored["disagreement"],
                "g_peer": scored["g_peer"],
                "d_shape": scored["d_shape"],
                "d_native": scored["d_native"],
                "abs_z1": scored["abs_z1"],
                "abs_z2": scored["abs_z2"],
                "physical_hash": physical_hash,
            }
        )

    assert len(frozen_fields) == 36

    fields_path = branch / "cent3_frozen_fields_v0.json"
    fields_path.write_text(
        json.dumps(
            {
                "version": "centers-cent3-pilot-fields-v0",
                "representation": REP,
                "n_fields": 36,
                "n_al": 30,
                "n_controls": 6,
                "bucket_targets": {
                    "high_occupancy_high_risk": 16,
                    "model_disagreement": 8,
                    "k0_or_drift": 6,
                    "control": 6,
                },
                "production_model": "activity_direction_hurdle",
                "fields": frozen_fields,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    catalog_path = branch / "cent3_stimulus_catalog_v0.json"
    catalog = {f["field_id"]: f["stimulus"] for f in frozen_fields}
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    _write_csv(branch / "cent3_human_review_v0.csv", review_rows)
    _write_csv(branch / "cent3_prospective_lock_v0.csv", prospective)
    prospective_json = {
        "status": "pre_acquisition_locked",
        "representation": REP,
        "production_model": "activity_direction_hurdle",
        "risk_calibrator": "risk_calibrator_centers_v0.json",
        "n_fields": 36,
        "rows": prospective,
    }
    (branch / "cent3_prospective_lock_v0.json").write_text(
        json.dumps(prospective_json, indent=2), encoding="utf-8"
    )

    profile_ids = [f["field_id"] for f in frozen_fields]
    protocols = {}
    for block_id, base_seed, schedule_seed in (
        ("block_1", 2026072431, 2026072481),
        ("block_2", 2026072432, 2026072482),
    ):
        protocol = {
            "protocol_name": "centers_cent3_coverage_pilot",
            "protocol_version": f"centers-cent3-v0-{block_id}",
            "prompt_version": "response-law-v0.1",
            "seed_block_id": block_id,
            "base_seed": base_seed,
            "seed_strategy": "hash_v1",
            "schedule_seed": schedule_seed,
            "offset_design": {"kind": "fixed", "values": [0.0]},
            "offset_radians": [0.0],
            "stimulus_profiles": profile_ids,
            "stimulus_catalog_file": str(catalog_path.relative_to(ROOT).as_posix()),
            "frozen_fields_file": str(fields_path.relative_to(ROOT).as_posix()),
            "representations": [REP],
            "repetitions_per_condition": 16,
            "bootstrap_samples": 400,
            "bootstrap_pseudocount": 0.5,
            "primary_fourier_order": 1,
            "sensitivity_fourier_orders": [1],
            "criteria": {
                "minimum_valid_samples_per_condition": 12,
                "purpose": "centers_prospective_risk_validation",
            },
            "pilot_rules": {
                "measure_all_fields": True,
                "no_post_hoc_field_dropping": True,
                "centers_only": True,
                "peer_counts_allowed": [8, 16],
                "collective_N_candidates": [9, 17],
                "prospective_eval_before_refit": True,
                "no_cyclic_shift_augmentation": True,
            },
            "post_pilot_decision_rules": {
                "require_risk_spearman_positive_on_new_fields": True,
                "require_high_risk_error_enrichment": True,
                "require_centers_phenotype_checks": True,
                "moments_thresholds_not_copied": True,
            },
            "cost_ceiling_calls": 1152,
            "expected_calls_this_block": 36 * 1 * 16,
        }
        out_proto = (
            ROOT
            / "experiments"
            / "stage_b_response_law"
            / f"protocol_centers_cent3_v0_{block_id}.json"
        )
        out_proto.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        protocols[block_id] = {
            "path": str(out_proto.relative_to(ROOT).as_posix()),
            "sha256": _sha256_file(out_proto),
            "expected_calls": protocol["expected_calls_this_block"],
        }

    decision = json.loads((branch / "cent12_decision.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "pending_human_review_and_paid_authorization",
        "paid_run_authorized": False,
        "branch": "centers",
        "version": "centers-cent3-pilot-v0",
        "representation": REP,
        "scientific_question": (
            "Can center-bin encoding yield a predictable microscopic surrogate, "
            "and does it select different macroscopic dynamics from moments?"
        ),
        "offline_gate_pass": decision.get("offline_gate_pass"),
        "production_model": decision.get("production_model"),
        "dataset_manifest_sha256": _sha256_file(branch / "dataset_manifest_v0.json"),
        "risk_calibrator_sha256": _sha256_file(
            branch / "risk_calibrator_centers_v0.json"
        ),
        "ad_sha256": _sha256_file(
            branch / "applicability_domain_centers_v0" / "meta.json"
        ),
        "cent12_decision_sha256": _sha256_file(branch / "cent12_decision.json"),
        "field_bank_sha256": _sha256_file(BANK_PATH),
        "frozen_fields_sha256": _sha256_file(fields_path),
        "stimulus_catalog_sha256": _sha256_file(catalog_path),
        "prospective_lock_sha256": _sha256_file(
            branch / "cent3_prospective_lock_v0.json"
        ),
        "selected_field_ids": profile_ids,
        "bucket_counts": dict(Counter(f["bucket"] for f in frozen_fields)),
        "protocols": protocols,
        "expected_total_calls": 1152,
        "cost_assumptions": {
            "model": "gpt-5.4-mini",
            "input_usd_per_m": 0.75,
            "output_usd_per_m": 4.5,
        },
    }
    (branch / "cent3_freeze_manifest_v0.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch-dir",
        type=Path,
        default=BRANCH,
    )
    parser.add_argument(
        "--skip-circular",
        action="store_true",
    )
    parser.add_argument(
        "--skip-select",
        action="store_true",
        help="only run circular holdout compare",
    )
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir
    if not args.skip_circular:
        print("Native distance holdout (Hellinger vs circular EMD)...")
        circ = compare_circular_holdout(branch)
        print(
            json.dumps(
                {
                    "preferred": circ["preferred_native_metric_by_holdout_log_loss"],
                    "hellinger_ll": circ["metrics"]["hellinger"]["log_loss"],
                    "circular_ll": circ["metrics"]["circular_emd"]["log_loss"],
                },
                indent=2,
            )
        )
    if args.skip_select:
        return 0
    print("Selecting + freezing CENT-3 pilot fields...")
    manifest = select_and_freeze(branch)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "n_fields": len(manifest["selected_field_ids"]),
                "buckets": manifest["bucket_counts"],
                "expected_calls": manifest["expected_total_calls"],
                "manifest": str((branch / "cent3_freeze_manifest_v0.json").as_posix()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
