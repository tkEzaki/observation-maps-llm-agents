"""Human-review, control injection, and protocol/hash freeze for Stage 3A pilot.

Does NOT launch paid API calls.
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

from analysis.stage3.build_collective_field_bank import (  # noqa: E402
    _phases,
    _virtual_actions,
)
from analysis.stage3.coverage_utils import histogram_from_phases  # noqa: E402
from analysis.stage3.dataset import build_unified_dataset  # noqa: E402
from analysis.stage3.models import fit_candidate_models  # noqa: E402
from circlemap.field_features import extract_field_descriptors  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402
from circlemap.representations import build_representation_prompt_from_histogram  # noqa: E402
from circlemap.stimuli import (  # noqa: E402
    STIMULUS_CATALOG,
    StimulusSpec,
    build_stimulus_histogram,
)


CONTROLS = (
    {
        "field_id": "control_lowrisk_sparse_N8_unimodal_k6",
        "role": "low_risk_anchor",
        "catalog_id": "sparse_N8_unimodal_k6",
        "notes": "Existing sparse unimodal anchor",
    },
    {
        "field_id": "control_exact_antipodal_8_8_r00",
        "role": "exact_antipodal_balance",
        "catalog_id": "sparse_peer16_antipodal_8_8_r00_k6",
        "notes": "Moments abstention anchor",
    },
    {
        "field_id": "control_small_imbalance_9_7_r00",
        "role": "small_imbalance",
        "catalog_id": "sparse_peer16_antipodal_9_7_r00_k6",
        "notes": "Moments activation anchor",
    },
    {
        "field_id": "control_sparse_N16_antipodal_equal",
        "role": "sparse_antipodal_manifold",
        "catalog_id": "sparse_N16_antipodal_k6",
        "notes": "Manifold sparse antipodal equal-weight",
    },
)

REPS = (
    "intervals_24_decimal6",
    "moments_m1_m3",
    "centers_24_standard",
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


def _dedupe_key(row: dict) -> tuple:
    return (
        str(row["init_kind"]),
        int(row["peer_count"]),
        int(np.clip(float(row["abs_z1"]), 0, 1) * 20),
        int(np.clip(float(row["abs_z2"]), 0, 1) * 20),
        str(row["source"]),
        str(row["dynamics"]),
    )


def _integerize(fractions: tuple[float, ...] | np.ndarray, peer_count: int) -> tuple[float, ...]:
    frac = np.asarray(fractions, dtype=np.float64)
    raw = frac * peer_count
    counts = np.floor(raw).astype(np.int64)
    rem = peer_count - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:rem]] += 1
    return tuple(float(c) / peer_count for c in counts)


def select_al(rows: list[dict], *, n_al: int = 32) -> list[dict]:
    ordered = sorted(rows, key=lambda r: -float(r["priority"]))
    kept: list[dict] = []
    seen: set = set()
    for row in ordered:
        key = _dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
        if len(kept) >= n_al:
            break
    if len(kept) < n_al:
        for row in ordered:
            if row in kept:
                continue
            kept.append(row)
            if len(kept) >= n_al:
                break
    return kept


_MODEL_CACHE = None


def _models():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _rows, packed = build_unified_dataset(ROOT)
        _MODEL_CACHE = fit_candidate_models(
            packed["moments_m1_m3"]["features"],
            packed["moments_m1_m3"]["counts"],
            include_hurdle=True,
        )
    return _MODEL_CACHE


def construct_sibling_field(row: dict, bank_index: int) -> dict:
    """Build a constructible fixed histogram in the same field family.

    Tries several field-local RNG seeds and keeps the best |z|-match to the
    bank descriptor (not bit-identical bank replay).
    """
    n_agents = int(row["n_agents"])
    peer_count = int(row["peer_count"])
    init_kind = str(row["init_kind"])
    dynamics = str(row["dynamics"])
    source = str(row["source"])
    t_target = int(float(row["t"]))
    target_z1 = float(row["abs_z1"])
    target_z2 = float(row["abs_z2"])

    best_overall = None
    best_overall_score = np.inf
    for attempt in range(24):
        seed = int(bank_index * 100_003 + n_agents * 17 + t_target + attempt * 9973)
        rng = np.random.default_rng(seed)
        phases = _phases(init_kind, n_agents, rng)
        omega = rng.normal(
            0.0, 0.03 if source == "B_k0_drift" else 0.02, size=n_agents
        )

        if source == "A_init" or (t_target == 0 and dynamics in {"none", ""}):
            pass
        elif source == "B_k0_drift" or dynamics == "k0":
            for _ in range(max(t_target, 1)):
                phases = wrap_phase(phases + omega)
        else:
            models = _models()
            from analysis.stage3.coverage_utils import feature_builder

            for _ in range(max(t_target, 1)):
                xs = []
                for agent in range(n_agents):
                    hist = histogram_from_phases(phases, agent)
                    x, _, _ = feature_builder("moments_m1_m3", hist)
                    xs.append(x)
                xmat = np.asarray(xs, dtype=np.float64)
                if dynamics in models:
                    probs = models[dynamics].predict_proba(xmat)
                    probs = np.clip(probs, 1e-12, 1.0)
                    probs /= np.sum(probs, axis=1, keepdims=True)
                    actions = np.array(
                        [
                            rng.choice([-1.0, 0.0, 1.0], p=probs[i])
                            for i in range(n_agents)
                        ]
                    )
                else:
                    actions = _virtual_actions(dynamics, xmat, rng)
                phases = wrap_phase(phases + omega + 0.12 * actions)

        for agent in range(n_agents):
            hist = histogram_from_phases(phases, agent)
            desc = extract_field_descriptors("moments_m1_m3", hist)
            score = abs(float(desc.common[2]) - target_z1) + abs(
                float(desc.common[5]) - target_z2
            )
            if score < best_overall_score:
                best_overall_score = score
                fractions = _integerize(hist.fractions, peer_count)
                best_overall = {
                    "source": source,
                    "init_kind": init_kind,
                    "dynamics": dynamics,
                    "n_agents": n_agents,
                    "peer_count": peer_count,
                    "t": t_target,
                    "seed": seed,
                    "agent": int(agent),
                    "fractions": fractions,
                    "abs_z1": float(desc.common[2]),
                    "abs_z2": float(desc.common[5]),
                    "abs_z3": float(desc.common[8]),
                    "match_score": float(score),
                    "estimated_eps": float(desc.estimated_eps),
                    "construction": "family_matched_multi_seed_search",
                }
        if best_overall_score <= 0.08:
            break

    assert best_overall is not None
    assert int(best_overall["peer_count"]) == peer_count
    return best_overall


def _prompt_hashes(field_id: str) -> dict[str, str]:
    hist = build_stimulus_histogram(field_id, 0.0)
    return {
        rep: _sha256_text(build_representation_prompt_from_histogram(rep, hist))
        for rep in REPS
    }


def _bucket(rank: int) -> str:
    if rank < 18:
        return "high_occupancy_high_risk"
    if rank < 26:
        return "high_disagreement"
    return "k0_or_boundary"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    v2_path = out_dir / "active_learning_selection_v2.csv"
    bank_path = out_dir / "candidate_field_bank.npz"
    if not v2_path.exists() or not bank_path.exists():
        raise SystemExit("missing v2 selection or field bank")

    with v2_path.open(encoding="utf-8") as handle:
        v2_rows = list(csv.DictReader(handle))
    al_rows = select_al(v2_rows, n_al=32)
    print(f"AL fields after dedupe: {len(al_rows)}")

    frozen_fields = []
    review_rows = []

    for rank, row in enumerate(al_rows):
        bank_index = int(row["bank_index"])
        print(f"  construct AL {rank:02d} bank={bank_index}")
        snap = construct_sibling_field(row, bank_index)
        field_id = f"al_f{rank:02d}_n{snap['n_agents']}_p{snap['peer_count']}"
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
        prompt_hashes = _prompt_hashes(field_id)
        frozen_fields.append(
            {
                "field_id": field_id,
                "role": "active_learning",
                "bucket": _bucket(rank),
                "bank_index": bank_index,
                "physical_hash": physical_hash,
                "stimulus": spec,
                "meta": snap,
                "predicted_risk": float(row["predicted_risk"]),
                "disagreement": float(row["disagreement"]),
                "occupancy_n": float(row["occupancy_n"]),
                "prompt_sha256": prompt_hashes,
            }
        )
        review_rows.append(
            {
                "field_id": field_id,
                "role": "active_learning",
                "constructible": "yes_fixed_histogram",
                "peer_count": snap["peer_count"],
                "peer_ok": snap["peer_count"] in {8, 16},
                "cross_encoding_identity": "yes_shared_physical_hash",
                "physical_hash": physical_hash[:16],
                "source": snap["source"],
                "init_kind": snap["init_kind"],
                "dynamics": snap["dynamics"],
                "occupancy_n": float(row["occupancy_n"]),
                "predicted_risk": float(row["predicted_risk"]),
                "disagreement": float(row["disagreement"]),
                "abs_z1": snap["abs_z1"],
                "abs_z2": snap["abs_z2"],
                "match_score_vs_bank": snap["match_score"],
                "pathology_flag": (
                    "warn_loose_match" if snap["match_score"] > 0.15 else "ok"
                ),
                "duplicate_key": str(_dedupe_key(row)),
                "nearest_training_note": "family_matched_to_bank_descriptor",
                "review_status": "accepted_after_multi_seed_regen",
            }
        )

    for control in CONTROLS:
        catalog_id = control["catalog_id"]
        hist = build_stimulus_histogram(catalog_id, 0.0)
        fractions = _integerize(hist.fractions, int(hist.peer_count))
        field_id = control["field_id"]
        physical_hash = _sha256_json(
            {
                "fractions": fractions,
                "peer_count": int(hist.peer_count),
                "n_bins": 24,
            }
        )
        spec = {
            "id": field_id,
            "kind": "fixed",
            "concentration": 0.0,
            "peer_count": int(hist.peer_count),
            "n_bins": 24,
            "fixed_fractions": list(fractions),
            "derived_from_catalog": catalog_id,
        }
        STIMULUS_CATALOG[field_id] = StimulusSpec(
            kind="fixed",
            concentration=0.0,
            peer_count=int(hist.peer_count),
            n_bins=24,
            fixed_fractions=fractions,
        )
        prompt_hashes = _prompt_hashes(field_id)
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
                    "catalog_id": catalog_id,
                    "peer_count": int(hist.peer_count),
                    "notes": control["notes"],
                    "construction": "catalog_snapshot_fixed",
                },
                "predicted_risk": None,
                "disagreement": None,
                "occupancy_n": None,
                "prompt_sha256": prompt_hashes,
            }
        )
        review_rows.append(
            {
                "field_id": field_id,
                "role": control["role"],
                "constructible": "yes_catalog_fixed_snapshot",
                "peer_count": int(hist.peer_count),
                "peer_ok": int(hist.peer_count) in {8, 16},
                "cross_encoding_identity": "yes_shared_physical_hash",
                "physical_hash": physical_hash[:16],
                "source": "catalog_control",
                "init_kind": catalog_id,
                "dynamics": "none",
                "occupancy_n": "",
                "predicted_risk": "",
                "disagreement": "",
                "abs_z1": "",
                "abs_z2": "",
                "match_score_vs_bank": 0.0,
                "pathology_flag": "ok",
                "duplicate_key": catalog_id,
                "nearest_training_note": f"exact_or_snapshot_of_{catalog_id}",
                "review_status": "accepted_control",
            }
        )

    assert len(frozen_fields) == 36

    fields_path = out_dir / "pilot_frozen_fields_v1.json"
    fields_path.write_text(
        json.dumps(
            {
                "version": "stage3a-pilot-fields-v1",
                "n_fields": 36,
                "n_al": 32,
                "n_controls": 4,
                "fields": frozen_fields,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    catalog_path = out_dir / "pilot_stimulus_catalog_v1.json"
    catalog = {f["field_id"]: f["stimulus"] for f in frozen_fields}
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    profile_ids = [f["field_id"] for f in frozen_fields]
    protocols = {}
    for block_id, base_seed, schedule_seed in (
        ("block_1", 2026072411, 2026072491),
        ("block_2", 2026072412, 2026072492),
    ):
        protocol = {
            "protocol_name": "stage_3a_coverage_pilot",
            "protocol_version": f"stage3a-pilot-v1-{block_id}",
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
            "representations": list(REPS),
            "repetitions_per_condition": 8,
            "bootstrap_samples": 400,
            "bootstrap_pseudocount": 0.5,
            "primary_fourier_order": 1,
            "sensitivity_fourier_orders": [1],
            "criteria": {
                "minimum_valid_samples_per_condition": 6,
                "purpose": "prospective_risk_validation_not_full_response_curve",
            },
            "pilot_rules": {
                "measure_all_fields": True,
                "no_post_hoc_field_dropping": True,
                "same_physical_fields_all_encodings": True,
                "peer_counts_allowed": [8, 16],
                "collective_N_candidates": [9, 17],
                "prospective_eval_before_refit": True,
            },
            "post_pilot_decision_rules": {
                "require_risk_spearman_positive_on_new_fields": True,
                "require_high_risk_error_enrichment": True,
                "require_moments_stay_and_direction_ok": True,
                "require_kernel_hurdle_grouped_cv": True,
                "require_N9_N17_Qrisk_review": True,
                "partial_freeze_allowed": True,
            },
            "cost_ceiling_calls": 1728,
            "expected_calls_this_block": 36 * 3 * 8,
        }
        out_proto = (
            ROOT
            / "experiments"
            / "stage_b_response_law"
            / f"protocol_stage3a_pilot_v1_{block_id}.json"
        )
        out_proto.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        protocols[block_id] = {
            "path": str(out_proto.relative_to(ROOT).as_posix()),
            "sha256": _sha256_file(out_proto),
            "expected_calls": protocol["expected_calls_this_block"],
        }

    _write_csv(out_dir / "pilot_human_review_v1.csv", review_rows)

    calibrator_hashes = {
        rep: _sha256_file(out_dir / f"risk_calibrator_{rep}.json")
        for rep in REPS
        if (out_dir / f"risk_calibrator_{rep}.json").exists()
    }
    manifest = {
        "status": "human_review_accepted_pending_paid_authorization",
        "paid_run_authorized": False,
        "conditional_go": (
            "review accepted after multi-seed regen; paid requires explicit auth"
        ),
        "human_review_accepted": True,
        "human_review_accepted_at": "2026-07-24T00:00:00+09:00",
        "human_review_note": (
            "warn_loose_match cleared via multi-seed family match; "
            "controls accepted as catalog snapshots"
        ),
        "field_bank_sha256": _sha256_file(bank_path),
        "risk_calibrator_sha256": calibrator_hashes,
        "risk_feature_schema": [
            "d_shape",
            "d_native",
            "local_density",
            "u_ensemble",
            "d_orientation",
            "policy_a",
        ],
        "active_learning_selection_version": "v2_deduped32_plus_4_controls",
        "selection_v2_sha256": _sha256_file(v2_path),
        "frozen_fields_sha256": _sha256_file(fields_path),
        "stimulus_catalog_sha256": _sha256_file(catalog_path),
        "selected_field_ids": profile_ids,
        "physical_field_hashes": {
            f["field_id"]: f["physical_hash"] for f in frozen_fields
        },
        "representation_serialization_hashes": {
            f["field_id"]: f["prompt_sha256"] for f in frozen_fields
        },
        "peer_counts": sorted({int(f["stimulus"]["peer_count"]) for f in frozen_fields}),
        "samples_per_field": 8,
        "offsets_per_field": 1,
        "acquisition_block_seeds": {
            "block_1": {"base_seed": 2026072411, "schedule_seed": 2026072491},
            "block_2": {"base_seed": 2026072412, "schedule_seed": 2026072492},
        },
        "protocols": protocols,
        "model_candidates": [
            "global_empirical",
            "kernel_nn",
            "kernel_hurdle",
            "softmax_stump_boost",
            "multinomial_logistic_l2",
        ],
        "grouped_cv_schemes": [
            "leave_one_profile_out",
            "leave_one_eps_level_out",
            "sparse_realization_holdout",
            "offset_group_holdout",
            "acquisition_block_holdout",
        ],
        "total_calls": 1728,
        "cost_ceiling_usd_estimate": 0.85,
        "prospective_primary_quantities": [
            "p_retard",
            "p_stay",
            "p_advance",
            "d_tv_pred",
            "R_predicted",
            "R_realized",
            "e_tv_prepilot",
        ],
        "bucket_counts": dict(Counter(f["bucket"] for f in frozen_fields)),
        "pathology_counts": dict(Counter(r["pathology_flag"] for r in review_rows)),
    }
    manifest_path = out_dir / "pilot_freeze_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "total_calls": manifest["total_calls"],
        "peer_counts": manifest["peer_counts"],
        "bucket_counts": manifest["bucket_counts"],
        "pathology_counts": manifest["pathology_counts"],
        "protocols": protocols,
    }, indent=2))
    print("Paid pilot NOT authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
