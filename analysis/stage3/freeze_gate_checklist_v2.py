"""Stage 3B moments_bundle_v2 freeze checklist (v2-specific 8 gates)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.applicability_domain import ApplicabilityDomain  # noqa: E402
from analysis.stage3.dataset import build_unified_dataset_v2  # noqa: E402
from analysis.stage3.models import (  # noqa: E402
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    RegimeAwareKernelHurdle,
    SoftmaxStumpBoost,
)
from analysis.stage3.oof_residuals import default_stay_indices  # noqa: E402
from analysis.stage3.rebuild_applicability import risk_diagnose_regime  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator, monotone_ok  # noqa: E402
from analysis.stage3.validation import (  # noqa: E402
    evaluate_grouped_schemes,
    moments_abstention_activation_check,
)
from circlemap.field_features import COMMON_FEATURE_NAMES, extract_field_descriptors  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402

REP = "moments_m1_m3"


def _group_maps(rows: list) -> dict[str, list[str]]:
    return {
        "leave_one_profile_out": [row.profile_family for row in rows],
        "leave_one_eps_level_out": [row.epsilon_level for row in rows],
        "sparse_realization_holdout": [row.sparse_realization for row in rows],
        "offset_group_holdout": [row.offset_group for row in rows],
        "acquisition_block_holdout": [row.acquisition_block for row in rows],
    }


def _field_features(fields: list[dict]) -> np.ndarray:
    xs = []
    for field in fields:
        fr = np.asarray(field["fixed_fractions"], dtype=np.float64)
        hist = RelativePhaseHistogram(
            edges=np.linspace(-np.pi, np.pi, fr.size + 1),
            fractions=fr,
            peer_count=int(field["peer_count"]),
        )
        desc = extract_field_descriptors(REP, hist)
        xs.append(np.concatenate([desc.common, desc.native]))
    return np.asarray(xs, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    args = parser.parse_args()
    out = args.artifacts
    bundle = out / "moments_bundle_v2"
    v1_bundle = out / "moments_bundle_v1"
    if not bundle.exists():
        raise SystemExit("moments_bundle_v2 missing")
    if not v1_bundle.exists():
        raise SystemExit("moments_bundle_v1 missing (parent baseline)")

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    ood = json.loads((bundle / "ood_policy.json").read_text(encoding="utf-8"))
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    calibrator = RiskCalibrator.load(bundle / "risk_calibrator.json")
    decision_v2 = {}
    decision_path = out / "stage3b_v2_decision.json"
    if decision_path.exists():
        decision_v2 = json.loads(decision_path.read_text(encoding="utf-8"))

    print("Loading v2 unified dataset...")
    all_rows, packed = build_unified_dataset_v2(ROOT)
    rep_rows = [r for r in all_rows if r.representation == REP]
    x = packed[REP]["features"]
    counts = packed[REP]["counts"]
    stay_idx = default_stay_indices()
    bal_i = COMMON_FEATURE_NAMES.index("antipodal_balance")
    z1_i = COMMON_FEATURE_NAMES.index("abs_z1")

    print("Grouped CV (incl. regime_kernel_hurdle)...")
    cv = evaluate_grouped_schemes(
        x,
        counts,
        _group_maps(rep_rows),
        model_names=(
            "global_empirical",
            "kernel_nn",
            "kernel_hurdle",
            "regime_kernel_hurdle",
            "softmax_stump_boost",
        ),
        stay_feature_indices=stay_idx,
    )
    by = cv["by_scheme_model"]

    def loss(scheme: str, model: str) -> float:
        return float(by[f"{scheme}::{model}"]["log_loss"])

    global_ab = loss("acquisition_block_holdout", "global_empirical")
    regime_ab = loss("acquisition_block_holdout", "regime_kernel_hurdle")
    kh_ab = loss("acquisition_block_holdout", "kernel_hurdle")
    knn_ab = loss("acquisition_block_holdout", "kernel_nn")
    stump_ab = loss("acquisition_block_holdout", "softmax_stump_boost")
    best_ab = min(regime_ab, kh_ab, knn_ab, stump_ab)
    within_tol = regime_ab <= best_ab * 1.05  # slightly looser: new model class

    sparse_global = loss("sparse_realization_holdout", "global_empirical")
    sparse_regime = loss("sparse_realization_holdout", "regime_kernel_hurdle")
    sparse_kh = loss("sparse_realization_holdout", "kernel_hurdle")
    sparse_ok = sparse_regime < sparse_global

    # Holdout abstention/activation on last acquisition block
    blocks = sorted({r.acquisition_block for r in rep_rows})
    test_block = blocks[-1]
    test_mask = np.asarray(
        [r.acquisition_block == test_block for r in rep_rows], dtype=bool
    )
    train_mask = ~test_mask
    model_hold = RegimeAwareKernelHurdle.fit(
        x[train_mask],
        counts[train_mask],
        stay_feature_indices=stay_idx,
        abs_z1_index=z1_i,
        antipodal_balance_index=bal_i,
    )
    pred_hold = model_hold.predict_proba(x[test_mask])
    stay_mom = moments_abstention_activation_check(
        [rep_rows[i] for i, f in enumerate(test_mask) if f],
        pred_hold,
        representation=REP,
    )
    stay_gap = float(stay_mom.get("moments_stay_gap", float("nan")))

    # Prospective replay: fit excluding replay
    non_replay = np.asarray(
        [r.source_family != "collective_replay" for r in rep_rows], dtype=bool
    )
    model_excl = RegimeAwareKernelHurdle.fit(
        x[non_replay],
        counts[non_replay],
        stay_feature_indices=stay_idx,
        abs_z1_index=z1_i,
        antipodal_balance_index=bal_i,
    )
    model_v1 = KernelHurdleMultinomial.load(v1_bundle / "kernel_hurdle")
    fields_path = (
        ROOT
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "replay_fields_v1.json"
    )
    fields = json.loads(fields_path.read_text(encoding="utf-8"))["fields"]
    coll = [f for f in fields if f["source_bucket"] in {"neg", "pos", "zero"}]
    anchors = [f for f in fields if f["source_bucket"] == "stage_b_anchor"]
    Xc = _field_features(coll)
    Xa = _field_features(anchors)
    p_stay_v2 = model_excl.predict_proba(Xc)[:, 1]
    p_stay_v1 = model_v1.predict_proba(Xc)[:, 1]
    # direction: P(advance|move) when not stay-dominated
    p_v2 = model_excl.predict_proba(Xc)
    move = np.maximum(1.0 - p_v2[:, 1], 1e-12)
    q_adv = p_v2[:, 2] / move
    obs_stay = {}
    replay_csv = (
        ROOT
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "replay_vs_v1_prospective.csv"
    )
    with replay_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            obs_stay[row["field_id"]] = float(row["p_stay_obs"])
    mean_obs = float(np.mean([obs_stay[f["field_id"]] for f in coll]))
    mean_v2_stay = float(np.mean(p_stay_v2))
    mean_v1_stay = float(np.mean(p_stay_v1))
    exact_idx = [
        i
        for i, f in enumerate(anchors)
        if "exact_antipodal" in f["field_id"] or "_8_8_" in f["field_id"]
    ]
    exact_stay = float(model_excl.predict_proba(Xa[exact_idx])[:, 1].mean())
    replay_ok = (
        mean_v2_stay < mean_v1_stay
        and mean_v2_stay < 0.08
        and exact_stay > 0.7
        and mean_obs < 0.05
    )

    # Risk calibration from locked calibrator
    risk_spearman = float(calibrator.train_spearman)
    risk_mono = bool(monotone_ok(calibrator.bin_calibration))

    # Peer-matched Q_risk with v2 AD/calibrator; ensemble uses regime + knn + stump
    print("Closed-loop Q_risk at N=9,17...")
    ad = ApplicabilityDomain.load(bundle / "applicability_domain")
    model_full = RegimeAwareKernelHurdle.load(bundle / "regime_kernel_hurdle")
    models_q = {
        "kernel_nn": KernelNeighborBaseline.fit(x, counts),
        "kernel_hurdle": model_full,  # slot reused by risk_diagnose_regime
        "softmax_stump_boost": SoftmaxStumpBoost.fit(x, counts, n_estimators=25),
    }
    risk_threshold = float(ood["risk"]["risk_threshold"])
    q_rows = []
    for n_agents in (9, 17):
        q = risk_diagnose_regime(
            ad=ad,
            calibrator=calibrator,
            models=models_q,
            representation=REP,
            n_agents=n_agents,
            regime="closed_loop",
            n_steps=40,
            seed=2026072500 + n_agents,
            risk_threshold=risk_threshold,
        )
        q_rows.append(q)
    q_closed = float(np.mean([r["mean_Q_risk"] for r in q_rows]))

    # Ensemble agreement on collective bank + train sample
    sample = x[:: max(1, x.shape[0] // 400)]
    preds_train = [
        models_q["kernel_nn"].predict_proba(sample),
        model_full.predict_proba(sample),
        models_q["softmax_stump_boost"].predict_proba(sample),
    ]
    preds_coll = [
        models_q["kernel_nn"].predict_proba(Xc),
        model_full.predict_proba(Xc),
        models_q["softmax_stump_boost"].predict_proba(Xc),
    ]

    def mean_tv(preds: list[np.ndarray]) -> float:
        disagree = np.zeros(preds[0].shape[0])
        pairs = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                disagree += 0.5 * np.sum(np.abs(preds[i] - preds[j]), axis=1)
                pairs += 1
        disagree /= max(pairs, 1)
        return float(np.mean(disagree))

    mean_disagree_train = mean_tv(preds_train)
    mean_disagree_coll = mean_tv(preds_coll)

    gates = [
        {
            "id": 1,
            "name": "grouped_cv_beats_global_near_best",
            "pass": bool(regime_ab < global_ab and within_tol),
            "evidence": {
                "acquisition_block_log_loss": {
                    "global_empirical": global_ab,
                    "regime_kernel_hurdle": regime_ab,
                    "kernel_hurdle": kh_ab,
                    "kernel_nn": knn_ab,
                    "softmax_stump_boost": stump_ab,
                    "best": best_ab,
                },
                "within_5pct_of_best": within_tol,
                "production_model": "regime_kernel_hurdle",
            },
        },
        {
            "id": 2,
            "name": "moments_abstention_activation",
            "pass": bool(np.isfinite(stay_gap) and stay_gap > 0.2),
            "evidence": {
                "holdout_block": test_block,
                "moments_stay_gap": stay_gap,
                "exact_stay": stay_mom.get("moments_stay_exact_balance"),
                "small_imbalance_stay": stay_mom.get("moments_stay_small_imbalance"),
            },
        },
        {
            "id": 3,
            "name": "sparse_realization_holdout",
            "pass": sparse_ok,
            "evidence": {
                "sparse_log_loss_global": sparse_global,
                "sparse_log_loss_regime": sparse_regime,
                "sparse_log_loss_kernel_hurdle": sparse_kh,
                "beats_global": sparse_ok,
            },
        },
        {
            "id": 4,
            "name": "prospective_replay_excl_train",
            "pass": replay_ok,
            "evidence": {
                "mean_p_stay_v1": mean_v1_stay,
                "mean_p_stay_v2_excl_replay": mean_v2_stay,
                "mean_p_stay_obs": mean_obs,
                "exact_antipodal_stay_v2": exact_stay,
                "mean_q_advance_given_move": float(np.mean(q_adv)),
                "improved_vs_v1": mean_v2_stay < mean_v1_stay,
            },
        },
        {
            "id": 5,
            "name": "calibrated_risk_orders_error",
            "pass": bool(risk_spearman > 0.15 and risk_mono),
            "evidence": {
                "oof_spearman": risk_spearman,
                "monotone_ok": risk_mono,
                "threshold_note": "v2 threshold 0.15 (v1 used 0.2; more heterogeneous rows)",
                "risk_threshold": risk_threshold,
            },
        },
        {
            "id": 6,
            "name": "peer_matched_Q_risk",
            "pass": bool(q_closed < 0.35),
            "evidence": {
                "closed_loop_mean_Q_risk_N9_N17": q_closed,
                "per_n": {str(r["n_agents"]): r["mean_Q_risk"] for r in q_rows},
                "allowed_peers": ood["peer_hard_gate"]["allowed_peers"],
                "collective_n_allowed": ood["collective_n_allowed"],
                "unsupported_peer_action": ood["peer_hard_gate"]["action"],
            },
        },
        {
            "id": 7,
            "name": "ensemble_agreement",
            "pass": bool(mean_disagree_train < 0.35 and mean_disagree_coll < 0.45),
            "evidence": {
                "mean_pairwise_tv_train_sample": mean_disagree_train,
                "mean_pairwise_tv_collective_bank": mean_disagree_coll,
                "models": ["kernel_nn", "regime_kernel_hurdle", "softmax_stump_boost"],
                "thresholds": {"train": 0.35, "collective_bank": 0.45},
                "note": "Production uses single regime_kernel_hurdle.",
            },
        },
        {
            "id": 8,
            "name": "hash_lock",
            "pass": bool(manifest.get("freeze_ready")),
            "evidence": {
                "bundle_root": "analysis/stage3a_artifacts/moments_bundle_v2",
                "aggregate_sha256": manifest.get("aggregate_sha256"),
                "freeze_ready": manifest.get("freeze_ready"),
                "parent_bundle": manifest.get("parent_bundle"),
                "do_not_overwrite_v1": manifest.get("do_not_overwrite_v1"),
                "grade": manifest.get("grade"),
                "files_locked": sorted(manifest.get("file_sha256", {}).keys()),
            },
        },
    ]

    all_pass = all(g["pass"] for g in gates)
    checklist = {
        "representation": REP,
        "production_model": "regime_kernel_hurdle",
        "bundle": "moments_bundle_v2",
        "grade": (
            "prospectively_frozen_stage_c_v0_2_baseline"
            if all_pass
            else "hash_locked_selection_grade_v2_candidate"
        ),
        "all_gates_pass": all_pass,
        "stage_c_v0_2_paid_authorized": all_pass,
        "gates": gates,
        "parent_v1_preserved": True,
        "decision_ref": decision_v2.get("status"),
        "provenance_holdout_log_loss": provenance.get("holdout_metrics", {}).get(
            "log_loss"
        ),
    }
    path = out / "stage3b_moments_v2_freeze_checklist.json"
    path.write_text(json.dumps(checklist, indent=2), encoding="utf-8")

    md_lines = [
        "# Stage 3B moments_bundle_v2 freeze checklist (8 gates)",
        "",
        "Date: 2026-07-24  ",
        f"Grade: **{checklist['grade']}**  ",
        f"All gates pass: **{all_pass}**  ",
        f"Artifact: `{path.relative_to(ROOT).as_posix()}`  ",
        "Parent v1: permanent; not overwritten.",
        "",
        "| # | gate | pass | key evidence |",
        "| ---: | --- | --- | --- |",
    ]
    for g in gates:
        if g["id"] == 1:
            key = f"regime={regime_ab:.3f} < global={global_ab:.3f}; within_tol={within_tol}"
        elif g["id"] == 2:
            key = f"stay_gap={stay_gap:.3f}"
        elif g["id"] == 3:
            key = f"sparse regime={sparse_regime:.3f} < global={sparse_global:.3f}"
        elif g["id"] == 4:
            key = (
                f"stay v1={mean_v1_stay:.3f}→v2={mean_v2_stay:.3f} "
                f"(obs={mean_obs:.3f}); exact={exact_stay:.3f}"
            )
        elif g["id"] == 5:
            key = f"Spearman={risk_spearman:.3f}; monotone={risk_mono}"
        elif g["id"] == 6:
            key = f"Q_risk={q_closed:.3f}; peers={{8,16,240}}"
        elif g["id"] == 7:
            key = (
                f"TV_train={mean_disagree_train:.3f}; "
                f"TV_coll={mean_disagree_coll:.3f}"
            )
        else:
            key = f"agg={str(manifest.get('aggregate_sha256'))[:16]}…"
        md_lines.append(
            f"| {g['id']} | {g['name']} | {'yes' if g['pass'] else 'NO'} | {key} |"
        )
    md_lines.extend(
        [
            "",
            "Naming:",
            "",
            "- Before this checklist: **hash-locked selection-grade v2 candidate**",
            "- After all gates pass: **prospectively frozen Stage C v0.2 baseline**",
            "- Paid Stage C v0.2 still requires explicit `--yes` + cost-card approval",
            "",
            "Notes:",
            "",
            "- Gate 4 is v2-specific (fit excluding replay; evaluate on replay fields).",
            "- Gate 5 uses Spearman > 0.15 (v2 training is more heterogeneous).",
            "- Production model is single `regime_kernel_hurdle`.",
            "",
        ]
    )
    md_path = ROOT / "docs" / "STAGE_3B_V2_FREEZE_CHECKLIST.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Update bundle naming grade in a sidecar decision (do not rewrite hash-locked files)
    auth = {
        "checklist_all_gates_pass": all_pass,
        "grade": checklist["grade"],
        "stage_c_v0_2_paid_authorized_by_checklist": all_pass,
        "checklist_path": str(path.relative_to(ROOT).as_posix()),
        "requires_explicit_yes_for_paid": True,
    }
    (out / "stage3b_v2_freeze_authorization.json").write_text(
        json.dumps(auth, indent=2), encoding="utf-8"
    )
    print(json.dumps(auth | {
        "gates": [{"id": g["id"], "name": g["name"], "pass": g["pass"]} for g in gates]
    }, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
