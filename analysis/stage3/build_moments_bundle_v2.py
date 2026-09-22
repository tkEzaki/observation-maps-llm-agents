"""Build and hash-lock moments_bundle_v2 (regime-aware stay; never overwrite v1)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.applicability_domain import (  # noqa: E402
    ApplicabilityDomain,
    DEFAULT_TRAIN_PEERS,
)
from analysis.stage3.dataset import (  # noqa: E402
    build_unified_dataset_v2,
    write_dataset_csv,
)
from analysis.stage3.models import (  # noqa: E402
    KernelHurdleMultinomial,
    RegimeAwareKernelHurdle,
)
from analysis.stage3.oof_residuals import (  # noqa: E402
    build_oof_residual_table,
    default_stay_indices,
    write_csv,
)
from analysis.stage3.risk_calibrator import RiskCalibrator, monotone_ok  # noqa: E402
from analysis.stage3.validation import (  # noqa: E402
    action_metrics,
    moments_abstention_activation_check,
)
from analysis.stage_c.run_surrogate_stage_c import run_one  # noqa: E402
from circlemap.field_features import COMMON_FEATURE_NAMES  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.field_features import extract_field_descriptors  # noqa: E402

REP = "moments_m1_m3"
BUNDLE_VERSION = "stage3b-moments-v2"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    return hashes


def _group_maps(rows: list) -> dict[str, list[str]]:
    return {
        "leave_one_profile_out": [row.profile_family for row in rows],
        "leave_one_eps_level_out": [row.epsilon_level for row in rows],
        "sparse_realization_holdout": [row.sparse_realization for row in rows],
        "offset_group_holdout": [row.offset_group for row in rows],
        "acquisition_block_holdout": [row.acquisition_block for row in rows],
    }


def _replay_stay_check(
    *,
    model_excl: RegimeAwareKernelHurdle,
    model_v1: KernelHurdleMultinomial,
    fields_path: Path,
) -> list[dict]:
    fields = json.loads(fields_path.read_text(encoding="utf-8"))["fields"]
    rows = []
    for field in fields:
        fr = np.asarray(field["fixed_fractions"], dtype=np.float64)
        hist = RelativePhaseHistogram(
            edges=np.linspace(-np.pi, np.pi, fr.size + 1),
            fractions=fr,
            peer_count=int(field["peer_count"]),
        )
        desc = extract_field_descriptors(REP, hist)
        x = np.concatenate([desc.common, desc.native])[None, :]
        p2 = model_excl.predict_proba(x)[0]
        p1 = model_v1.predict_proba(x)[0]
        rows.append(
            {
                "field_id": field["field_id"],
                "source_bucket": field["source_bucket"],
                "p_stay_v1": float(p1[1]),
                "p_stay_v2_excl_replay": float(p2[1]),
                "replay_p_stay_obs": (
                    # from locked prospective file if we recompute later; placeholder
                    float("nan")
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts" / "moments_bundle_v2",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument("--dense-seeds", type=int, default=16)
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    v1_dir = args.artifacts / "moments_bundle_v1"
    if not v1_dir.exists():
        raise SystemExit("moments_bundle_v1 missing — refuse to build v2")
    if args.out_dir.resolve() == v1_dir.resolve():
        raise SystemExit("refusing to overwrite moments_bundle_v1")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    fields_path = (
        ROOT
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "replay_fields_v1.json"
    )
    replay_csv = (
        ROOT
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "replay_vs_v1_prospective.csv"
    )
    obs_stay = {}
    if replay_csv.exists():
        with replay_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                obs_stay[row["field_id"]] = float(row["p_stay_obs"])

    print("Loading v2 unified dataset (Stage B + pilot + replay)...")
    all_rows, packed = build_unified_dataset_v2(ROOT, replay_fields_path=fields_path)
    write_dataset_csv(all_rows, args.out_dir / "training_rows_v2.csv")
    rep_rows = [r for r in all_rows if r.representation == REP]
    x = packed[REP]["features"]
    counts = packed[REP]["counts"]
    unresolved = packed[REP]["unresolved_near_zero"]
    stay_idx = default_stay_indices()
    abs_z1_index = COMMON_FEATURE_NAMES.index("abs_z1")
    bal_index = COMMON_FEATURE_NAMES.index("antipodal_balance")

    source_counts: dict[str, int] = {}
    for row in rep_rows:
        source_counts[row.source_family] = source_counts.get(row.source_family, 0) + 1
    print("  rows by source:", source_counts)

    # Fit excluding replay for clean stay check on collective fields
    non_replay = np.asarray(
        [r.source_family != "collective_replay" for r in rep_rows], dtype=bool
    )
    print("Fitting v2 regime hurdle (full) and excl-replay check model...")
    model_v2 = RegimeAwareKernelHurdle.fit(
        x,
        counts,
        stay_feature_indices=stay_idx,
        abs_z1_index=abs_z1_index,
        antipodal_balance_index=bal_index,
    )
    model_excl = RegimeAwareKernelHurdle.fit(
        x[non_replay],
        counts[non_replay],
        stay_feature_indices=stay_idx,
        abs_z1_index=abs_z1_index,
        antipodal_balance_index=bal_index,
    )
    model_v1 = KernelHurdleMultinomial.load(v1_dir / "kernel_hurdle")

    stay_check = _replay_stay_check(
        model_excl=model_excl, model_v1=model_v1, fields_path=fields_path
    )
    for row in stay_check:
        row["replay_p_stay_obs"] = obs_stay.get(row["field_id"], float("nan"))
    write_csv(args.out_dir / "replay_stay_check_excl_replay_train.csv", stay_check)
    coll = [
        r
        for r in stay_check
        if r["source_bucket"] in {"neg", "pos", "zero"}
        and np.isfinite(r["replay_p_stay_obs"])
    ]
    mean_v2_stay = float(np.mean([r["p_stay_v2_excl_replay"] for r in coll]))
    mean_v1_stay = float(np.mean([r["p_stay_v1"] for r in coll]))
    mean_obs = float(np.mean([r["replay_p_stay_obs"] for r in coll]))
    print(
        f"  collective-field stay (excl-replay fit): "
        f"v1={mean_v1_stay:.3f} v2={mean_v2_stay:.3f} obs={mean_obs:.3f}"
    )

    # Holdout block metrics
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
        abs_z1_index=abs_z1_index,
        antipodal_balance_index=bal_index,
    )
    pred = model_hold.predict_proba(x[test_mask])
    hold_metrics = action_metrics(pred, counts[test_mask])
    stay_mom = moments_abstention_activation_check(
        [rep_rows[i] for i, f in enumerate(test_mask) if f],
        pred,
        representation=REP,
    )
    print(
        f"  holdout {test_block}: log_loss={hold_metrics['log_loss']:.3f} "
        f"stay_gap={stay_mom.get('moments_stay_gap', float('nan')):.3f}"
    )

    print("OOF risk recalibration...")
    oof_rows = build_oof_residual_table(
        x,
        counts,
        _group_maps(rep_rows),
        representation=REP,
        unresolved=unresolved,
    )
    # Replace kernel predictions in OOF with regime model? build_oof uses kernel_nn
    # internally — keep for risk features consistency, or fit calibrator on
    # regime-model residuals for production R.
    # Compute regime OOF residuals quickly via acquisition-block style folds.
    cal_rows = []
    for scheme, groups in _group_maps(rep_rows).items():
        if scheme == "sparse_realization_holdout":
            active = [i for i, g in enumerate(groups) if g != "na"]
            if len(set(groups[i] for i in active)) < 2:
                continue
            indices = active
        else:
            indices = list(range(len(rep_rows)))
        uniq = sorted({groups[i] for i in indices})
        for held in uniq:
            te = [i for i in indices if groups[i] == held]
            tr = [i for i in indices if groups[i] != held]
            if len(te) < 5 or len(tr) < 50:
                continue
            m = RegimeAwareKernelHurdle.fit(
                x[tr],
                counts[tr],
                stay_feature_indices=stay_idx,
                abs_z1_index=abs_z1_index,
                antipodal_balance_index=bal_index,
            )
            ad_fold = ApplicabilityDomain.fit(x[tr], representation=REP)
            p = m.predict_proba(x[te])
            empir = counts[te] / np.maximum(np.sum(counts[te], axis=1, keepdims=True), 1)
            tv = 0.5 * np.sum(np.abs(p - empir), axis=1)
            sc = ad_fold.score(x[te], unresolved_near_zero=unresolved[te])
            for j, idx in enumerate(te):
                cal_rows.append(
                    {
                        "representation": REP,
                        "scheme": scheme,
                        "d_shape": float(sc["d_shape"][j]),
                        "d_native": float(sc["d_native"][j]),
                        "local_density": float(sc["local_density"][j]),
                        "u_ensemble": 0.0,
                        "d_orientation": float(sc["d_orientation"][j]),
                        "policy_a": float(sc["policy_a"][j]),
                        "g_peer": int(bool(sc["g_peer"][j])),
                        "e_tv": float(tv[j]),
                    }
                )
    calibrator = RiskCalibrator.fit(cal_rows)
    calibrator.save(args.out_dir / "risk_calibrator.json")
    print(
        f"  risk Spearman={calibrator.train_spearman:.3f} "
        f"monotone={monotone_ok(calibrator.bin_calibration)}"
    )

    print("Fitting AD + saving production model...")
    ad = ApplicabilityDomain.fit(x, representation=REP)
    ad.save(args.out_dir / "applicability_domain")
    model_v2.save(args.out_dir / "regime_kernel_hurdle")
    # reload smoke
    reloaded = RegimeAwareKernelHurdle.load(args.out_dir / "regime_kernel_hurdle")
    delta = float(
        np.max(np.abs(model_v2.predict_proba(x[:32]) - reloaded.predict_proba(x[:32])))
    )
    if delta > 1e-12:
        raise RuntimeError(f"reload mismatch {delta}")

    np.savez_compressed(
        args.out_dir / "training_arrays_moments.npz",
        features=x,
        counts=counts,
        unresolved_near_zero=unresolved,
    )
    hyper = {
        "model": "regime_kernel_hurdle",
        "representation": REP,
        "balance_min": model_v2.balance_min,
        "abs_z1_max": model_v2.abs_z1_max,
        "stay_feature_indices": list(stay_idx),
        "stay_feature_names": [COMMON_FEATURE_NAMES[i] for i in stay_idx],
        "length_scale_stay_abstain": 1.2,
        "length_scale_stay_active": 0.25,
        "k_neighbors_active": 6,
        "alpha_active_stay": [1.5, 0.15],
        "length_scale_dir": 1.5,
        "note": "no K/t/source features; regime from antipodal_balance and abs_z1 only",
    }
    (args.out_dir / "hyperparameters.json").write_text(
        json.dumps(hyper, indent=2), encoding="utf-8"
    )
    r_hat = calibrator.predict_rows(cal_rows)
    risk_threshold = float(np.quantile(r_hat, 0.8))
    ood = {
        "version": "stage3b-moments-ood-v2",
        "representation": REP,
        "peer_hard_gate": {
            "allowed_peers": list(DEFAULT_TRAIN_PEERS),
            "action": "reject",
        },
        "collective_n_allowed": [9, 17],
        "risk": {
            "calibrator": "risk_calibrator.json",
            "threshold_quantile": 0.8,
            "risk_threshold": risk_threshold,
            "interpretation": "ranking_applicability_not_precise_error",
        },
        "stage_c": {
            "authorized": False,
            "requires": "v2_envelope_compare_then_explicit_go",
        },
        "parent_bundle": "moments_bundle_v1",
        "changes": "regime_aware_stay_gate_plus_replay_training",
    }
    (args.out_dir / "ood_policy.json").write_text(
        json.dumps(ood, indent=2), encoding="utf-8"
    )
    provenance = {
        "bundle_version": BUNDLE_VERSION,
        "n_training_rows_moments": len(rep_rows),
        "n_rows_by_source": source_counts,
        "holdout_block": test_block,
        "holdout_metrics": hold_metrics,
        "moments_stay": stay_mom,
        "collective_stay_excl_replay": {
            "mean_p_stay_v1": mean_v1_stay,
            "mean_p_stay_v2": mean_v2_stay,
            "mean_p_stay_obs": mean_obs,
            "improved": mean_v2_stay < mean_v1_stay,
        },
        "risk_spearman": calibrator.train_spearman,
        "reload_smoke_max_abs_delta": delta,
        "parent_v1_aggregate_sha256": json.loads(
            (v1_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("aggregate_sha256"),
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    # Also keep legacy OOF table for reference
    write_csv(args.out_dir / "oof_residual_table_v2.csv", oof_rows)

    file_hashes = _hash_tree(args.out_dir)
    manifest = {
        "status": "hash_locked",
        "bundle_version": BUNDLE_VERSION,
        "grade": "selection_grade_v2_candidate",
        "representation": REP,
        "production_model": "regime_kernel_hurdle",
        "freeze_ready": True,
        "stage_c_authorized": False,
        "parent_bundle": "moments_bundle_v1",
        "do_not_overwrite_v1": True,
        "file_sha256": file_hashes,
        "aggregate_sha256": _sha256_json(file_hashes),
        "constraints": {
            "peer_counts_allowed": list(DEFAULT_TRAIN_PEERS),
            "collective_n_allowed": [9, 17],
            "no_k_t_source_features": True,
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    dense_summary = None
    if not args.skip_dense:
        print("Dense K envelope compare v1 vs v2 (T=100, offline)...")
        dense_rows = []
        k_grid = [-0.15, 0.0] + [round(0.02 * i, 2) for i in range(1, 16)]
        for label, model in ("v1", model_v1), ("v2", model_v2):
            # v1 AD/risk for Q only; dynamics use each model
            ad_use = (
                ApplicabilityDomain.load(v1_dir / "applicability_domain")
                if label == "v1"
                else ad
            )
            cal_use = (
                RiskCalibrator.load(v1_dir / "risk_calibrator.json")
                if label == "v1"
                else calibrator
            )
            thr = (
                float(
                    json.loads((v1_dir / "ood_policy.json").read_text(encoding="utf-8"))[
                        "risk"
                    ]["risk_threshold"]
                )
                if label == "v1"
                else risk_threshold
            )
            for n_agents in (9, 17):
                for k in k_grid:
                    finals = []
                    for seed_i in range(args.dense_seeds):
                        seed = 2026072500 + 1000 * n_agents + int(abs(k) * 1000) + seed_i
                        # run_one expects KernelHurdle-like predict_proba — both have it
                        out = run_one(
                            model=model,
                            ad=ad_use,
                            calibrator=cal_use,
                            risk_threshold=thr,
                            n_agents=n_agents,
                            n_steps=100,
                            coupling=float(k),
                            omega_halfwidth=0.05,
                            seed=seed,
                            representation=REP,
                        )
                        finals.append(out["final_r"])
                    fa = np.asarray(finals)
                    dense_rows.append(
                        {
                            "bundle": label,
                            "n_agents": n_agents,
                            "coupling": float(k),
                            "final_r1_p50": float(np.median(fa)),
                            "p_final_r1_gt_0.9": float(np.mean(fa > 0.9)),
                        }
                    )
                    print(
                        f"  {label} N={n_agents} K={k:+.2f} "
                        f"p90={dense_rows[-1]['p_final_r1_gt_0.9']:.2f}"
                    )
        write_csv(args.artifacts / "dense_k_v1_vs_v2.csv", dense_rows)
        # recompute hashes after? dense is outside bundle — OK
        crosses = {}
        for label in ("v1", "v2"):
            crosses[label] = {}
            for n_agents in (9, 17):
                sub = [
                    r
                    for r in dense_rows
                    if r["bundle"] == label
                    and r["n_agents"] == n_agents
                    and r["coupling"] >= 0
                ]
                sub = sorted(sub, key=lambda r: r["coupling"])
                k_cross = None
                for r in sub:
                    if r["p_final_r1_gt_0.9"] >= 0.5:
                        k_cross = r["coupling"]
                        break
                crosses[label][str(n_agents)] = k_cross
        dense_summary = {"k_cross": crosses, "n_seeds": args.dense_seeds}
        (args.artifacts / "dense_k_v1_vs_v2_summary.json").write_text(
            json.dumps(dense_summary, indent=2), encoding="utf-8"
        )

    decision = {
        "status": "moments_bundle_v2_hash_locked",
        "bundle_root": str(args.out_dir.relative_to(ROOT).as_posix()),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "collective_stay_improved": mean_v2_stay < mean_v1_stay,
        "collective_stay": {
            "v1": mean_v1_stay,
            "v2_excl_replay_fit": mean_v2_stay,
            "obs": mean_obs,
        },
        "holdout": hold_metrics,
        "moments_stay_gap": stay_mom.get("moments_stay_gap"),
        "dense_k": dense_summary,
        "stage_c_v0_2": "blocked_until_explicit_go_after_envelope_review",
    }
    (args.artifacts / "stage3b_v2_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
