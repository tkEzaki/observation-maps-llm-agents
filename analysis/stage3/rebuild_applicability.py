"""Rebuild applicability domain + predictive risk; reselect AL fields.

Does NOT launch paid acquisition. Writes artifacts under stage3a_artifacts/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.applicability_domain import (  # noqa: E402
    ApplicabilityDomain,
    DEFAULT_TRAIN_PEERS,
)
from analysis.stage3.coverage_utils import (  # noqa: E402
    feature_builder,
    histogram_from_phases,
)
from analysis.stage3.dataset import build_unified_dataset  # noqa: E402
from analysis.stage3.models import (  # noqa: E402
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    SoftmaxStumpBoost,
)
from analysis.stage3.oof_residuals import (  # noqa: E402
    build_oof_residual_table,
    default_stay_indices,
    write_csv,
)
from analysis.stage3.risk_calibrator import (  # noqa: E402
    RiskCalibrator,
    monotone_ok,
)
from analysis.stage3.validation import (  # noqa: E402
    action_metrics,
    moments_abstention_activation_check,
)
from circlemap.observation import wrap_phase  # noqa: E402


def _group_maps(rows: list) -> dict[str, list[str]]:
    return {
        "leave_one_profile_out": [row.profile_family for row in rows],
        "leave_one_eps_level_out": [row.epsilon_level for row in rows],
        "sparse_realization_holdout": [row.sparse_realization for row in rows],
        "offset_group_holdout": [row.offset_group for row in rows],
        "acquisition_block_holdout": [row.acquisition_block for row in rows],
    }


def _init_phases(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "uniform":
        return rng.uniform(-np.pi, np.pi, size=n)
    if kind == "unimodal_strong":
        return rng.vonmises(0.0, 4.0, size=n)
    if kind == "antipodal":
        half = n // 2
        return np.concatenate(
            [
                rng.vonmises(0.0, 3.0, size=half),
                rng.vonmises(np.pi, 3.0, size=n - half),
            ]
        )
    raise KeyError(kind)


def _collect(representation: str, phases: np.ndarray):
    xs, flags = [], []
    for agent in range(phases.size):
        histogram = histogram_from_phases(phases, agent)
        x, unresolved, _ = feature_builder(representation, histogram)
        xs.append(x)
        flags.append(unresolved)
    return np.asarray(xs), np.asarray(flags, dtype=bool)


def risk_diagnose_regime(
    *,
    ad: ApplicabilityDomain,
    calibrator: RiskCalibrator,
    models: dict,
    representation: str,
    n_agents: int,
    regime: str,
    n_steps: int,
    seed: int,
    risk_threshold: float,
) -> dict:
    rng = np.random.default_rng(seed)
    phases = _init_phases("uniform", n_agents, rng)
    omega = rng.normal(0.0, 0.03, size=n_agents)
    rows = []
    for t in range(n_steps if regime != "t0_only" else 1):
        x, unresolved = _collect(representation, phases)
        scores = ad.score(x, unresolved_near_zero=unresolved)
        preds = [
            models["kernel_nn"].predict_proba(x),
            models["kernel_hurdle"].predict_proba(x),
            models["softmax_stump_boost"].predict_proba(x),
        ]
        disagree = np.zeros(x.shape[0])
        pairs = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                disagree += 0.5 * np.sum(np.abs(preds[i] - preds[j]), axis=1)
                pairs += 1
        disagree /= max(pairs, 1)
        risk = calibrator.predict_arrays(
            d_shape=scores["d_shape"],
            d_native=scores["d_native"],
            local_density=scores["local_density"],
            u_ensemble=disagree,
            d_orientation=scores["d_orientation"],
            policy_a=scores["policy_a"],
        )
        g_peer = scores["g_peer"]
        high_risk = (~g_peer) & (risk > risk_threshold)
        rows.append(
            {
                "t": t,
                "q_unsupported_peer": float(np.mean(g_peer)),
                "q_high_risk": float(np.mean(high_risk)),
                "q_policy_a": float(np.mean(scores["policy_a"] > 0.5)),
                "mean_R": float(np.mean(risk)),
                "mean_d_shape": float(np.mean(scores["d_shape"])),
                "mean_d_native": float(np.mean(scores["d_native"])),
                "occupancy_weighted_Q_risk": float(
                    np.sum(high_risk) / max(high_risk.size, 1)
                ),
            }
        )
        if regime == "t0_only":
            break
        if regime == "k0_drift":
            phases = wrap_phase(phases + omega)
        elif regime == "closed_loop":
            actions = np.sign(x[:, 1])
            phases = wrap_phase(phases + omega + 0.12 * actions)
    return {
        "representation": representation,
        "n_agents": n_agents,
        "peer_count": n_agents - 1,
        "regime": regime,
        "mean_q_unsupported_peer": float(np.mean([r["q_unsupported_peer"] for r in rows])),
        "mean_q_high_risk": float(np.mean([r["q_high_risk"] for r in rows])),
        "mean_Q_risk": float(np.mean([r["occupancy_weighted_Q_risk"] for r in rows])),
        "mean_R": float(np.mean([r["mean_R"] for r in rows])),
        "final_q_high_risk": rows[-1]["q_high_risk"],
        "final_mean_d_shape": rows[-1]["mean_d_shape"],
    }


def stratified_select(
    bank,
    priority: np.ndarray,
    *,
    n_high_occ_risk: int = 18,
    n_disagreement: int = 8,
    n_k0: int = 6,
    n_maximin: int = 4,
) -> list[int]:
    n = priority.shape[0]
    peer_ok = np.isin(bank["peer_count"], [8, 16])
    selected: list[int] = []

    def fill(mask: np.ndarray, k: int) -> None:
        cand = np.where(mask & peer_ok)[0]
        if cand.size == 0:
            return
        order = cand[np.argsort(-priority[cand])]
        added = 0
        for idx in order:
            if int(idx) in selected:
                continue
            selected.append(int(idx))
            added += 1
            if added >= k:
                break

    keys = [
        (
            int(np.clip(float(bank["abs_z1"][i]), 0, 1) * 10),
            int(bank["peer_count"][i]),
            str(bank["source"][i]),
        )
        for i in range(n)
    ]
    occ = Counter(keys)
    max_occ = max(occ.values()) if occ else 1
    occ_n = np.asarray([occ[k] / max_occ for k in keys], dtype=np.float64)
    if np.any(peer_ok):
        high_occ = occ_n >= np.quantile(occ_n[peer_ok], 0.6)
        high_risk = priority >= np.quantile(priority[peer_ok], 0.55)
    else:
        high_occ = occ_n >= 0
        high_risk = priority >= 0
    fill(high_occ & high_risk, n_high_occ_risk)

    if "disagreement" in bank:
        dis = bank["disagreement"]
        high_dis = dis >= np.quantile(dis[peer_ok], 0.7) if np.any(peer_ok) else dis >= 0
        fill(high_dis, n_disagreement)
    else:
        fill(~np.isin(np.arange(n), selected), n_disagreement)

    k0 = (bank["source"].astype(str) == "B_k0_drift") | (
        bank["dynamics"].astype(str) == "k0"
    )
    fill(k0, n_k0)

    remaining = np.where(peer_ok & ~np.isin(np.arange(n), selected))[0]
    if remaining.size and n_maximin > 0:
        pool = remaining[np.argsort(-priority[remaining])][: min(600, remaining.size)]
        feats = bank["features_canonical"][pool]
        mean = feats.mean(axis=0)
        scale = np.where(feats.std(axis=0) < 1e-8, 1.0, feats.std(axis=0))
        zs = (feats - mean) / scale
        local: list[int] = []
        for _ in range(min(n_maximin, pool.size)):
            if not local:
                local.append(0)
                continue
            sel = zs[local]
            dmin = np.full(zs.shape[0], np.inf)
            for start in range(0, zs.shape[0], 64):
                stop = min(start + 64, zs.shape[0])
                diff = zs[start:stop, None, :] - sel[None, :, :]
                dmin[start:stop] = np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)
            score = dmin.copy()
            for already in local:
                score[already] = -1.0
            local.append(int(np.argmax(score)))
        selected.extend(int(pool[i]) for i in local)

    out: list[int] = []
    for idx in selected:
        if idx not in out:
            out.append(idx)
    need = n_high_occ_risk + n_disagreement + n_k0 + n_maximin
    if len(out) < need:
        for idx in np.argsort(-priority):
            i = int(idx)
            if i not in out and peer_ok[i]:
                out.append(i)
            if len(out) >= need:
                break
    return out[:need]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument("--skip-oof", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading unified dataset...")
    all_rows, packed = build_unified_dataset(ROOT)
    oof_all: list[dict] = []
    risk_summary = {}
    stay_idx = default_stay_indices()
    model_compare_rows: list[dict] = []

    for representation, arrays in packed.items():
        print(f"\n=== {representation} ===")
        rep_rows = [r for r in all_rows if r.representation == representation]
        x = arrays["features"]
        counts = arrays["counts"]
        unresolved = arrays["unresolved_near_zero"]

        if not args.skip_oof:
            print("  building grouped OOF residual table...")
            oof_rows = build_oof_residual_table(
                x,
                counts,
                _group_maps(rep_rows),
                representation=representation,
                unresolved=unresolved,
            )
            oof_all.extend(oof_rows)
            print(f"  oof rows={len(oof_rows)}")

            calibrator = RiskCalibrator.fit(oof_rows)
            calibrator.save(out_dir / f"risk_calibrator_{representation}.json")
            print(
                f"  risk Spearman={calibrator.train_spearman:.3f} "
                f"monotone={monotone_ok(calibrator.bin_calibration)}"
            )
            risk_summary[representation] = {
                "n_oof": len(oof_rows),
                "spearman": calibrator.train_spearman,
                "monotone_ok": monotone_ok(calibrator.bin_calibration),
                "bin_calibration": calibrator.bin_calibration,
            }
        else:
            calibrator = RiskCalibrator.load(
                out_dir / f"risk_calibrator_{representation}.json"
            )

        # Kernel / kernel-hurdle comparison on acquisition-block holdout.
        blocks = sorted({r.acquisition_block for r in rep_rows})
        test_mask = np.asarray(
            [r.acquisition_block == blocks[-1] for r in rep_rows], dtype=bool
        )
        train_mask = ~test_mask
        models = {
            "kernel_nn": KernelNeighborBaseline.fit(x[train_mask], counts[train_mask]),
            "kernel_hurdle": KernelHurdleMultinomial.fit(
                x[train_mask],
                counts[train_mask],
                stay_feature_indices=stay_idx,
            ),
            "softmax_stump_boost": SoftmaxStumpBoost.fit(
                x[train_mask], counts[train_mask], n_estimators=25
            ),
        }
        for name, model in models.items():
            pred = model.predict_proba(x[test_mask])
            metrics = action_metrics(pred, counts[test_mask])
            stay = moments_abstention_activation_check(
                [rep_rows[i] for i, flag in enumerate(test_mask) if flag],
                pred,
                representation=representation,
            )
            model_compare_rows.append(
                {"representation": representation, "model": name, **metrics, **stay}
            )
            print(
                f"  {name}: log_loss={metrics['log_loss']:.3f} "
                f"stay_gap={stay.get('moments_stay_gap', float('nan'))}"
            )

        # Fit AD + full models for diagnosis / bank scoring.
        ad = ApplicabilityDomain.fit(x, representation=representation)
        full_models = {
            "kernel_nn": KernelNeighborBaseline.fit(x, counts),
            "kernel_hurdle": KernelHurdleMultinomial.fit(
                x, counts, stay_feature_indices=stay_idx
            ),
            "softmax_stump_boost": SoftmaxStumpBoost.fit(x, counts, n_estimators=25),
        }
        # Risk threshold = 80th percentile of OOF predicted risk on usable rows.
        if not args.skip_oof:
            oof_rep = [r for r in oof_all if r["representation"] == representation]
        else:
            # reload from file later
            oof_rep = []
        if oof_rep:
            r_hat = calibrator.predict_rows(oof_rep)
            risk_threshold = float(np.quantile(r_hat, 0.8))
        else:
            risk_threshold = 0.25
        risk_summary.setdefault(representation, {})["risk_threshold"] = risk_threshold

        diag_rows = []
        for n_agents in (9, 17):
            for regime in ("t0_only", "k0_drift", "closed_loop"):
                diag_rows.append(
                    risk_diagnose_regime(
                        ad=ad,
                        calibrator=calibrator,
                        models=full_models,
                        representation=representation,
                        n_agents=n_agents,
                        regime=regime,
                        n_steps=args.steps,
                        seed=100 + n_agents,
                        risk_threshold=risk_threshold,
                    )
                )
        write_csv(
            out_dir / f"risk_rediagnosis_{representation}.csv",
            diag_rows,
        )
        # stash for later JSON
        risk_summary[representation]["rediagnosis"] = diag_rows
        risk_summary[representation]["calibrator_path"] = str(
            out_dir / f"risk_calibrator_{representation}.json"
        )
        # Save AD peer set
        risk_summary[representation]["train_peers"] = list(DEFAULT_TRAIN_PEERS)

    if oof_all:
        write_csv(out_dir / "oof_residual_table.csv", oof_all)
    write_csv(out_dir / "kernel_hurdle_comparison.csv", model_compare_rows)

    # --- Rescore field bank (moments calibrator + AD as primary) ---
    bank_path = out_dir / "candidate_field_bank.npz"
    if not bank_path.exists():
        print("No candidate_field_bank.npz — skip reselection")
        (out_dir / "applicability_rebuild_summary.json").write_text(
            json.dumps(risk_summary, indent=2), encoding="utf-8"
        )
        return 0

    print("\nRescoring field bank with calibrated risk...")
    bank = np.load(bank_path, allow_pickle=True)
    # Use moments AD/risk for bank-level selection of physical fields.
    rep = "moments_m1_m3"
    x = packed[rep]["features"]
    counts = packed[rep]["counts"]
    ad = ApplicabilityDomain.fit(x, representation=rep)
    calibrator = RiskCalibrator.load(out_dir / f"risk_calibrator_{rep}.json")
    models = {
        "kernel_nn": KernelNeighborBaseline.fit(x, counts),
        "kernel_hurdle": KernelHurdleMultinomial.fit(
            x, counts, stay_feature_indices=stay_idx
        ),
        "softmax_stump_boost": SoftmaxStumpBoost.fit(x, counts, n_estimators=20),
    }

    # Bank stores common features only — rebuild minimal full vector for moments:
    # common ∥ native moments = common[:9 moments parts] duplicates native.
    # For moments, native is cos/sin of z1..z3 = common[0:6] interleaved? 
    # common: re1,im1,abs1,re2,im2,abs2,re3,im3,abs3,...
    # native moments: re1,im1,re2,im2,re3,im3
    common = bank["features_common"]
    native = common[:, [0, 1, 3, 4, 6, 7]]
    x_bank = np.concatenate([common, native], axis=1)
    # Restrict scoring to moments-labeled rows if present, else all.
    mask = bank["representation"].astype(str) == rep
    if not np.any(mask):
        mask = np.ones(common.shape[0], dtype=bool)
    idx = np.where(mask)[0]
    x_sub = x_bank[idx]
    unresolved = bank["unresolved"][idx]
    scores = ad.score(x_sub, unresolved_near_zero=unresolved)
    # Ensemble disagreement in chunks.
    disagree = np.zeros(x_sub.shape[0])
    chunk = 512
    for start in range(0, x_sub.shape[0], chunk):
        stop = min(start + chunk, x_sub.shape[0])
        preds = [m.predict_proba(x_sub[start:stop]) for m in models.values()]
        acc = np.zeros(stop - start)
        pairs = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                acc += 0.5 * np.sum(np.abs(preds[i] - preds[j]), axis=1)
                pairs += 1
        disagree[start:stop] = acc / max(pairs, 1)
    risk = calibrator.predict_arrays(
        d_shape=scores["d_shape"],
        d_native=scores["d_native"],
        local_density=scores["local_density"],
        u_ensemble=disagree,
        d_orientation=scores["d_orientation"],
        policy_a=scores["policy_a"],
    )
    # Map back to full-bank arrays (zeros elsewhere).
    n_bank = common.shape[0]
    risk_full = np.zeros(n_bank)
    dis_full = np.zeros(n_bank)
    d_shape_full = np.zeros(n_bank)
    g_peer_full = np.zeros(n_bank, dtype=bool)
    risk_full[idx] = risk
    dis_full[idx] = disagree
    d_shape_full[idx] = scores["d_shape"]
    g_peer_full[idx] = scores["g_peer"]

    # Occupancy among peer-matched moments rows.
    keys = [
        (
            int(np.clip(float(bank["abs_z1"][i]), 0, 1) * 10),
            int(bank["peer_count"][i]),
            str(bank["init_kind"][i]),
        )
        for i in idx
    ]
    occ = Counter(keys)
    max_occ = max(occ.values()) if occ else 1
    occ_full = np.zeros(n_bank)
    for local, global_i in enumerate(idx):
        occ_full[global_i] = occ[keys[local]] / max_occ

    # Priority for moments subset only; selection operates on full bank indices.
    # Downweight unsupported peers and pure peer-mismatch.
    priority = (
        1.0 * occ_full
        + 1.2 * (risk_full / max(float(risk_full.max()), 1e-12))
        + 1.0 * (dis_full / max(float(dis_full.max()), 1e-12))
        + 0.3 * (d_shape_full / max(float(d_shape_full.max()), 1e-12))
    )
    priority = np.where(g_peer_full, priority * 0.05, priority)
    priority = np.where(mask, priority, -1.0)

    bank_dict = {key: bank[key] for key in bank.files}
    bank_dict["disagreement"] = dis_full
    bank_dict["risk"] = risk_full
    bank_dict["features_canonical"] = bank["features_canonical"]
    selected = stratified_select(bank_dict, priority)
    print(f"  selected {len(selected)} fields")

    # Jaccard vs old selection.
    old_path = out_dir / "active_learning_selection.csv"
    old_ids = set()
    if old_path.exists():
        with old_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                old_ids.add(int(row["bank_index"]))
    new_ids = set(selected)
    inter = len(old_ids & new_ids)
    union = len(old_ids | new_ids) or 1
    jaccard = inter / union

    out_rows = []
    for rank, i in enumerate(selected):
        out_rows.append(
            {
                "rank": rank,
                "bank_index": int(i),
                "priority": float(priority[i]),
                "predicted_risk": float(risk_full[i]),
                "disagreement": float(dis_full[i]),
                "occupancy_n": float(occ_full[i]),
                "d_shape": float(d_shape_full[i]),
                "g_peer": int(g_peer_full[i]),
                "representation": str(bank["representation"][i]),
                "source": str(bank["source"][i]),
                "init_kind": str(bank["init_kind"][i]),
                "dynamics": str(bank["dynamics"][i]),
                "n_agents": int(bank["n_agents"][i]),
                "peer_count": int(bank["peer_count"][i]),
                "t": int(bank["t"][i]),
                "abs_z1": float(bank["abs_z1"][i]),
                "abs_z2": float(bank["abs_z2"][i]),
                "abs_z3": float(bank["abs_z3"][i]),
                "estimated_eps": float(bank["estimated_eps"][i]),
            }
        )
    write_csv(out_dir / "active_learning_selection_v2.csv", out_rows)

    # Source generator histogram for selected + whole bank.
    source_hist = Counter(str(s) for s in bank["source"])
    selected_source = Counter(r["source"] for r in out_rows)
    dynamics_hist = Counter(str(d) for d in bank["dynamics"])
    selected_dyn = Counter(r["dynamics"] for r in out_rows)

    selection_meta = {
        "n_select": len(out_rows),
        "jaccard_vs_v1": jaccard,
        "n_overlap_vs_v1": inter,
        "pilot_calls": 36 * 3 * 8 * 2,
        "bank_source_histogram": dict(source_hist),
        "selected_source_histogram": dict(selected_source),
        "bank_dynamics_histogram": dict(dynamics_hist),
        "selected_dynamics_histogram": dict(selected_dyn),
        "peer_count_histogram": dict(
            Counter(r["peer_count"] for r in out_rows)
        ),
        "allocation": {
            "high_occupancy_high_risk": 18,
            "high_disagreement": 8,
            "k0_drift": 6,
            "maximin_boundary": 4,
        },
        "note": "Physical fields to acquire once; all 3 encodings share the same fields.",
    }
    (out_dir / "active_learning_selection_v2_meta.json").write_text(
        json.dumps(selection_meta, indent=2), encoding="utf-8"
    )
    risk_summary["selection_v2"] = selection_meta
    (out_dir / "applicability_rebuild_summary.json").write_text(
        json.dumps(risk_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection_meta, indent=2))
    print("Done. Paid pilot NOT launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
