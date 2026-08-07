"""Prospective Stage 3A pilot evaluation BEFORE any refit on pilot data.

Uses frozen risk calibrators + models fit only on pre-pilot Stage B sources.
"""

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
from analysis.stage3.dataset import (  # noqa: E402
    ACTION_INDEX,
    build_unified_dataset,
    load_run_counts,
)
from analysis.stage3.models import (  # noqa: E402
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    SoftmaxStumpBoost,
    counts_to_probs,
)
from analysis.stage3.oof_residuals import default_stay_indices  # noqa: E402
from analysis.stage3.rebuild_applicability import risk_diagnose_regime  # noqa: E402
from analysis.stage3.risk_calibrator import (  # noqa: E402
    RiskCalibrator,
    monotone_ok,
)
from analysis.stage3.validation import tv_distance  # noqa: E402
from circlemap.field_features import extract_field_descriptors  # noqa: E402
from circlemap.stimuli import (  # noqa: E402
    STIMULUS_CATALOG,
    StimulusSpec,
    build_stimulus_histogram,
)

REPS = (
    "intervals_24_decimal6",
    "moments_m1_m3",
    "centers_24_standard",
)
PRIMARY_MODEL = "kernel_nn"


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _load_catalog(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, data in payload.items():
        data = dict(data)
        data.pop("id", None)
        data.pop("derived_from_catalog", None)
        for key in ("mode_offsets", "weights", "fixed_fractions"):
            if key in data:
                data[key] = tuple(float(v) for v in data[key])
        if "mode_counts" in data:
            data["mode_counts"] = tuple(int(v) for v in data["mode_counts"])
        STIMULUS_CATALOG[str(name)] = StimulusSpec(**data)  # type: ignore[arg-type]


def _latest_pilot_runs(project_root: Path) -> list[Path]:
    root = project_root / "runs" / "response_law_representation"
    runs: list[Path] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if "stage3a-pilot-v1" not in experiment_dir.name:
            continue
        stamped = [
            path
            for path in experiment_dir.iterdir()
            if path.is_dir() and (path / "trace.jsonl").exists()
        ]
        if stamped:
            runs.append(max(stamped, key=lambda path: path.name))
    return runs


def _pool_pilot_counts(run_dirs: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    pooled: dict[str, dict[str, np.ndarray]] = {}
    for run_dir in run_dirs:
        _protocol, counts = load_run_counts(run_dir)
        for representation, by_profile in counts.items():
            bucket = pooled.setdefault(representation, {})
            for profile, cell in by_profile.items():
                # Pilot uses a single offset; take offset 0 and sum across blocks.
                vec = np.asarray(cell[0], dtype=np.float64)
                if profile not in bucket:
                    bucket[profile] = vec.copy()
                else:
                    bucket[profile] += vec
    return pooled


def _ensemble_disagreement(preds: list[np.ndarray]) -> np.ndarray:
    disagree = np.zeros(preds[0].shape[0], dtype=np.float64)
    pairs = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            disagree += 0.5 * np.sum(np.abs(preds[i] - preds[j]), axis=1)
            pairs += 1
    return disagree / max(pairs, 1)


def _enrichment(r: np.ndarray, e: np.ndarray) -> dict[str, float]:
    order = np.argsort(r)
    n = int(r.size)
    q = max(1, n // 4)
    low = e[order[:q]]
    high = e[order[-q:]]
    mean_low = float(np.mean(low))
    mean_high = float(np.mean(high))
    return {
        "n_quartile": float(q),
        "mean_e_tv_low_R": mean_low,
        "mean_e_tv_high_R": mean_high,
        "enrichment_ratio": (
            mean_high / mean_low if mean_low > 1e-12 else float("nan")
        ),
        "enrichment_delta": mean_high - mean_low,
    }


def _bin_calibration(r: np.ndarray, e: np.ndarray, n_bins: int = 4) -> list[dict]:
    order = np.argsort(r)
    s = r[order]
    err = e[order]
    edges = np.linspace(0, s.size, n_bins + 1, dtype=int)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        rows.append(
            {
                "mean_R": float(np.mean(s[left:right])),
                "mean_e_tv": float(np.mean(err[left:right])),
                "n": int(right - left),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument("--primary-model", default=PRIMARY_MODEL)
    parser.add_argument("--qrisk-steps", type=int, default=20)
    args = parser.parse_args()
    out_dir = args.out_dir

    frozen_path = out_dir / "pilot_frozen_fields_v1.json"
    catalog_path = out_dir / "pilot_stimulus_catalog_v1.json"
    _load_catalog(catalog_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    fields = frozen["fields"]
    field_meta = {f["field_id"]: f for f in fields}

    print("Loading pre-pilot training dataset...")
    all_rows, packed = build_unified_dataset(ROOT)
    stay_idx = default_stay_indices()

    run_dirs = _latest_pilot_runs(ROOT)
    if len(run_dirs) < 2:
        raise SystemExit(f"expected 2 pilot runs, found {len(run_dirs)}: {run_dirs}")
    print("Pilot runs:")
    for path in run_dirs:
        print(f"  {path}")
    pooled = _pool_pilot_counts(run_dirs)

    detail_rows: list[dict] = []
    summary: dict[str, dict] = {}
    qrisk_rows: list[dict] = []

    for representation in REPS:
        print(f"\n=== {representation} ===")
        x_train = packed[representation]["features"]
        counts_train = packed[representation]["counts"]
        ad = ApplicabilityDomain.fit(x_train, representation=representation)
        calibrator = RiskCalibrator.load(
            out_dir / f"risk_calibrator_{representation}.json"
        )
        models = {
            "kernel_nn": KernelNeighborBaseline.fit(x_train, counts_train),
            "kernel_hurdle": KernelHurdleMultinomial.fit(
                x_train, counts_train, stay_feature_indices=stay_idx
            ),
            "softmax_stump_boost": SoftmaxStumpBoost.fit(
                x_train, counts_train, n_estimators=25
            ),
        }
        # Risk threshold from frozen OOF residuals (pre-pilot).
        oof_csv = out_dir / "oof_residual_table.csv"
        oof_rep_rows = []
        with oof_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["representation"] == representation:
                    oof_rep_rows.append(row)
        if oof_rep_rows:
            r_oof = calibrator.predict_rows(oof_rep_rows)
            risk_threshold = float(np.quantile(r_oof, 0.8))
        else:
            risk_threshold = 0.25

        profiles = sorted(pooled[representation].keys())
        xs = []
        unresolved = []
        counts_obs = []
        for profile in profiles:
            histogram = build_stimulus_histogram(profile, 0.0)
            desc = extract_field_descriptors(representation, histogram)
            xs.append(np.concatenate([desc.common, desc.native]))
            unresolved.append(bool(desc.unresolved_near_zero))
            counts_obs.append(pooled[representation][profile])
        x = np.asarray(xs, dtype=np.float64)
        unresolved_arr = np.asarray(unresolved, dtype=bool)
        counts_obs_arr = np.asarray(counts_obs, dtype=np.float64)
        p_obs = counts_to_probs(counts_obs_arr, alpha=0.0)

        preds = {name: model.predict_proba(x) for name, model in models.items()}
        p_hat = preds[args.primary_model]
        e_tv = tv_distance(p_hat, p_obs)
        scores = ad.score(x, unresolved_near_zero=unresolved_arr)
        u_ens = _ensemble_disagreement(list(preds.values()))
        r_pre = calibrator.predict_arrays(
            d_shape=scores["d_shape"],
            d_native=scores["d_native"],
            local_density=scores["local_density"],
            u_ensemble=u_ens,
            d_orientation=scores["d_orientation"],
            policy_a=scores["policy_a"],
        )
        # Frozen bank risk (moments-scored at selection) for reference.
        r_bank = np.asarray(
            [
                float(field_meta[p].get("predicted_risk") or np.nan)
                for p in profiles
            ],
            dtype=np.float64,
        )

        spearman = _spearman(r_pre, e_tv)
        enrich = _enrichment(r_pre, e_tv)
        bins = _bin_calibration(r_pre, e_tv, n_bins=4)
        mono = monotone_ok(
            [{"mean_R": b["mean_R"], "mean_error": b["mean_e_tv"], "n": b["n"]} for b in bins],
            tol=0.02,
        )

        # Moments stay / direction on control fields (observed + predicted).
        control_ids = {
            "exact": "control_exact_antipodal_8_8_r00",
            "small": "control_small_imbalance_9_7_r00",
        }
        stay_obs = {}
        stay_hat = {}
        for key, fid in control_ids.items():
            if fid not in profiles:
                stay_obs[key] = float("nan")
                stay_hat[key] = float("nan")
                continue
            idx = profiles.index(fid)
            stay_obs[key] = float(p_obs[idx, ACTION_INDEX["stay"]])
            stay_hat[key] = float(p_hat[idx, ACTION_INDEX["stay"]])
        stay_gap_obs = stay_obs["exact"] - stay_obs["small"]
        stay_gap_hat = stay_hat["exact"] - stay_hat["small"]
        direction_ok = bool(
            np.isfinite(stay_gap_obs)
            and np.isfinite(stay_gap_hat)
            and stay_gap_obs > 0
            and stay_gap_hat > 0
        )

        for i, profile in enumerate(profiles):
            meta = field_meta[profile]
            detail_rows.append(
                {
                    "representation": representation,
                    "field_id": profile,
                    "role": meta.get("role", ""),
                    "bucket": meta.get("bucket", ""),
                    "peer_count": int(meta["stimulus"]["peer_count"]),
                    "n_obs": int(np.sum(counts_obs_arr[i])),
                    "p_retard_obs": float(p_obs[i, 0]),
                    "p_stay_obs": float(p_obs[i, 1]),
                    "p_advance_obs": float(p_obs[i, 2]),
                    "p_retard_hat": float(p_hat[i, 0]),
                    "p_stay_hat": float(p_hat[i, 1]),
                    "p_advance_hat": float(p_hat[i, 2]),
                    "e_tv": float(e_tv[i]),
                    "R_pre": float(r_pre[i]),
                    "R_bank_moments": float(r_bank[i]),
                    "d_shape": float(scores["d_shape"][i]),
                    "d_native": float(scores["d_native"][i]),
                    "u_ensemble": float(u_ens[i]),
                    "g_peer": int(bool(scores["g_peer"][i])),
                    "policy_a": float(scores["policy_a"][i]),
                }
            )

        gate_spearman = bool(np.isfinite(spearman) and spearman > 0.0)
        gate_enrich = bool(
            np.isfinite(enrich["enrichment_delta"]) and enrich["enrichment_delta"] > 0.0
        )
        summary[representation] = {
            "n_fields": len(profiles),
            "primary_model": args.primary_model,
            "mean_e_tv": float(np.mean(e_tv)),
            "median_e_tv": float(np.median(e_tv)),
            "spearman_R_pre_e_tv": spearman,
            "spearman_R_bank_e_tv": _spearman(r_bank, e_tv),
            "monotone_ok_4bin": mono,
            "bin_calibration": bins,
            "enrichment": enrich,
            "moments_stay_exact_obs": stay_obs["exact"],
            "moments_stay_small_obs": stay_obs["small"],
            "moments_stay_gap_obs": float(stay_gap_obs),
            "moments_stay_exact_hat": stay_hat["exact"],
            "moments_stay_small_hat": stay_hat["small"],
            "moments_stay_gap_hat": float(stay_gap_hat),
            "moments_direction_ok": direction_ok,
            "risk_threshold_oof80": risk_threshold,
            "gates": {
                "risk_spearman_positive": gate_spearman,
                "high_risk_error_enrichment": gate_enrich,
                "moments_stay_direction_ok": (
                    direction_ok if representation == "moments_m1_m3" else None
                ),
            },
        }
        print(
            f"  spearman(R_pre,e_tv)={spearman:.3f} "
            f"enrich_delta={enrich['enrichment_delta']:.3f} "
            f"mean_e_tv={float(np.mean(e_tv)):.3f} "
            f"stay_gap_obs={stay_gap_obs:.3f} stay_gap_hat={stay_gap_hat:.3f}"
        )

        # Peer-matched collective Q_risk scan (still pre-refit models).
        for n_agents in (9, 17):
            for regime in ("t0_only", "k0_drift", "closed_loop"):
                row = risk_diagnose_regime(
                    ad=ad,
                    calibrator=calibrator,
                    models=models,
                    representation=representation,
                    n_agents=n_agents,
                    regime=regime,
                    n_steps=args.qrisk_steps,
                    seed=100 + n_agents,
                    risk_threshold=risk_threshold,
                )
                qrisk_rows.append(row)
                print(
                    f"  Q_risk N={n_agents} {regime}: "
                    f"mean_Q={row['mean_Q_risk']:.3f} "
                    f"mean_R={row['mean_R']:.3f}"
                )

    detail_path = out_dir / "pilot_prospective_detail.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    qrisk_out = []
    for row in qrisk_rows:
        qrisk_out.append(
            {
                "representation": row["representation"],
                "n_agents": row["n_agents"],
                "peer_count": row["peer_count"],
                "regime": row["regime"],
                "mean_Q_risk": row["mean_Q_risk"],
                "mean_q_high_risk": row["mean_q_high_risk"],
                "final_q_high_risk": row["final_q_high_risk"],
                "mean_R": row["mean_R"],
                "mean_q_unsupported_peer": row["mean_q_unsupported_peer"],
            }
        )
    qrisk_path = out_dir / "pilot_prospective_qrisk.csv"
    with qrisk_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qrisk_out[0].keys()))
        writer.writeheader()
        writer.writerows(qrisk_out)

    # Decision rollup
    moments = summary["moments_m1_m3"]
    all_spearman_ok = all(s["gates"]["risk_spearman_positive"] for s in summary.values())
    all_enrich_ok = all(s["gates"]["high_risk_error_enrichment"] for s in summary.values())
    moments_ok = bool(moments["gates"]["moments_stay_direction_ok"])
    # Soft Q_risk review: mean closed-loop occ-weighted Q_risk at N=9,17
    closed = [
        r
        for r in qrisk_out
        if r["regime"] == "closed_loop" and r["n_agents"] in (9, 17)
    ]
    mean_closed_q = float(np.mean([r["mean_Q_risk"] for r in closed]))
    decision = {
        "status": "prospective_eval_complete_pre_refit",
        "pilot_runs": [str(p) for p in run_dirs],
        "primary_model": args.primary_model,
        "per_representation": summary,
        "gates": {
            "risk_spearman_positive_all_reps": all_spearman_ok,
            "high_risk_error_enrichment_all_reps": all_enrich_ok,
            "moments_stay_direction_ok": moments_ok,
            "N9_N17_Qrisk_reviewed": True,
            "mean_closed_loop_Q_risk_N9_N17": mean_closed_q,
        },
        "stage_3b_freeze": False,
        "notes": (
            "Prospective only — do not refit before reading these metrics. "
            "Partial freeze allowed per protocol."
        ),
        "artifact_paths": {
            "detail": str(detail_path.relative_to(ROOT).as_posix()),
            "qrisk": str(qrisk_path.relative_to(ROOT).as_posix()),
        },
    }
    # Provisional GO if ranking + enrichment + moments direction hold.
    decision["prospective_go"] = bool(
        all_spearman_ok and all_enrich_ok and moments_ok
    )
    decision_path = out_dir / "pilot_prospective_summary.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\n" + json.dumps({
        "prospective_go": decision["prospective_go"],
        "gates": decision["gates"],
        "spearman": {
            rep: summary[rep]["spearman_R_pre_e_tv"] for rep in REPS
        },
        "enrichment_delta": {
            rep: summary[rep]["enrichment"]["enrichment_delta"] for rep in REPS
        },
        "mean_closed_Q_risk": mean_closed_q,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
