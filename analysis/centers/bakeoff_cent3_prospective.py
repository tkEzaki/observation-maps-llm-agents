"""CENT-4B: pre-refit prospective bake-off of CENT-1 candidates on CENT-3 fields.

Fits all candidates on the frozen pre-CENT-3 training snapshot only
(`training_arrays_centers_v0.npz`). Does not include CENT-3 rows in fit.
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

from analysis.centers.cent3_io import (  # noqa: E402
    BRANCH,
    REP,
    action_channels,
    field_feature_row,
    latest_cent3_runs,
    load_catalog,
    pooled_and_block_counts,
)
from analysis.centers.models import fit_centers_candidates  # noqa: E402
from analysis.stage3.models import counts_to_probs  # noqa: E402
from analysis.stage3.validation import (  # noqa: E402
    multinomial_log_loss,
    tv_distance,
)


def _metrics(pred: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    empir = counts_to_probs(counts[None, :], alpha=0.0)[0]
    p = pred
    ch_p = action_channels(p)
    ch_o = action_channels(empir)
    return {
        "log_loss": multinomial_log_loss(p[None, :], counts[None, :]),
        "d_TV": float(tv_distance(p[None, :], empir[None, :])[0]),
        "abs_e_A": abs(ch_p["A"] - ch_o["A"]),
        "abs_e_a0": abs(ch_p["a0"] - ch_o["a0"]),
        "abs_e_C1": abs(ch_p["C1"] - ch_o["C1"]),
        "abs_e_C2": abs(ch_p["C2"] - ch_o["C2"]),
        "n_valid": float(np.sum(counts)),
    }


def _mean_metrics(rows: list[dict], keys: list[str]) -> dict[str, float]:
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir

    load_catalog(branch / "cent3_stimulus_catalog_v0.json")
    fields = json.loads((branch / "cent3_frozen_fields_v0.json").read_text(encoding="utf-8"))[
        "fields"
    ]
    lock = json.loads((branch / "cent3_prospective_lock_v0.json").read_text(encoding="utf-8"))
    lock_by = {row["field_id"]: row for row in lock["rows"]}

    arrays = np.load(branch / "training_arrays_centers_v0.npz", allow_pickle=True)
    x_train = arrays["features"]
    counts_train = arrays["counts"]
    print(f"Fitting CENT-1 candidates on pre-CENT-3 train n={x_train.shape[0]} ...")
    models = fit_centers_candidates(x_train, counts_train)

    # Prefer saved production hurdle weights if present (bit-identical to CENT-3 lock).
    hurdle_dir = branch / "production_model_activity_direction_hurdle"
    if hurdle_dir.exists():
        from analysis.stage3.models import KernelHurdleMultinomial

        models["activity_direction_hurdle"] = KernelHurdleMultinomial.load(hurdle_dir)
        print("Loaded production_model_activity_direction_hurdle from disk")

    run_dirs = latest_cent3_runs()
    pooled, _block = pooled_and_block_counts(run_dirs)

    # Build feature matrix for the 36 fields
    xs = []
    meta = []
    for field in fields:
        fid = field["field_id"]
        x, unresolved, peer, sparsity = field_feature_row(field)
        xs.append(x)
        meta.append(
            {
                "field_id": fid,
                "bucket": field["bucket"],
                "role": field["role"],
                "peer_count": peer,
                "bin_sparsity": sparsity,
                "unresolved_near_zero": int(unresolved),
                "R_pre": float(lock_by[fid]["R_pre"]),
            }
        )
    x_test = np.asarray(xs, dtype=np.float64)

    detail_rows = []
    metric_keys = [
        "log_loss",
        "d_TV",
        "abs_e_A",
        "abs_e_a0",
        "abs_e_C1",
        "abs_e_C2",
    ]
    for name, model in models.items():
        preds = model.predict_proba(x_test)
        for i, field_meta in enumerate(meta):
            fid = field_meta["field_id"]
            counts = pooled[fid]
            m = _metrics(preds[i], counts)
            ch_hat = action_channels(preds[i])
            ch_obs = action_channels(counts_to_probs(counts[None, :], alpha=0.0)[0])
            detail_rows.append(
                {
                    "model": name,
                    **field_meta,
                    **{k: m[k] for k in metric_keys},
                    "n_valid": m["n_valid"],
                    "p_minus_hat": ch_hat["p_minus"],
                    "p_stay_hat": ch_hat["p_stay"],
                    "p_plus_hat": ch_hat["p_plus"],
                    "p_minus_obs": ch_obs["p_minus"],
                    "p_stay_obs": ch_obs["p_stay"],
                    "p_plus_obs": ch_obs["p_plus"],
                    "A_hat": ch_hat["A"],
                    "A_obs": ch_obs["A"],
                    "a0_hat": ch_hat["a0"],
                    "a0_obs": ch_obs["a0"],
                }
            )

    # Summaries
    summaries = []
    for name in models:
        rows = [r for r in detail_rows if r["model"] == name]
        overall = _mean_metrics(rows, metric_keys)
        by_peer = {}
        for peer in (8, 16):
            sub = [r for r in rows if int(r["peer_count"]) == peer]
            by_peer[f"peer_{peer}"] = {
                "n_fields": len(sub),
                **_mean_metrics(sub, metric_keys),
            }
        # Activity vs direction split diagnostic
        act_ok_dir_bad = [
            r
            for r in rows
            if r["abs_e_A"] <= 0.15 and r["abs_e_a0"] >= 0.5
        ]
        summaries.append(
            {
                "model": name,
                "n_fields": len(rows),
                "overall": overall,
                "by_peer": by_peer,
                "n_activity_ok_direction_bad": len(act_ok_dir_bad),
                "fraction_activity_ok_direction_bad": len(act_ok_dir_bad) / max(len(rows), 1),
            }
        )

    ranked = sorted(summaries, key=lambda s: s["overall"]["log_loss"])
    best = ranked[0]["model"]
    hurdle = next(s for s in summaries if s["model"] == "activity_direction_hurdle")
    best_ll = ranked[0]["overall"]["log_loss"]
    hurdle_ll = hurdle["overall"]["log_loss"]
    ll_spread = ranked[-1]["overall"]["log_loss"] - ranked[0]["overall"]["log_loss"]

    # Branch interpretation
    peer8 = [s["by_peer"]["peer_8"]["log_loss"] for s in summaries]
    peer16 = [s["by_peer"]["peer_16"]["log_loss"] for s in summaries]
    peer_gap = float(np.mean(np.abs(np.asarray(peer8) - np.asarray(peer16))))

    native_names = {
        "hybrid_circular_emd",
        "hybrid_hellinger",
        "kernel_nn_euclidean",
    }
    native_best = min(
        (s for s in summaries if s["model"] in native_names),
        key=lambda s: s["overall"]["log_loss"],
    )
    native_beats_hurdle = native_best["overall"]["log_loss"] + 1e-6 < hurdle_ll

    direction_frac = float(
        np.mean([s["fraction_activity_ok_direction_bad"] for s in summaries])
    )

    structured = [s for s in summaries if s["model"] != "global_empirical"]
    structured_best = min(structured, key=lambda s: s["overall"]["log_loss"])
    global_row = next(s for s in summaries if s["model"] == "global_empirical")
    global_beats_structured = (
        global_row["overall"]["log_loss"] + 1e-6 < structured_best["overall"]["log_loss"]
    )
    all_tv_high = all(s["overall"]["d_TV"] > 0.35 for s in structured)

    if global_beats_structured and all_tv_high:
        branch_read = "coverage_or_feature_insufficiency"
    elif native_beats_hurdle and (hurdle_ll - native_best["overall"]["log_loss"]) > 0.02:
        branch_read = "model_selection_failure"
    elif direction_frac > 0.25 and hurdle["overall"]["abs_e_a0"] > 0.5:
        branch_read = "direction_component_structural_gap"
    elif peer_gap > 0.15:
        branch_read = "peer_specific_experts_indicated"
    else:
        branch_read = "mixed_revise_family_and_coverage"

    out = {
        "status": "cent4b_pre_refit_prospective_bakeoff",
        "representation": REP,
        "fit_snapshot": "training_arrays_centers_v0.npz",
        "n_train_rows": int(x_train.shape[0]),
        "n_test_fields": len(fields),
        "run_dirs": [str(p) for p in run_dirs],
        "channel_definitions": {
            "A": "p_plus + p_minus",
            "a0": "p_plus - p_minus",
            "C1": "|a0|",
            "C2": "p_stay",
        },
        "ranking_by_log_loss": [
            {
                "model": s["model"],
                **s["overall"],
                "n_activity_ok_direction_bad": s["n_activity_ok_direction_bad"],
            }
            for s in ranked
        ],
        "by_model": summaries,
        "interpretation": {
            "best_model": best,
            "best_structured_model": structured_best["model"],
            "hurdle_log_loss": hurdle_ll,
            "best_log_loss": best_ll,
            "best_structured_log_loss": structured_best["overall"]["log_loss"],
            "log_loss_spread_max_minus_min": ll_spread,
            "global_beats_structured": global_beats_structured,
            "native_or_kernel_best": native_best["model"],
            "native_beats_hurdle": native_beats_hurdle,
            "mean_abs_peer8_minus_peer16_log_loss": peer_gap,
            "mean_fraction_activity_ok_direction_bad": direction_frac,
            "branch_read": branch_read,
            "secondary_reads": {
                "hurdle_worse_than_native_kernels": native_beats_hurdle,
                "direction_errors_large": bool(hurdle["overall"]["abs_e_a0"] > 0.5),
                "peer8_vs_peer16_gap_large": bool(peer_gap > 0.15),
            },
        },
        "decision": {
            "centers_bundle_v1_freeze": "NO-GO",
            "centers_collective_stage_c": "NO-GO",
            "next": (
                "revise centers model family offline; do not refit on CENT-3 "
                "as freeze evidence; plan confirmatory pilot after revision"
            ),
        },
    }

    with (branch / "cent4b_bakeoff_detail.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    (branch / "cent4b_bakeoff_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "ranking": out["ranking_by_log_loss"],
        "interpretation": out["interpretation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
