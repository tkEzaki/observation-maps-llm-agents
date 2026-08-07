"""CENT-3 prospective evaluation BEFORE any refit on pilot data.

Uses locked R_pre / p_hat_pre from cent3_prospective_lock_v0.json.
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

from analysis.stage3.dataset import load_run_counts  # noqa: E402
from analysis.stage3.models import counts_to_probs  # noqa: E402
from analysis.stage3.risk_calibrator import monotone_ok  # noqa: E402
from analysis.stage3.validation import tv_distance  # noqa: E402
from circlemap.field_features import extract_field_descriptors  # noqa: E402
from circlemap.stimuli import (  # noqa: E402
    STIMULUS_CATALOG,
    StimulusSpec,
    build_stimulus_histogram,
)

REP = "centers_24_standard"
BRANCH = ROOT / "analysis" / "centers_branch"


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
        data.pop("shift_bins", None)
        if "fixed_fractions" in data:
            data["fixed_fractions"] = tuple(float(v) for v in data["fixed_fractions"])
        STIMULUS_CATALOG[str(name)] = StimulusSpec(**data)  # type: ignore[arg-type]


def _latest_cent3_runs() -> list[Path]:
    root = ROOT / "runs" / "centers_cent3"
    runs: list[Path] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        if "centers-cent3-v0-block_" not in experiment_dir.name:
            continue
        stamped = [
            path
            for path in experiment_dir.iterdir()
            if path.is_dir() and (path / "trace.jsonl").exists()
        ]
        if stamped:
            runs.append(max(stamped, key=lambda path: path.name))
    return sorted(runs)


def _enrichment(r: np.ndarray, e: np.ndarray) -> dict[str, float]:
    order = np.argsort(r)
    n = int(r.size)
    q = max(1, n // 5)
    low = e[order[:q]]
    high = e[order[-q:]]
    mean_low = float(np.mean(low))
    mean_high = float(np.mean(high))
    return {
        "n_top_bottom": float(q),
        "mean_e_tv_low_R": mean_low,
        "mean_e_tv_high_R": mean_high,
        "enrichment_ratio": mean_high / mean_low if mean_low > 1e-12 else float("nan"),
    }


def _complex_moments(p: np.ndarray) -> tuple[float, float, float]:
    """Return (C1, C2, a0) from action probs [-1,0,+1] as first-harmonic proxy.

    Here a0 = E[f] = p+ - p-, C1 = activity-weighted polarity proxy = |E[f]|
    on the action simplex; C2 unused for trinomial — use stay as second channel.
    """
    p_m, p0, p_p = float(p[0]), float(p[1]), float(p[2])
    a0 = p_p - p_m
    activity = p_p + p_m
    return activity, a0, p0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir

    _load_catalog(branch / "cent3_stimulus_catalog_v0.json")
    lock = json.loads((branch / "cent3_prospective_lock_v0.json").read_text(encoding="utf-8"))
    lock_by_id = {row["field_id"]: row for row in lock["rows"]}
    fields = json.loads((branch / "cent3_frozen_fields_v0.json").read_text(encoding="utf-8"))[
        "fields"
    ]

    run_dirs = _latest_cent3_runs()
    if len(run_dirs) < 2:
        raise SystemExit(f"need 2 cent3 runs, found {run_dirs}")
    print("Runs:")
    for path in run_dirs:
        print(f"  {path}")

    # Per-block and pooled counts
    block_counts: list[dict[str, np.ndarray]] = []
    for run_dir in run_dirs:
        _protocol, counts = load_run_counts(run_dir)
        by_profile = counts[REP]
        block_counts.append(
            {pid: np.asarray(cell[0], dtype=np.float64) for pid, cell in by_profile.items()}
        )
    pooled: dict[str, np.ndarray] = {}
    for block in block_counts:
        for pid, vec in block.items():
            pooled[pid] = pooled.get(pid, np.zeros(3)) + vec

    detail = []
    r_pre_list = []
    e_tv_list = []
    act_err = []
    a0_err = []
    stay_err = []
    block_tvs = []

    for field in fields:
        fid = field["field_id"]
        locked = lock_by_id[fid]
        p_hat = np.asarray(
            [locked["p_minus_pre"], locked["p_stay_pre"], locked["p_plus_pre"]],
            dtype=np.float64,
        )
        r_pre = float(locked["R_pre"])
        counts = pooled[fid]
        p_obs = counts_to_probs(counts[None, :], alpha=0.0)[0]
        e_tv = float(tv_distance(p_hat[None, :], p_obs[None, :])[0])
        act_hat, a0_hat, stay_hat = _complex_moments(p_hat)
        act_obs, a0_obs, stay_obs = _complex_moments(p_obs)
        # Block replication TV between empirical block distributions
        p_b1 = counts_to_probs(block_counts[0][fid][None, :], alpha=0.5)[0]
        p_b2 = counts_to_probs(block_counts[1][fid][None, :], alpha=0.5)[0]
        b_tv = float(tv_distance(p_b1[None, :], p_b2[None, :])[0])
        n_valid = int(np.sum(counts))
        detail.append(
            {
                "field_id": fid,
                "bucket": field["bucket"],
                "role": field["role"],
                "n_valid": n_valid,
                "R_pre": r_pre,
                "e_tv": e_tv,
                "p_minus_pre": float(p_hat[0]),
                "p_stay_pre": float(p_hat[1]),
                "p_plus_pre": float(p_hat[2]),
                "p_minus_obs": float(p_obs[0]),
                "p_stay_obs": float(p_obs[1]),
                "p_plus_obs": float(p_obs[2]),
                "activity_abs_err": abs(act_hat - act_obs),
                "a0_abs_err": abs(a0_hat - a0_obs),
                "stay_abs_err": abs(stay_hat - stay_obs),
                "block_tv": b_tv,
            }
        )
        r_pre_list.append(r_pre)
        e_tv_list.append(e_tv)
        act_err.append(abs(act_hat - act_obs))
        a0_err.append(abs(a0_hat - a0_obs))
        stay_err.append(abs(stay_hat - stay_obs))
        block_tvs.append(b_tv)

    r_arr = np.asarray(r_pre_list, dtype=np.float64)
    e_arr = np.asarray(e_tv_list, dtype=np.float64)
    spearman = _spearman(r_arr, e_arr)
    enrich = _enrichment(r_arr, e_arr)
    order = np.argsort(r_arr)
    n_bins = 4
    edges = np.linspace(0, len(order), n_bins + 1, dtype=int)
    bins = []
    for left, right in zip(edges[:-1], edges[1:]):
        idx = order[left:right]
        bins.append(
            {
                "mean_R": float(np.mean(r_arr[idx])),
                "mean_error": float(np.mean(e_arr[idx])),
                "n": int(right - left),
            }
        )
    mono = monotone_ok(bins, tol=0.05)

    # Phenotype controls
    ctrl = {d["field_id"]: d for d in detail if d["bucket"] == "control"}
    phenotype = {
        "exact_antipodal_stay_obs": ctrl["cent_ctrl_exact_antipodal"]["p_stay_obs"],
        "small_imbalance_stay_obs": ctrl["cent_ctrl_small_imbalance"]["p_stay_obs"],
        "unimodal_a0_obs": (
            ctrl["cent_ctrl_unimodal_positive_polar"]["p_plus_obs"]
            - ctrl["cent_ctrl_unimodal_positive_polar"]["p_minus_obs"]
        ),
        "sign_reversed_a0_obs": (
            ctrl["cent_ctrl_unimodal_sign_reversed"]["p_plus_obs"]
            - ctrl["cent_ctrl_unimodal_sign_reversed"]["p_minus_obs"]
        ),
        "note": "centers exact-balance stay not expected near 1.0",
    }
    polarity_flip = (
        phenotype["unimodal_a0_obs"] * phenotype["sign_reversed_a0_obs"] < 0
    )

    costs = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        costs.append(summary)

    gates = {
        "risk_spearman_positive": bool(spearman > 0.0),
        "high_risk_enrichment_gt_1_0": bool(enrich["enrichment_ratio"] > 1.0),
        "risk_bin_monotone_or_spearman_gt_0_15": bool(mono or spearman > 0.15),
        "min_valid_per_field_ge_12": bool(all(d["n_valid"] >= 12 for d in detail)),
        "polarity_sign_flip_on_controls": bool(polarity_flip),
    }
    # Freeze recommendation uses centers-calibrated soft gate, not moments copy.
    recommend_freeze = bool(
        gates["risk_spearman_positive"]
        and gates["high_risk_enrichment_gt_1_0"]
        and gates["min_valid_per_field_ge_12"]
    )

    out = {
        "status": "prospective_eval_before_refit",
        "representation": REP,
        "n_fields": len(detail),
        "run_dirs": [str(p) for p in run_dirs],
        "actual_cost_usd_total": float(sum(c["actual_cost_usd"] for c in costs)),
        "valid_rates": [c["valid_rate"] for c in costs],
        "risk": {
            "spearman_R_pre_vs_e_tv": spearman,
            "enrichment": enrich,
            "bin_calibration": bins,
            "monotone_ok": mono,
        },
        "errors": {
            "mean_e_tv": float(np.mean(e_arr)),
            "mean_activity_abs_err": float(np.mean(act_err)),
            "mean_a0_abs_err": float(np.mean(a0_err)),
            "mean_stay_abs_err": float(np.mean(stay_err)),
            "mean_block_tv": float(np.mean(block_tvs)),
        },
        "phenotype_controls": phenotype,
        "gates": gates,
        "recommend_proceed_toward_freeze": recommend_freeze,
        "note": (
            "Prospective only — no refit. Full centers_bundle_v1 freeze still "
            "requires CENT-4 eight-gate checklist."
        ),
    }

    detail_path = branch / "cent3_prospective_detail.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)
    summary_path = branch / "cent3_prospective_summary.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if recommend_freeze else 2


if __name__ == "__main__":
    raise SystemExit(main())
