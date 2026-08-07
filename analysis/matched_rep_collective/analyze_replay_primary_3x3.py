"""Primary 3×3 replay analysis (field-blocked; before any surrogate)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
)


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def _dist(rows: list[dict]) -> tuple[np.ndarray, float, float]:
    """Return (p-,p0,p+), activity A, signed mean a0 from valid rows."""
    vals = [int(r["action_value"]) for r in rows if r.get("valid") and r.get("action_value") is not None]
    if not vals:
        return np.array([np.nan, np.nan, np.nan]), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    p_m = float(np.mean(arr == -1))
    p_0 = float(np.mean(arr == 0))
    p_p = float(np.mean(arr == 1))
    a0 = float(np.mean(arr))
    activity = float(np.mean(arr != 0))
    return np.array([p_m, p_0, p_p]), activity, a0


def run_primary_report(run_dir: Path, fields: list[dict]) -> Path:
    trace_path = run_dir / "trace.jsonl"
    rows = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    by_ft: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_ft_block: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["field_id"], r["target_representation"])
        by_ft[key].append(r)
        by_ft_block[(r["field_id"], r["target_representation"], int(r["acquisition_block"]))].append(r)

    field_meta = {f["field_id"]: f for f in fields}
    field_rows = []
    for field in fields:
        fid = field["field_id"]
        entry = {
            "field_id": fid,
            "physical_hash": field["physical_hash"],
            "source_representation": field["source_representation"],
            "stratum": field["stratum"],
            "targets": {},
        }
        dists = {}
        for target in TARGETS:
            p, A, a0 = _dist(by_ft[(fid, target)])
            dists[target] = p
            entry["targets"][target] = {
                "p": p.tolist(),
                "A": A,
                "a0": a0,
                "n_valid": sum(
                    1 for r in by_ft[(fid, target)] if r.get("valid")
                ),
                "n_total": len(by_ft[(fid, target)]),
            }
        # Pairwise target contrasts
        pairs = {}
        for i, r1 in enumerate(TARGETS):
            for r2 in TARGETS[i + 1 :]:
                pairs[f"{r1}__vs__{r2}"] = {
                    "d_TV": _tv(dists[r1], dists[r2]),
                    "delta_A": float(
                        entry["targets"][r1]["A"] - entry["targets"][r2]["A"]
                    ),
                    "delta_a0": float(
                        entry["targets"][r1]["a0"] - entry["targets"][r2]["a0"]
                    ),
                }
        entry["pairwise"] = pairs
        field_rows.append(entry)

    # Aggregate target main effect (mean over fields)
    pair_keys = list(field_rows[0]["pairwise"].keys())
    target_main = {
        k: {
            "mean_d_TV": float(np.nanmean([fr["pairwise"][k]["d_TV"] for fr in field_rows])),
            "mean_delta_A": float(
                np.nanmean([fr["pairwise"][k]["delta_A"] for fr in field_rows])
            ),
            "mean_delta_a0": float(
                np.nanmean([fr["pairwise"][k]["delta_a0"] for fr in field_rows])
            ),
        }
        for k in pair_keys
    }

    # Source main: average a0/A over targets within source
    source_stats = {}
    for source in TARGETS:
        subset = [fr for fr in field_rows if fr["source_representation"] == source]
        a0s = []
        As = []
        for fr in subset:
            for t in TARGETS:
                a0s.append(fr["targets"][t]["a0"])
                As.append(fr["targets"][t]["A"])
        source_stats[source] = {
            "n_fields": len(subset),
            "mean_a0": float(np.nanmean(a0s)) if a0s else float("nan"),
            "mean_A": float(np.nanmean(As)) if As else float("nan"),
        }

    # Source × target: mean A by source×target
    interaction = {}
    for source in TARGETS:
        interaction[source] = {}
        subset = [fr for fr in field_rows if fr["source_representation"] == source]
        for target in TARGETS:
            vals = [fr["targets"][target]["A"] for fr in subset]
            interaction[source][target] = float(np.nanmean(vals)) if vals else float("nan")

    # Block drift: mean |ΔA| between block0 and block1 on same field×target
    drifts = []
    for field in fields:
        fid = field["field_id"]
        for target in TARGETS:
            _, A0, _ = _dist(by_ft_block[(fid, target, 0)])
            _, A1, _ = _dist(by_ft_block[(fid, target, 1)])
            if np.isfinite(A0) and np.isfinite(A1):
                drifts.append(abs(A0 - A1))

    report = {
        "status": "primary_3x3_v0_1",
        "inference_unit": "48_physical_fields",
        "n_fields": len(fields),
        "n_trace_rows": len(rows),
        "layers": {
            "1_target_representation_effect": target_main,
            "2_source_field_effect": source_stats,
            "3_source_x_target_interaction_mean_A": interaction,
        },
        "block_drift_mean_abs_delta_A": float(np.mean(drifts)) if drifts else float("nan"),
        "surrogate_refit": "deferred_until_after_primary",
        "field_rows": field_rows,
        "field_meta_note": "stratum/source retained per field for blocked inference",
    }
    out = run_dir / "primary_3x3_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    slim = {k: v for k, v in report.items() if k != "field_rows"}
    (run_dir / "primary_3x3_summary.json").write_text(
        json.dumps(slim, indent=2), encoding="utf-8"
    )
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--fields",
        type=Path,
        default=ROOT
        / "analysis"
        / "matched_rep_collective"
        / "replay_fields_v0_1.json",
    )
    args = parser.parse_args()
    fields = json.loads(args.fields.read_text(encoding="utf-8"))["fields"]
    path = run_primary_report(args.run_dir, fields)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
