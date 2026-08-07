"""Prospective compare: locked v1 predictions vs replay empirical response law."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTION_INDEX = {"retard": 0, "stay": 1, "advance": 2}


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--pred-lock",
        type=Path,
        default=ROOT
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "v1_predictions_locked.json",
    )
    parser.add_argument(
        "--fields",
        type=Path,
        default=ROOT
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "replay_fields_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage_c_artifacts" / "replay_panel_v1",
    )
    args = parser.parse_args()

    pred = {
        row["field_id"]: row
        for row in json.loads(args.pred_lock.read_text(encoding="utf-8"))["fields"]
    }
    meta = {
        row["field_id"]: row
        for row in json.loads(args.fields.read_text(encoding="utf-8"))["fields"]
    }
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    with (args.run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if not rec.get("valid"):
                continue
            profile = rec["profile"]
            counts[profile][ACTION_INDEX[rec["action_label"]]] += 1

    rows = []
    for field_id, c in sorted(counts.items()):
        n = float(np.sum(c))
        p_obs = c / max(n, 1.0)
        p_hat = np.asarray(
            [
                pred[field_id]["pred_p_retard"],
                pred[field_id]["pred_p_stay"],
                pred[field_id]["pred_p_advance"],
            ],
            dtype=np.float64,
        )
        active_obs = p_obs[0] + p_obs[2]
        active_hat = p_hat[0] + p_hat[2]
        q_adv_obs = p_obs[2] / active_obs if active_obs > 1e-12 else float("nan")
        q_adv_hat = p_hat[2] / active_hat if active_hat > 1e-12 else float("nan")
        m = meta[field_id]
        rows.append(
            {
                "field_id": field_id,
                "source_bucket": m["source_bucket"],
                "n_obs": int(n),
                "p_stay_hat": float(p_hat[1]),
                "p_stay_obs": float(p_obs[1]),
                "delta_p_stay": float(p_obs[1] - p_hat[1]),
                "p_retard_obs": float(p_obs[0]),
                "p_advance_obs": float(p_obs[2]),
                "e_tv": _tv(p_hat, p_obs),
                "q_advance_obs": q_adv_obs,
                "q_advance_hat": q_adv_hat,
                "R": m["R"],
                "coupling": m.get("coupling"),
                "r1": m.get("r1"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "replay_vs_v1_prospective.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def bucket_stats(name: str | None = None) -> dict:
        sub = [r for r in rows if name is None or r["source_bucket"] == name]
        return {
            "n": len(sub),
            "mean_e_tv": float(np.mean([r["e_tv"] for r in sub])),
            "mean_delta_p_stay": float(np.mean([r["delta_p_stay"] for r in sub])),
            "mean_p_stay_obs": float(np.mean([r["p_stay_obs"] for r in sub])),
            "mean_p_stay_hat": float(np.mean([r["p_stay_hat"] for r in sub])),
            "frac_obs_stay_lt_0.05": float(
                np.mean([r["p_stay_obs"] < 0.05 for r in sub])
            ),
            "frac_obs_stay_lt_0.10": float(
                np.mean([r["p_stay_obs"] < 0.10 for r in sub])
            ),
        }

    # Branch heuristic
    overall = bucket_stats()
    neg = bucket_stats("neg")
    pos = bucket_stats("pos")
    # A: replay stay near zero (confirms Stage C)
    # B: replay stay matches v1 high stay
    # C: differs by bucket systematically
    mean_obs_stay = overall["mean_p_stay_obs"]
    mean_hat_stay = overall["mean_p_stay_hat"]
    if mean_obs_stay < 0.05 and mean_hat_stay > 0.08:
        branch = "A_replay_confirms_low_stay_v1_smoothes"
    elif abs(mean_obs_stay - mean_hat_stay) < 0.05 and mean_obs_stay > 0.08:
        branch = "B_replay_matches_v1_stage_c_sampling_issue"
    elif abs(neg["mean_delta_p_stay"] - pos["mean_delta_p_stay"]) > 0.08:
        branch = "C_bucket_dependent_discrepancy_feature_limit"
    else:
        branch = "mixed_inspect_per_field"

    summary = {
        "run_dir": str(args.run_dir),
        "branch": branch,
        "overall": overall,
        "by_bucket": {
            b: bucket_stats(b)
            for b in sorted({r["source_bucket"] for r in rows})
        },
        "artifact": str(out_csv.relative_to(ROOT).as_posix()),
    }
    (args.out_dir / "replay_prospective_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
