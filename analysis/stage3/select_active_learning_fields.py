"""Select coverage-directed active-learning fields from the candidate bank."""

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

from analysis.stage3.dataset import build_unified_dataset  # noqa: E402
from analysis.stage3.models import fit_candidate_models  # noqa: E402
from analysis.stage3.coverage_utils import N_COMMON  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument("--n-select", type=int, default=36)
    parser.add_argument("--w-occupancy", type=float, default=1.0)
    parser.add_argument("--w-ood", type=float, default=1.0)
    parser.add_argument("--w-disagreement", type=float, default=1.0)
    args = parser.parse_args()
    out_dir = args.out_dir
    bank_path = out_dir / "candidate_field_bank.npz"
    if not bank_path.exists():
        raise SystemExit(f"missing {bank_path}; run build_collective_field_bank.py first")

    bank = np.load(bank_path, allow_pickle=True)
    features = bank["features_canonical"]
    features_common = bank["features_common"]
    n = features.shape[0]

    keys = [
        _quantize_key(
            float(bank["abs_z1"][i]),
            float(bank["abs_z2"][i]),
            int(bank["peer_count"][i]),
            float(bank["estimated_eps"][i]),
        )
        for i in range(n)
    ]
    occ = Counter(keys)
    max_occ = max(occ.values()) if occ else 1

    # Disagreement from models fitted on moments common features only.
    _rows, packed = build_unified_dataset(ROOT)
    rep = "moments_m1_m3"
    x_train = packed[rep]["features"][:, :N_COMMON]
    models = fit_candidate_models(
        x_train,
        packed[rep]["counts"],
        include_hurdle=True,
    )
    preds = []
    for name in ("kernel_nn", "hurdle_multinomial", "softmax_stump_boost"):
        preds.append(models[name].predict_proba(features_common))
    disagreement = np.zeros(n, dtype=np.float64)
    pairs = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            disagreement += 0.5 * np.sum(np.abs(preds[i] - preds[j]), axis=1)
            pairs += 1
    disagreement /= max(pairs, 1)

    ood = np.nan_to_num(bank["d_combined"].astype(np.float64), nan=0.0)
    ood_n = (ood - ood.min()) / max(float(ood.max() - ood.min()), 1e-12)
    dis_n = (disagreement - disagreement.min()) / max(
        float(disagreement.max() - disagreement.min()), 1e-12
    )
    occ_n = np.asarray([occ[k] / max_occ for k in keys], dtype=np.float64)

    priority = (
        args.w_occupancy * occ_n
        + args.w_ood * ood_n
        + args.w_disagreement * dis_n
    )
    priority = priority + 0.15 * bank["ood_combined"].astype(np.float64)
    # Prefer peer-matched collectives; downweight pure peer-mismatch artifacts.
    priority = priority + 0.20 * np.isin(bank["peer_count"], [8, 16]).astype(np.float64)
    priority = priority - 0.25 * bank["ood_peer"].astype(np.float64)

    order = np.argsort(-priority)
    pool = order[: min(800, n)]
    canon = features[pool]
    mean = canon.mean(axis=0)
    scale = np.where(canon.std(axis=0) < 1e-8, 1.0, canon.std(axis=0))
    zs = (canon - mean) / scale

    selected_local: list[int] = []
    for _ in range(min(args.n_select, pool.size)):
        if not selected_local:
            selected_local.append(0)
            continue
        sel = zs[selected_local]
        dmin = np.full(zs.shape[0], np.inf)
        chunk = 64
        for start in range(0, zs.shape[0], chunk):
            stop = min(start + chunk, zs.shape[0])
            diff = zs[start:stop, None, :] - sel[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)
            dmin[start:stop] = dist
        score = 0.7 * (dmin / max(float(dmin.max()), 1e-12)) + 0.3 * (
            priority[pool] / max(float(priority[pool].max()), 1e-12)
        )
        for already in selected_local:
            score[already] = -1.0
        selected_local.append(int(np.argmax(score)))

    selected = pool[np.asarray(selected_local, dtype=int)]
    out_rows = []
    near_zero_flags = []
    for rank, i in enumerate(selected):
        abs_z1 = float(bank["abs_z1"][i])
        eps = float(bank["estimated_eps"][i])
        near_zero = bool(abs_z1 < 0.25 and 1e-6 < abs(eps) < 0.02)
        near_zero_flags.append(near_zero)
        out_rows.append(
            {
                "rank": rank,
                "bank_index": int(i),
                "priority": float(priority[i]),
                "occupancy_n": float(occ_n[i]),
                "ood_n": float(ood_n[i]),
                "disagreement_n": float(dis_n[i]),
                "representation": str(bank["representation"][i]),
                "source": str(bank["source"][i]),
                "init_kind": str(bank["init_kind"][i]),
                "dynamics": str(bank["dynamics"][i]),
                "n_agents": int(bank["n_agents"][i]),
                "peer_count": int(bank["peer_count"][i]),
                "t": int(bank["t"][i]),
                "abs_z1": abs_z1,
                "abs_z2": float(bank["abs_z2"][i]),
                "abs_z3": float(bank["abs_z3"][i]),
                "estimated_eps": eps,
                "ood_combined": bool(bank["ood_combined"][i]),
                "ood_peer": bool(bank["ood_peer"][i]),
                "ood_canonical": bool(bank["ood_canonical"][i]),
                "near_zero_band": near_zero,
            }
        )

    _write_csv(out_dir / "active_learning_selection.csv", out_rows)
    # Bank-wide near-zero occupancy among peer-matched fields.
    peer_matched = np.isin(bank["peer_count"], [8, 16])
    near_zero_bank = (bank["abs_z1"] < 0.25) & (np.abs(bank["estimated_eps"]) < 0.02) & (
        np.abs(bank["estimated_eps"]) > 1e-6
    )
    summary = {
        "n_select": len(out_rows),
        "frac_ood_combined": float(np.mean([r["ood_combined"] for r in out_rows])),
        "frac_ood_peer": float(np.mean([r["ood_peer"] for r in out_rows])),
        "frac_ood_canonical": float(np.mean([r["ood_canonical"] for r in out_rows])),
        "frac_near_zero_band": float(np.mean(near_zero_flags)) if near_zero_flags else 0.0,
        "bank_frac_near_zero_peer_matched": float(
            np.mean(near_zero_bank[peer_matched]) if np.any(peer_matched) else 0.0
        ),
        "peer_count_histogram": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique([r["peer_count"] for r in out_rows], return_counts=True))
        },
        "recommended_pilot_calls_low": 24 * 3 * 8 * 2,
        "recommended_pilot_calls_high": 48 * 3 * 8 * 2,
        "near_zero_topup_priority": (
            "consider_eps_0.005_0.01"
            if float(np.mean(near_zero_bank[peer_matched])) >= 0.15
            else "not_first_priority"
        ),
    }
    (out_dir / "active_learning_selection_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
