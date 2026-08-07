"""Build peer{8,16} collective-scope snapshot; park peer=240 as dense exploratory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.centers.peer_stratified_models import (  # noqa: E402
    COLLECTIVE_PEERS,
    PEER_IDX,
    bin_sparsity,
)
from circlemap.field_features import COMMON_FEATURE_NAMES  # noqa: E402
from circlemap.response import ACTION_VALUES  # noqa: E402

BRANCH = ROOT / "analysis" / "centers_branch"
REP = "centers_24_standard"


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _scan_fail_rates(run_roots: list[Path]) -> dict[tuple[str, str, int], dict]:
    """Aggregate strict parse fails by (run_parent_key, profile, offset_index)."""
    cells: dict[tuple[str, str, int], dict] = {}
    for root in run_roots:
        if not root.exists():
            continue
        for trace in root.rglob("trace.jsonl"):
            key_root = str(trace.parent)
            with trace.open(encoding="utf-8") as handle:
                for line in handle:
                    rec = json.loads(line)
                    if rec.get("representation") != REP:
                        continue
                    profile = str(rec["profile"])
                    off = int(rec["offset_index"])
                    slot = cells.setdefault(
                        (key_root, profile, off),
                        {"n_calls": 0, "n_invalid": 0, "n_valid": 0},
                    )
                    slot["n_calls"] += 1
                    if rec.get("valid"):
                        slot["n_valid"] += 1
                    else:
                        slot["n_invalid"] += 1
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir

    arrays = np.load(branch / "training_arrays_centers_v0.npz", allow_pickle=True)
    x = arrays["features"]
    counts = arrays["counts"]
    unresolved = arrays["unresolved_near_zero"]
    peers = np.rint(x[:, PEER_IDX]).astype(np.int64)

    rows = []
    with (branch / "training_rows_centers_v0.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != x.shape[0]:
        raise SystemExit("row/array mismatch")

    collective_mask = np.isin(peers, list(COLLECTIVE_PEERS))
    dense_mask = peers == 240

    # Optional fail-rate join from known run families (best-effort).
    fail_cells = _scan_fail_rates(
        [
            ROOT / "runs" / "sparse_peer",
            ROOT / "runs" / "stimulus_manifold",
            ROOT / "runs" / "centers_cent3",
            ROOT / "runs" / "response_law_representation",
        ]
    )
    # Index by profile+offset aggregated across runs
    by_profile_off: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"n_calls": 0, "n_invalid": 0, "n_valid": 0}
    )
    for (_run, profile, off), slot in fail_cells.items():
        key = (profile, off)
        for k in ("n_calls", "n_invalid", "n_valid"):
            by_profile_off[key][k] += slot[k]

    p_fail = np.full(x.shape[0], np.nan, dtype=np.float64)
    n_calls_arr = np.zeros(x.shape[0], dtype=np.int64)
    for i, row in enumerate(rows):
        key = (row["profile"], int(row["offset_index"]))
        slot = by_profile_off.get(key)
        if slot and slot["n_calls"] > 0:
            p_fail[i] = slot["n_invalid"] / slot["n_calls"]
            n_calls_arr[i] = slot["n_calls"]

    out = branch / "collective_scope_v0"
    out.mkdir(parents=True, exist_ok=True)
    dense_out = branch / "dense_exploratory_v0"
    dense_out.mkdir(parents=True, exist_ok=True)

    def _write_split(mask: np.ndarray, directory: Path, label: str) -> dict:
        idx = np.where(mask)[0]
        xs = x[idx]
        cs = counts[idx]
        uns = unresolved[idx]
        pf = p_fail[idx]
        nc = n_calls_arr[idx]
        split_rows = [rows[i] for i in idx]
        np.savez_compressed(
            directory / "arrays.npz",
            features=xs,
            counts=cs,
            unresolved_near_zero=uns,
            p_fail=pf,
            n_trace_calls=nc,
            peer_count=np.rint(xs[:, PEER_IDX]).astype(np.int64),
            bin_sparsity=bin_sparsity(xs),
            common_feature_names=np.asarray(COMMON_FEATURE_NAMES),
            native_feature_names=np.asarray([f"bin_{i:02d}" for i in range(24)]),
            source_row_index=idx,
        )
        with (directory / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(split_rows[0].keys())
                + ["peer_count", "bin_sparsity", "p_fail", "n_trace_calls"],
            )
            writer.writeheader()
            for j, row in enumerate(split_rows):
                payload = dict(row)
                payload["peer_count"] = int(np.rint(xs[j, PEER_IDX]))
                payload["bin_sparsity"] = float(bin_sparsity(xs[j : j + 1])[0])
                payload["p_fail"] = "" if not np.isfinite(pf[j]) else float(pf[j])
                payload["n_trace_calls"] = int(nc[j])
                writer.writerow(payload)
        peer_counts = Counter(int(v) for v in np.rint(xs[:, PEER_IDX]).tolist())
        source_counts = Counter(r["source_family"] for r in split_rows)
        meta = {
            "label": label,
            "n_rows": int(idx.size),
            "peer_counts": {str(k): int(v) for k, v in sorted(peer_counts.items())},
            "source_counts": dict(source_counts),
            "p_fail_coverage": {
                "n_with_trace_fail_rate": int(np.sum(np.isfinite(pf))),
                "mean_p_fail_where_known": float(np.nanmean(pf))
                if np.any(np.isfinite(pf))
                else None,
            },
            "intended_use": (
                "Stage C collective surrogate training (N in {9,17})"
                if label == "collective"
                else "Dense peer=240 phenotype comparison only; not Stage C"
            ),
            "arrays_sha256": hashlib.sha256(
                (directory / "arrays.npz").read_bytes()
            ).hexdigest(),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    collective_meta = _write_split(collective_mask, out, "collective")
    dense_meta = _write_split(dense_mask, dense_out, "dense_exploratory")

    summary = {
        "status": "centers_scope_split_v0",
        "representation": REP,
        "collective_peers": list(COLLECTIVE_PEERS),
        "collective": collective_meta,
        "dense_exploratory": dense_meta,
        "excluded_from_collective_training": {"peer_240_rows": int(np.sum(dense_mask))},
        "bundle_names": {
            "centers_collective_bundle_v1": "peer in {8,16}; intended N in {9,17}",
            "centers_dense_exploratory": "peer=240; not used in Stage C",
        },
        "manifest_sha256": _sha(
            {"collective": collective_meta, "dense": dense_meta}
        ),
    }
    (branch / "collective_scope_split_v0.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
