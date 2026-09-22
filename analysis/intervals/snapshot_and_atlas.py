"""Intervals INT-0/1: snapshot, scope split, phenotype atlas, hash lock."""

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

from analysis.centers.discrete_bin_features import (  # noqa: E402
    DISCRETE_FEATURE_NAMES,
    discrete_bin_features_from_native,
)
from analysis.centers.peer_stratified_models import PEER_IDX, bin_sparsity  # noqa: E402
from analysis.stage3.applicability_domain import N_COMMON  # noqa: E402
from analysis.stage3.dataset import (  # noqa: E402
    build_unified_dataset,
    post_pilot_source_runs,
    write_dataset_csv,
)
from analysis.stage3.models import counts_to_probs  # noqa: E402
from circlemap.field_features import COMMON_FEATURE_NAMES  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.representations import build_representation_prompt_from_histogram  # noqa: E402

REP = "intervals_24_decimal6"
BRANCH = ROOT / "analysis" / "intervals_branch"
COLLECTIVE_PEERS = (8, 16)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_json(payload: object) -> str:
    return _sha_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def _physical_hash(fractions: np.ndarray, peer: int) -> str:
    payload = {
        "fractions": [round(float(v), 12) for v in fractions.tolist()],
        "peer_count": int(peer),
        "n_bins": int(fractions.size),
    }
    return _sha_json(payload)


def _prompt_hash(histogram: RelativePhaseHistogram) -> str:
    prompt = build_representation_prompt_from_histogram(REP, histogram)
    return _sha_bytes(prompt.encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    out = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Building unified dataset for intervals...")
    all_rows, packed = build_unified_dataset(
        ROOT, source_runs=post_pilot_source_runs(ROOT)
    )
    rows = [r for r in all_rows if r.representation == REP]
    if not rows:
        raise SystemExit(f"no rows for {REP}")
    block = packed[REP]
    x = block["features"]
    counts = block["counts"]
    unresolved = block["unresolved_near_zero"]
    if len(rows) != x.shape[0]:
        raise RuntimeError("row/feature misalignment")

    # Per-row physical + prompt hashes
    physical_hashes = []
    prompt_hashes = []
    discrete = []
    for i, row in enumerate(rows):
        # Rebuild histogram from features' native bins + peer
        native = x[i, N_COMMON:]
        peer = int(np.rint(x[i, PEER_IDX]))
        # Renormalize native to fractions
        s = float(np.sum(native))
        frac = native / s if s > 0 else native
        edges = np.linspace(-np.pi, np.pi, frac.size + 1, dtype=np.float64)
        hist = RelativePhaseHistogram(edges=edges, fractions=frac, peer_count=peer)
        physical_hashes.append(_physical_hash(frac, peer))
        prompt_hashes.append(_prompt_hash(hist))
        discrete.append(discrete_bin_features_from_native(frac))
    discrete_arr = np.asarray(discrete, dtype=np.float64)
    peers = np.rint(x[:, PEER_IDX]).astype(np.int64)

    write_dataset_csv(rows, out / "training_rows_intervals_v0.csv")
    # Augment CSV with hashes
    with (out / "training_rows_intervals_v0.csv").open(encoding="utf-8") as handle:
        base_rows = list(csv.DictReader(handle))
    with (out / "training_rows_intervals_v0.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(base_rows[0].keys()) + [
            "peer_count",
            "bin_sparsity",
            "physical_hash",
            "prompt_hash",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(base_rows):
            payload = dict(row)
            payload["peer_count"] = int(peers[i])
            payload["bin_sparsity"] = float(bin_sparsity(x[i : i + 1])[0])
            payload["physical_hash"] = physical_hashes[i]
            payload["prompt_hash"] = prompt_hashes[i]
            writer.writerow(payload)

    np.savez_compressed(
        out / "training_arrays_intervals_v0.npz",
        features=x,
        counts=counts,
        unresolved_near_zero=unresolved,
        estimated_eps=block["estimated_eps"],
        peer_count=peers,
        bin_sparsity=bin_sparsity(x),
        discrete_features=discrete_arr,
        discrete_feature_names=np.asarray(DISCRETE_FEATURE_NAMES),
        physical_hash=np.asarray(physical_hashes),
        prompt_hash=np.asarray(prompt_hashes),
        common_feature_names=np.asarray(COMMON_FEATURE_NAMES),
        native_feature_names=np.asarray([f"bin_{i:02d}" for i in range(24)]),
    )

    # Scope split
    collective_mask = np.isin(peers, list(COLLECTIVE_PEERS))
    dense_mask = peers == 240

    def _write_scope(mask: np.ndarray, label: str) -> dict:
        idx = np.where(mask)[0]
        directory = out / (
            "collective_scope_v0" if label == "collective" else "dense_exploratory_v0"
        )
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "arrays.npz",
            features=x[idx],
            counts=counts[idx],
            unresolved_near_zero=unresolved[idx],
            peer_count=peers[idx],
            bin_sparsity=bin_sparsity(x[idx]),
            discrete_features=discrete_arr[idx],
            physical_hash=np.asarray(physical_hashes)[idx],
            prompt_hash=np.asarray(prompt_hashes)[idx],
            source_row_index=idx,
        )
        scope_rows = []
        with (out / "training_rows_intervals_v0.csv").open(encoding="utf-8") as handle:
            all_csv = list(csv.DictReader(handle))
        for j in idx:
            scope_rows.append(all_csv[j])
        csv_name = (
            "training_rows_intervals_collective_v0.csv"
            if label == "collective"
            else "training_rows_intervals_dense_v0.csv"
        )
        with (out / csv_name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scope_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scope_rows)
        # also copy into scope dir
        with (directory / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scope_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scope_rows)
        meta = {
            "label": label,
            "n_rows": int(idx.size),
            "peer_counts": dict(Counter(int(p) for p in peers[idx].tolist())),
            "source_counts": dict(Counter(all_csv[j]["source_family"] for j in idx)),
            "arrays_sha256": hashlib.sha256(
                (directory / "arrays.npz").read_bytes()
            ).hexdigest(),
            "intended_use": (
                "Stage C collective surrogate (N in {9,17})"
                if label == "collective"
                else "Dense peer=240 exploratory only; not collective freeze"
            ),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    collective_meta = _write_scope(collective_mask, "collective")
    dense_meta = _write_scope(dense_mask, "dense")

    # Phenotype atlas (peer × family)
    atlas_rows = []
    family_peer: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        family_peer[(row.profile_family, int(peers[i]))].append(i)
    for (family, peer), idxs in sorted(family_peer.items()):
        c = counts[idxs]
        p = counts_to_probs(c, alpha=0.0)
        a0 = p[:, 2] - p[:, 0]
        A = p[:, 0] + p[:, 2]
        C1 = np.abs(a0)
        C2 = p[:, 1]
        atlas_rows.append(
            {
                "profile_family": family,
                "peer_count": peer,
                "n_rows": len(idxs),
                "mean_p_minus": float(np.mean(p[:, 0])),
                "mean_p_stay": float(np.mean(p[:, 1])),
                "mean_p_plus": float(np.mean(p[:, 2])),
                "mean_A": float(np.mean(A)),
                "mean_a0": float(np.mean(a0)),
                "mean_C1": float(np.mean(C1)),
                "mean_C2": float(np.mean(C2)),
                "mean_bin_sparsity": float(np.mean(bin_sparsity(x[idxs]))),
                "mean_n_occupied": float(
                    np.mean(discrete_arr[idxs, 0])
                ),
                "mean_max_zero_run": float(np.mean(discrete_arr[idxs, 2])),
                "mean_focal_mass": float(np.mean(discrete_arr[idxs, 7])),
            }
        )
    with (out / "phenotype_atlas_intervals_v0.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(atlas_rows[0].keys()))
        writer.writeheader()
        writer.writerows(atlas_rows)

    source_summary = []
    for src, n in Counter(r.source_family for r in rows).items():
        source_summary.append({"source_family": src, "n_rows": n})
    with (out / "source_profile_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_family", "n_rows"])
        writer.writeheader()
        writer.writerows(source_summary)

    totals = np.sum(counts, axis=1, keepdims=True)
    p_all = counts / np.maximum(totals, 1.0)
    manifest = {
        "version": "intervals-int0-v0",
        "representation": REP,
        "n_rows_total": len(rows),
        "peer_counts_total": dict(Counter(int(p) for p in peers.tolist())),
        "mean_p_stay": float(np.mean(p_all[:, 1])),
        "mean_activity": float(np.mean(p_all[:, 0] + p_all[:, 2])),
        "mean_abs_a0": float(np.mean(np.abs(p_all[:, 2] - p_all[:, 0]))),
        "collective": collective_meta,
        "dense_exploratory": dense_meta,
        "n_unique_physical_hash": len(set(physical_hashes)),
        "n_unique_prompt_hash": len(set(prompt_hashes)),
        "arrays_sha256": hashlib.sha256(
            (out / "training_arrays_intervals_v0.npz").read_bytes()
        ).hexdigest(),
        "scope_policy": {
            "collective_peers": list(COLLECTIVE_PEERS),
            "dense_peer": 240,
            "optional_narrow": "intervals_collective_bundle_v1_N17 peer=16",
            "no_cyclic_shift_augmentation": True,
            "baseline": "peer_specific_global",
        },
        "paid_authorized": False,
        "freeze_ready": False,
    }
    manifest["manifest_sha256"] = _sha_json(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    (out / "dataset_manifest_intervals_v0.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out / "scope_split_v0.json").write_text(
        json.dumps(
            {
                "collective": collective_meta,
                "dense_exploratory": dense_meta,
                "bundle_names": {
                    "intervals_collective_scope": "peer in {8,16}",
                    "intervals_dense_exploratory": "peer=240",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
