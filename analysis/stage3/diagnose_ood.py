"""Stage 3A.2: OOD cause diagnosis (peer-count, components, time axes)."""

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

from analysis.stage3.coverage_utils import (  # noqa: E402
    ComponentOOD,
    feature_builder,
    histogram_from_phases,
)
from analysis.stage3.dataset import build_unified_dataset  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402


PEER_SCAN = (
    # (N_collective, expected_peers, matched_training_label)
    (9, 8, "sparse_N8 / peer=8"),
    (16, 15, "unmeasured (N-1=15)"),
    (17, 16, "sparse_peer16 / peer=16"),
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _init_phases(n_agents: int, rng: np.random.Generator, *, kind: str = "uniform"):
    if kind == "uniform":
        return rng.uniform(-np.pi, np.pi, size=n_agents)
    if kind == "unimodal_weak":
        return rng.vonmises(0.0, 0.5, size=n_agents)
    if kind == "unimodal_strong":
        return rng.vonmises(0.0, 4.0, size=n_agents)
    if kind == "antipodal":
        half = n_agents // 2
        a = rng.vonmises(0.0, 3.0, size=half)
        b = rng.vonmises(np.pi, 3.0, size=n_agents - half)
        return np.concatenate([a, b])
    raise KeyError(kind)


def _collect_fields(
    phases: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, list]:
    rows = []
    unresolved = []
    descriptors = []
    for agent in range(phases.size):
        histogram = histogram_from_phases(phases, agent)
        x, flag, desc = feature_builder(representation, histogram)
        rows.append(x)
        unresolved.append(flag)
        descriptors.append(desc)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(unresolved, dtype=bool),
        descriptors,
    )


def summarize_flags(flags: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.mean(value)) for key, value in flags.items()}


def run_regime_scan(
    component: ComponentOOD,
    *,
    representation: str,
    n_agents: int,
    n_steps: int,
    omega_scale: float,
    coupling: float,
    seed: int,
    regime: str,
    init_kind: str = "uniform",
) -> tuple[list[dict], list[dict]]:
    """Return (time trajectory rows, nearest-neighbor rows for OOD points)."""
    rng = np.random.default_rng(seed)
    phases = _init_phases(n_agents, rng, kind=init_kind)
    omega = rng.normal(0.0, omega_scale, size=n_agents)
    traj_rows: list[dict] = []
    neighbor_rows: list[dict] = []

    for t in range(n_steps):
        x, unresolved, _descs = _collect_fields(phases, representation)
        dists = component.distances(x, unresolved_near_zero=unresolved)
        flags = component.flags(x, unresolved_near_zero=unresolved)
        summary = summarize_flags(flags)
        traj_rows.append(
            {
                "representation": representation,
                "regime": regime,
                "n_agents": n_agents,
                "peer_count": n_agents - 1,
                "init_kind": init_kind,
                "t": t,
                "q_ood_canonical": summary["ood_canonical"],
                "q_ood_native": summary["ood_native"],
                "q_ood_peer_count": summary["ood_peer_count"],
                "q_ood_near_zero": summary["ood_near_zero"],
                "q_ood_combined": summary["ood_combined"],
                "mean_d_canonical": float(np.mean(dists["d_canonical"])),
                "mean_d_native": float(np.mean(dists["d_native"])),
                "mean_d_peer": float(np.mean(dists["d_peer"])),
                "mean_abs_z1": float(np.mean(np.abs(x[:, 2]))),
            }
        )

        # Store nearest neighbors for OOD-combined points (cap per step).
        ood_idx = np.where(flags["ood_combined"])[0]
        if ood_idx.size:
            take = ood_idx[: min(8, ood_idx.size)]
            nn_idx, nn_dist = component.nearest_training_indices(x[take], view="canonical")
            for local, train_i, dist in zip(take, nn_idx, nn_dist):
                neighbor_rows.append(
                    {
                        "representation": representation,
                        "regime": regime,
                        "n_agents": n_agents,
                        "t": t,
                        "agent": int(local),
                        "peer_count_query": int(round(x[local, 14])),
                        "abs_z1": float(x[local, 2]),
                        "abs_z2": float(x[local, 5]),
                        "abs_z3": float(x[local, 8]),
                        "d_canonical": float(dists["d_canonical"][local]),
                        "d_native": float(dists["d_native"][local]),
                        "d_peer": float(dists["d_peer"][local]),
                        "policy_a": int(dists["d_policy_a"][local] > 0.5),
                        "nn_train_index": int(train_i),
                        "nn_canonical_dist": float(dist),
                    }
                )

        if regime == "t0_only":
            break
        if regime == "k0_drift":
            phases = wrap_phase(phases + omega)
        elif regime == "closed_loop":
            # Use kernel-NN only for dynamics envelope; diagnosis still OOD-focused.
            # Actions from a trivial polar heuristic to avoid self-fitting bias:
            # g ~ Im(z1) sign — not the training surrogate.
            actions = np.sign(x[:, 1])  # im_z1
            phases = wrap_phase(phases + omega + coupling * actions)
        else:
            raise KeyError(regime)

    return traj_rows, neighbor_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stage 3A training manifold...")
    all_rows, packed = build_unified_dataset(ROOT)

    peer_scan_rows: list[dict] = []
    traj_rows: list[dict] = []
    neighbor_rows: list[dict] = []
    component_summary: list[dict] = []

    for representation, arrays in packed.items():
        print(f"\n=== diagnose {representation} ===")
        x = arrays["features"]
        rep_rows = [row for row in all_rows if row.representation == representation]
        component = ComponentOOD.fit(x)
        train_peers = sorted({int(round(v)) for v in component.peer_train_values})
        print(f"  training peer_counts={train_peers}")

        # 1.1 peer-count scan at t=0 and short closed-loop
        for n_agents, peers, label in PEER_SCAN:
            for regime in ("t0_only", "k0_drift", "closed_loop"):
                steps = 1 if regime == "t0_only" else args.steps
                traj, neigh = run_regime_scan(
                    component,
                    representation=representation,
                    n_agents=n_agents,
                    n_steps=steps,
                    omega_scale=0.03,
                    coupling=0.12,
                    seed=args.seed + n_agents,
                    regime=regime,
                    init_kind="uniform",
                )
                traj_rows.extend(traj)
                # Attach metadata only for this representation's row list.
                for item in neigh:
                    train_i = item["nn_train_index"]
                    row = rep_rows[train_i]
                    item["nn_source_family"] = row.source_family
                    item["nn_profile"] = row.profile
                    item["nn_representation"] = row.representation
                    # peer from features of nearest train point
                    item["nn_peer_count"] = int(round(x[train_i, 14]))
                    item["nn_abs_z1"] = float(x[train_i, 2])
                    item["nn_abs_z2"] = float(x[train_i, 5])
                    item["nn_abs_z3"] = float(x[train_i, 8])
                neighbor_rows.extend(neigh)

                # Aggregate q_ood over trajectory (mean over t).
                last = traj[-1]
                first = traj[0]
                peer_scan_rows.append(
                    {
                        "representation": representation,
                        "n_agents": n_agents,
                        "peer_count": peers,
                        "matched_training": label,
                        "regime": regime,
                        "q_ood_combined_t0": first["q_ood_combined"],
                        "q_ood_combined_mean": float(
                            np.mean([r["q_ood_combined"] for r in traj])
                        ),
                        "q_ood_combined_final": last["q_ood_combined"],
                        "q_ood_canonical_mean": float(
                            np.mean([r["q_ood_canonical"] for r in traj])
                        ),
                        "q_ood_native_mean": float(
                            np.mean([r["q_ood_native"] for r in traj])
                        ),
                        "q_ood_peer_count_mean": float(
                            np.mean([r["q_ood_peer_count"] for r in traj])
                        ),
                        "q_ood_near_zero_mean": float(
                            np.mean([r["q_ood_near_zero"] for r in traj])
                        ),
                        "training_peer_counts": ",".join(str(p) for p in train_peers),
                    }
                )

        # Component summary at N=17 (matched) vs N=16 (mismatch), closed loop.
        for n_agents in (16, 17):
            traj, _ = run_regime_scan(
                component,
                representation=representation,
                n_agents=n_agents,
                n_steps=args.steps,
                omega_scale=0.03,
                coupling=0.12,
                seed=101 + n_agents,
                regime="closed_loop",
                init_kind="uniform",
            )
            component_summary.append(
                {
                    "representation": representation,
                    "n_agents": n_agents,
                    "peer_count": n_agents - 1,
                    "q_ood_canonical": float(np.mean([r["q_ood_canonical"] for r in traj])),
                    "q_ood_native": float(np.mean([r["q_ood_native"] for r in traj])),
                    "q_ood_peer_count": float(
                        np.mean([r["q_ood_peer_count"] for r in traj])
                    ),
                    "q_ood_near_zero": float(
                        np.mean([r["q_ood_near_zero"] for r in traj])
                    ),
                    "q_ood_combined": float(np.mean([r["q_ood_combined"] for r in traj])),
                }
            )

    _write_csv(out_dir / "collective_peer_count_scan.csv", peer_scan_rows)
    _write_csv(out_dir / "ood_time_trajectories.csv", traj_rows)
    _write_csv(out_dir / "nearest_training_neighbors.csv", neighbor_rows)
    _write_csv(out_dir / "ood_component_summary.csv", component_summary)

    # Compact JSON verdict helper.
    verdict = {
        "training_peer_counts_union": sorted(
            {
                int(round(float(v)))
                for arrays in packed.values()
                for v in arrays["features"][:, 14]
            }
        ),
        "peer_scan_preview": peer_scan_rows[:12],
        "component_summary": component_summary,
    }
    (out_dir / "ood_diagnosis_preview.json").write_text(
        json.dumps(verdict, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote peer/component/time diagnosis under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
