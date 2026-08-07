"""Build offline collective-like candidate field bank (no API).

Sources:
  A. Designed initial conditions
  B. K=0 exact drift trajectories
  C. Multiple virtual interaction kernels (not only the fitted surrogate)
"""

from __future__ import annotations

import argparse
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
from analysis.stage3.models import fit_candidate_models  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402


def _phases(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "uniform":
        return rng.uniform(-np.pi, np.pi, size=n)
    if kind == "unimodal_weak":
        return rng.vonmises(0.0, 0.4, size=n)
    if kind == "unimodal_moderate":
        return rng.vonmises(0.0, 1.5, size=n)
    if kind == "unimodal_strong":
        return rng.vonmises(0.0, 5.0, size=n)
    if kind == "asymmetric_two_cluster":
        n1 = max(1, int(round(0.7 * n)))
        a = rng.vonmises(-0.6, 3.0, size=n1)
        b = rng.vonmises(0.9, 3.0, size=n - n1)
        return np.concatenate([a, b])
    if kind == "antipodal_two_cluster":
        n1 = n // 2
        a = rng.vonmises(0.0, 4.0, size=n1)
        b = rng.vonmises(np.pi, 4.0, size=n - n1)
        return np.concatenate([a, b])
    if kind == "three_cluster":
        cuts = np.array_split(np.arange(n), 3)
        centers = (-2.0, 0.0, 2.0)
        parts = [
            rng.vonmises(center, 3.0, size=len(idx))
            for center, idx in zip(centers, cuts)
        ]
        return np.concatenate(parts)
    if kind == "cluster_plus_background":
        n_core = max(2, n // 3)
        core = rng.vonmises(0.0, 6.0, size=n_core)
        bg = rng.uniform(-np.pi, np.pi, size=n - n_core)
        return np.concatenate([core, bg])
    raise KeyError(kind)


INIT_KINDS = (
    "uniform",
    "unimodal_weak",
    "unimodal_moderate",
    "unimodal_strong",
    "asymmetric_two_cluster",
    "antipodal_two_cluster",
    "three_cluster",
    "cluster_plus_background",
)


def _virtual_actions(name: str, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic / lightly stochastic virtual kernels on common features."""
    im_z1 = x[:, 1]
    re_z1 = x[:, 0]
    im_z2 = x[:, 4]
    if name == "polar_kernel":
        return np.sign(im_z1)
    if name == "second_harmonic_kernel":
        return np.sign(im_z2)
    if name == "negative_odd_kernel":
        return -np.sign(im_z1)
    if name == "soft_polar":
        return np.tanh(3.0 * im_z1)
    if name == "random_walk":
        return rng.choice([-1.0, 0.0, 1.0], size=x.shape[0])
    # Surrogate names handled by caller.
    raise KeyError(name)


def _append_snapshot(
    store: dict,
    *,
    representation: str,
    source: str,
    n_agents: int,
    t: int,
    seed: int,
    init_kind: str,
    dynamics: str,
    phases: np.ndarray,
    component: ComponentOOD | None,
) -> None:
    from analysis.stage3.coverage_utils import N_COMMON, CANONICAL_INDICES

    for agent in range(phases.size):
        histogram = histogram_from_phases(phases, agent)
        x, unresolved, desc = feature_builder(representation, histogram)
        # Store common+canonical views with fixed width across representations.
        store["features_common"].append(x[:N_COMMON])
        store["features_canonical"].append(x[list(CANONICAL_INDICES)])
        store["unresolved"].append(unresolved)
        store["representation"].append(representation)
        store["source"].append(source)
        store["n_agents"].append(n_agents)
        store["peer_count"].append(n_agents - 1)
        store["t"].append(t)
        store["seed"].append(seed)
        store["init_kind"].append(init_kind)
        store["dynamics"].append(dynamics)
        store["abs_z1"].append(float(desc.common[2]))
        store["abs_z2"].append(float(desc.common[5]))
        store["abs_z3"].append(float(desc.common[8]))
        store["estimated_eps"].append(float(desc.estimated_eps))
        if component is not None:
            dists = component.distances(
                x[None, :],
                unresolved_near_zero=np.asarray([unresolved], dtype=bool),
            )
            flags = component.flags(
                x[None, :],
                unresolved_near_zero=np.asarray([unresolved], dtype=bool),
            )
            store["d_combined"].append(float(dists["d_combined"][0]))
            store["ood_combined"].append(bool(flags["ood_combined"][0]))
            store["ood_peer"].append(bool(flags["ood_peer_count"][0]))
            store["ood_canonical"].append(bool(flags["ood_canonical"][0]))
            store["ood_native"].append(bool(flags["ood_native"][0]))
        else:
            store["d_combined"].append(np.nan)
            store["ood_combined"].append(False)
            store["ood_peer"].append(False)
            store["ood_canonical"].append(False)
            store["ood_native"].append(False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _rows, packed = build_unified_dataset(ROOT)
    components = {
        representation: ComponentOOD.fit(arrays["features"])
        for representation, arrays in packed.items()
    }
    models_by_rep = {
        representation: fit_candidate_models(
            arrays["features"],
            arrays["counts"],
            include_hurdle=True,
        )
        for representation, arrays in packed.items()
    }

    store = {
        key: []
        for key in (
            "features_common",
            "features_canonical",
            "unresolved",
            "representation",
            "source",
            "n_agents",
            "peer_count",
            "t",
            "seed",
            "init_kind",
            "dynamics",
            "abs_z1",
            "abs_z2",
            "abs_z3",
            "estimated_eps",
            "d_combined",
            "ood_combined",
            "ood_peer",
            "ood_canonical",
            "ood_native",
        )
    }

    n_list = (9, 16, 17)
    representations = list(packed.keys())
    virtual = (
        "polar_kernel",
        "second_harmonic_kernel",
        "negative_odd_kernel",
        "soft_polar",
    )
    surrogate_names = ("kernel_nn", "hurdle_multinomial", "softmax_stump_boost")

    for representation in representations:
        component = components[representation]
        models = models_by_rep[representation]
        for n_agents in n_list:
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 + seed * 17 + n_agents)
                for init_kind in INIT_KINDS:
                    # A. initial snapshot
                    phases0 = _phases(init_kind, n_agents, rng)
                    _append_snapshot(
                        store,
                        representation=representation,
                        source="A_init",
                        n_agents=n_agents,
                        t=0,
                        seed=seed,
                        init_kind=init_kind,
                        dynamics="none",
                        phases=phases0,
                        component=component,
                    )
                    # B. K=0 drift
                    phases = phases0.copy()
                    omega = rng.normal(0.0, 0.03, size=n_agents)
                    for t in range(1, args.steps + 1):
                        phases = wrap_phase(phases + omega)
                        if t % 5 == 0 or t == args.steps:
                            _append_snapshot(
                                store,
                                representation=representation,
                                source="B_k0_drift",
                                n_agents=n_agents,
                                t=t,
                                seed=seed,
                                init_kind=init_kind,
                                dynamics="k0",
                                phases=phases,
                                component=component,
                            )
                    # C. virtual + surrogate dynamics (short)
                    for dyn in list(virtual) + list(surrogate_names):
                        phases = phases0.copy()
                        omega = rng.normal(0.0, 0.02, size=n_agents)
                        for t in range(1, min(12, args.steps) + 1):
                            xs = []
                            for agent in range(n_agents):
                                histogram = histogram_from_phases(phases, agent)
                                x, _, _ = feature_builder(representation, histogram)
                                xs.append(x)
                            xmat = np.asarray(xs, dtype=np.float64)
                            if dyn in surrogate_names:
                                probs = models[dyn].predict_proba(xmat)
                                probs = np.clip(probs, 1e-12, 1.0)
                                probs = probs / np.sum(probs, axis=1, keepdims=True)
                                actions = np.array(
                                    [
                                        rng.choice([-1.0, 0.0, 1.0], p=probs[i])
                                        for i in range(n_agents)
                                    ]
                                )
                            else:
                                actions = _virtual_actions(dyn, xmat, rng)
                            phases = wrap_phase(phases + omega + 0.12 * actions)
                            if t in {4, 8, 12} or t == min(12, args.steps):
                                _append_snapshot(
                                    store,
                                    representation=representation,
                                    source="C_virtual",
                                    n_agents=n_agents,
                                    t=t,
                                    seed=seed,
                                    init_kind=init_kind,
                                    dynamics=dyn,
                                    phases=phases,
                                    component=component,
                                )

    features_common = np.asarray(store["features_common"], dtype=np.float64)
    features_canonical = np.asarray(store["features_canonical"], dtype=np.float64)
    payload = {
        "features_common": features_common,
        "features_canonical": features_canonical,
        "unresolved": np.asarray(store["unresolved"], dtype=bool),
        "n_agents": np.asarray(store["n_agents"], dtype=np.int16),
        "peer_count": np.asarray(store["peer_count"], dtype=np.int16),
        "t": np.asarray(store["t"], dtype=np.int16),
        "seed": np.asarray(store["seed"], dtype=np.int16),
        "abs_z1": np.asarray(store["abs_z1"], dtype=np.float64),
        "abs_z2": np.asarray(store["abs_z2"], dtype=np.float64),
        "abs_z3": np.asarray(store["abs_z3"], dtype=np.float64),
        "estimated_eps": np.asarray(store["estimated_eps"], dtype=np.float64),
        "d_combined": np.asarray(store["d_combined"], dtype=np.float64),
        "ood_combined": np.asarray(store["ood_combined"], dtype=bool),
        "ood_peer": np.asarray(store["ood_peer"], dtype=bool),
        "ood_canonical": np.asarray(store["ood_canonical"], dtype=bool),
        "ood_native": np.asarray(store["ood_native"], dtype=bool),
        "representation": np.asarray(store["representation"]),
        "source": np.asarray(store["source"]),
        "init_kind": np.asarray(store["init_kind"]),
        "dynamics": np.asarray(store["dynamics"]),
    }
    out_path = out_dir / "candidate_field_bank.npz"
    np.savez_compressed(out_path, **payload)
    meta = {
        "n_fields": int(features_common.shape[0]),
        "n_features_common": int(features_common.shape[1]),
        "sources": sorted(set(store["source"])),
        "n_agents": list(n_list),
        "frac_ood_combined": float(np.mean(payload["ood_combined"])),
        "frac_ood_peer": float(np.mean(payload["ood_peer"])),
        "frac_ood_canonical": float(np.mean(payload["ood_canonical"])),
        "frac_near_zero_abs_z1": float(
            np.mean((payload["abs_z1"] < 0.25) & (np.abs(payload["estimated_eps"]) < 0.02))
        ),
    }
    (out_dir / "candidate_field_bank_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path} n={meta['n_fields']} ood_frac={meta['frac_ood_combined']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
