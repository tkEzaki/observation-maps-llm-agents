"""Peer-stratified centers models for collective-scope (peer in {8,16})."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.centers.distances import circular_emd_pairwise
from analysis.stage3.applicability_domain import N_COMMON
from analysis.stage3.models import (
    GlobalEmpiricalBaseline,
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    counts_to_probs,
)
from circlemap.field_features import COMMON_FEATURE_NAMES

PEER_IDX = COMMON_FEATURE_NAMES.index("peer_count")
ABS_Z1 = COMMON_FEATURE_NAMES.index("abs_z1")
ABS_Z2 = COMMON_FEATURE_NAMES.index("abs_z2")
ANTIPODAL = COMMON_FEATURE_NAMES.index("antipodal_balance")
BIMODALITY = COMMON_FEATURE_NAMES.index("bimodality")
ENTROPY = COMMON_FEATURE_NAMES.index("circular_entropy")

COLLECTIVE_PEERS = (8, 16)


def peer_labels(x: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.rint(x[:, PEER_IDX]).astype(np.int64)


def bin_sparsity(x: NDArray[np.float64]) -> NDArray[np.float64]:
    native = x[:, N_COMMON:]
    return np.mean(native > 1e-12, axis=1)


def _standardize_fit(x: NDArray[np.float64]):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _standardize_apply(x, mean, scale):
    return (x - mean) / scale


@dataclass
class PeerSpecificGlobal:
    """Constant multinomial per peer count."""

    name: str
    probs_by_peer: dict[int, NDArray[np.float64]]
    fallback: NDArray[np.float64]

    @classmethod
    def fit(cls, x: NDArray[np.float64], counts: NDArray[np.float64]) -> "PeerSpecificGlobal":
        peers = peer_labels(x)
        probs: dict[int, NDArray[np.float64]] = {}
        for peer in COLLECTIVE_PEERS:
            mask = peers == peer
            if not np.any(mask):
                continue
            totals = np.sum(counts[mask], axis=0).astype(np.float64) + 0.5
            probs[peer] = totals / np.sum(totals)
        fallback = GlobalEmpiricalBaseline.fit(counts).probs
        return cls(name="peer_specific_global", probs_by_peer=probs, fallback=fallback)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        peers = peer_labels(x)
        out = np.empty((x.shape[0], 3), dtype=np.float64)
        for i, peer in enumerate(peers):
            out[i] = self.probs_by_peer.get(int(peer), self.fallback)
        return out


@dataclass
class CircularEmdKernel:
    """Direct multinomial kernel using circular-EMD on native bins (+ optional canonical)."""

    name: str
    native_train: NDArray[np.float64]
    common_train: NDArray[np.float64] | None
    p_train: NDArray[np.float64]
    common_mean: NDArray[np.float64] | None
    common_scale: NDArray[np.float64] | None
    length_scale_native: float
    length_scale_common: float
    k_neighbors: int
    use_canonical: bool

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        use_canonical: bool = True,
        length_scale_native: float = 0.75,
        length_scale_common: float = 1.5,
        k_neighbors: int = 32,
        alpha: float = 0.5,
        name: str = "circular_emd_kernel",
    ) -> "CircularEmdKernel":
        native = x[:, N_COMMON:].astype(np.float64)
        common = x[:, :N_COMMON].astype(np.float64) if use_canonical else None
        mean = scale = None
        common_train = None
        if use_canonical:
            common_train, mean, scale = _standardize_fit(common)
        return cls(
            name=name,
            native_train=native,
            common_train=common_train,
            p_train=counts_to_probs(counts, alpha=alpha),
            common_mean=mean,
            common_scale=scale,
            length_scale_native=float(length_scale_native),
            length_scale_common=float(length_scale_common),
            k_neighbors=int(k_neighbors),
            use_canonical=bool(use_canonical),
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        native = x[:, N_COMMON:].astype(np.float64)
        n = x.shape[0]
        out = np.empty((n, self.p_train.shape[1]), dtype=np.float64)
        k = min(self.k_neighbors, self.native_train.shape[0])
        inv_n = 1.0 / (self.length_scale_native**2)
        inv_c = 1.0 / (self.length_scale_common**2) if self.use_canonical else 0.0
        common = None
        if self.use_canonical:
            common = _standardize_apply(
                x[:, :N_COMMON], self.common_mean, self.common_scale
            )
        chunk = 64
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            d_n = circular_emd_pairwise(native[start:stop], self.native_train)
            dist2 = (d_n**2) * inv_n
            if self.use_canonical:
                diff = common[start:stop, None, :] - self.common_train[None, :, :]
                dist2 = dist2 + np.sum(diff * diff, axis=-1) * inv_c
            if k < self.native_train.shape[0]:
                idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
                rows = np.arange(stop - start)[:, None]
                dist2_k = dist2[rows, idx]
                p_k = self.p_train[idx]
            else:
                dist2_k = dist2
                p_k = np.repeat(self.p_train[None, :, :], stop - start, axis=0)
            weights = np.exp(-0.5 * dist2_k)
            weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
            out[start:stop] = np.sum(weights[:, :, None] * p_k, axis=1)
        return out


@dataclass
class PeerStratifiedModel:
    """Route predictions to peer-specific experts; no cross-peer neighbors."""

    name: str
    experts: dict[int, object]
    fallback: object

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        peers = peer_labels(x)
        out = np.empty((x.shape[0], 3), dtype=np.float64)
        for peer in np.unique(peers):
            mask = peers == peer
            expert = self.experts.get(int(peer), self.fallback)
            out[mask] = expert.predict_proba(x[mask])
        return out


def _fit_peer_map(
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
    factory,
) -> tuple[dict[int, object], object]:
    peers = peer_labels(x)
    experts: dict[int, object] = {}
    for peer in COLLECTIVE_PEERS:
        mask = peers == peer
        if int(np.sum(mask)) < 5:
            continue
        experts[peer] = factory(x[mask], counts[mask])
    fallback = factory(x, counts)
    return experts, fallback


@dataclass
class HierarchicalPeerKernel:
    """Peer-specific EMD kernel shrunk toward a pooled peer{8,16} kernel."""

    name: str
    peer_model: PeerStratifiedModel
    pooled: CircularEmdKernel
    shrink: float

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_peer = self.peer_model.predict_proba(x)
        p_pool = self.pooled.predict_proba(x)
        w = float(np.clip(self.shrink, 0.0, 1.0))
        return (1.0 - w) * p_peer + w * p_pool


@dataclass
class DescriptorCollapseMixture:
    """Soft abstention/active mixture gated by descriptors (no field IDs)."""

    name: str
    abstain: CircularEmdKernel
    active: CircularEmdKernel
    gate_mean: NDArray[np.float64]
    gate_scale: NDArray[np.float64]
    gate_weights: NDArray[np.float64]
    gate_bias: float

    @staticmethod
    def _gate_x(x: NDArray[np.float64]) -> NDArray[np.float64]:
        peers = peer_labels(x).astype(np.float64)
        return np.column_stack(
            [
                peers / 16.0,
                bin_sparsity(x),
                x[:, ABS_Z1],
                x[:, ABS_Z2],
                x[:, ANTIPODAL],
                x[:, BIMODALITY],
                x[:, ENTROPY],
            ]
        )

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        shrink_gate: float = 1.0,
    ) -> "DescriptorCollapseMixture":
        empir = counts_to_probs(counts, alpha=0.0)
        stay = empir[:, 1]
        # Pseudo-label: high-stay rows as abstention regime.
        y = (stay >= 0.55).astype(np.float64)
        gx = cls._gate_x(x)
        gs, mean, scale = _standardize_fit(gx)
        # Ridge logistic on gate features.
        design = np.concatenate([np.ones((gs.shape[0], 1)), gs], axis=1)
        # Newton-ish IRLS few steps
        beta = np.zeros(design.shape[1], dtype=np.float64)
        for _ in range(12):
            logits = design @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            w = np.clip(p * (1.0 - p), 1e-4, None)
            grad = design.T @ (p - y) + 1e-2 * beta
            hess = design.T @ (design * w[:, None]) + 1e-2 * np.eye(design.shape[1])
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            beta = beta - step
            if float(np.max(np.abs(step))) < 1e-6:
                break
        # Fit experts on soft-weighted subsets (hard threshold for stability).
        abs_mask = stay >= 0.55
        act_mask = stay < 0.45
        if int(np.sum(abs_mask)) < 20:
            abs_mask = stay >= np.median(stay)
        if int(np.sum(act_mask)) < 20:
            act_mask = ~abs_mask
        abstain = CircularEmdKernel.fit(
            x[abs_mask],
            counts[abs_mask],
            use_canonical=True,
            name="collapse_abstain",
            k_neighbors=24,
        )
        active = CircularEmdKernel.fit(
            x[act_mask],
            counts[act_mask],
            use_canonical=True,
            name="collapse_active",
            k_neighbors=24,
        )
        return cls(
            name="descriptor_collapse_mixture",
            abstain=abstain,
            active=active,
            gate_mean=mean,
            gate_scale=scale,
            gate_weights=beta[1:] * float(shrink_gate),
            gate_bias=float(beta[0]),
        )

    def collapse_weight(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        gs = _standardize_apply(self._gate_x(x), self.gate_mean, self.gate_scale)
        logits = self.gate_bias + gs @ self.gate_weights
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        w = self.collapse_weight(x)[:, None]
        return w * self.abstain.predict_proba(x) + (1.0 - w) * self.active.predict_proba(x)


def fit_collective_candidates(
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
) -> dict[str, object]:
    """Fit the CENT-4 revision candidate set on peer{8,16} rows only."""

    def emd_only(xt, ct):
        return CircularEmdKernel.fit(
            xt, ct, use_canonical=False, name="emd_only", k_neighbors=24
        )

    def emd_can(xt, ct):
        return CircularEmdKernel.fit(
            xt, ct, use_canonical=True, name="emd_canonical", k_neighbors=32
        )

    def hurdle(xt, ct):
        return KernelHurdleMultinomial.fit(xt, ct, k_neighbors=24)

    peer_emd_experts, peer_emd_fb = _fit_peer_map(x, counts, emd_only)
    peer_can_experts, peer_can_fb = _fit_peer_map(x, counts, emd_can)
    peer_hurdle_experts, peer_hurdle_fb = _fit_peer_map(x, counts, hurdle)

    peer_emd = PeerStratifiedModel(
        name="peer_specific_circular_emd",
        experts=peer_emd_experts,
        fallback=peer_emd_fb,
    )
    pooled = CircularEmdKernel.fit(
        x, counts, use_canonical=True, name="pooled_emd_canonical", k_neighbors=32
    )
    models: dict[str, object] = {
        "pooled_global": GlobalEmpiricalBaseline.fit(counts),
        "peer_specific_global": PeerSpecificGlobal.fit(x, counts),
        "peer_specific_circular_emd": peer_emd,
        "peer_specific_canonical_emd": PeerStratifiedModel(
            name="peer_specific_canonical_emd",
            experts=peer_can_experts,
            fallback=peer_can_fb,
        ),
        "hierarchical_emd_kernel": HierarchicalPeerKernel(
            name="hierarchical_emd_kernel",
            peer_model=peer_emd,
            pooled=pooled,
            shrink=0.25,
        ),
        "peer_specific_activity_direction": PeerStratifiedModel(
            name="peer_specific_activity_direction",
            experts=peer_hurdle_experts,
            fallback=peer_hurdle_fb,
        ),
        "descriptor_collapse_mixture": DescriptorCollapseMixture.fit(x, counts),
        # Reference: old pooled canonical NN (not stratified)
        "pooled_kernel_nn": KernelNeighborBaseline.fit(x, counts, k_neighbors=32),
    }
    return models
