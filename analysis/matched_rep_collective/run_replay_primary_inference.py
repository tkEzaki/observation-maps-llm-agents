"""Formal field-blocked primary inference for matched 3×3 replay.

Inference unit: 48 physical fields (not 4608 calls).
Primary global tests:
  1. target main effect (operator)
  2. source main effect (endogenous manifold)
  3. source × target interaction (feedback)

Pairwise TV / ΔA / Δa0 and stratum splits are effect-size decompositions
(Holm-adjusted only if reported as secondary tests).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_DIR = (
    ROOT
    / "runs"
    / "matched_rep_collective_replay"
    / "matched-rep-collective-3x3-replay-v0.1_1e976b79b71d"
    / "20260724T114515Z"
)
FIELDS_PATH = ROOT / "analysis" / "matched_rep_collective" / "replay_fields_v0_1.json"
FREEZE_PATH = (
    ROOT / "analysis" / "matched_rep_collective" / "replay_hash_freeze_v0_1.json"
)
OUT = ROOT / "analysis" / "matched_rep_collective" / "replay_primary"
FIG = OUT / "figures"

TARGETS = (
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
)
TARGET_SHORT = {
    "moments_m1_m3": "M",
    "centers_24_standard": "C",
    "intervals_24_decimal6": "I",
}
PAIRS = (
    ("moments_m1_m3", "centers_24_standard"),
    ("moments_m1_m3", "intervals_24_decimal6"),
    ("centers_24_standard", "intervals_24_decimal6"),
)
N_PERM = 5000
N_BOOT = 5000

# Backend prices in force for this acquisition, in USD per token.  These two
# constants reproduce the cost the runner recorded for the session subset to
# within 1e-6, which the inference asserts at run time.
PRICE_USD_PER_INPUT_TOKEN = 0.75e-6
PRICE_USD_PER_OUTPUT_TOKEN = 4.50e-6
RNG = np.random.default_rng(20260724)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def _counts_to_p(n: np.ndarray) -> np.ndarray:
    s = float(np.sum(n))
    if s <= 0:
        return np.full(3, np.nan)
    return n.astype(float) / s


def _stats_from_counts(n: np.ndarray) -> dict:
    p = _counts_to_p(n)
    # actions: -1,0,+1 corresponding to retard,stay,advance
    a0 = float((-1) * p[0] + 0 * p[1] + 1 * p[2]) if np.all(np.isfinite(p)) else float("nan")
    A = float(p[0] + p[2]) if np.all(np.isfinite(p)) else float("nan")
    return {"n": n.astype(int).tolist(), "p": p.tolist(), "a0": a0, "A": A}


def load_data() -> tuple[list[dict], list[dict], dict]:
    fields = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))["fields"]
    rows = []
    with (RUN_DIR / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    session = json.loads((RUN_DIR / "session_summary.json").read_text(encoding="utf-8"))
    return fields, rows, session


def aggregate(fields: list[dict], rows: list[dict]) -> dict:
    """Per-field counts by target and by (target, block); keep raw actions."""
    by_ft: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=int))
    by_ftb: dict[tuple[str, str, int], np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=int)
    )
    # field -> list of (target, action_idx, block)
    raw_by_field: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for r in rows:
        if not r.get("valid"):
            continue
        v = int(r["action_value"])
        idx = {-1: 0, 0: 1, 1: 2}[v]
        fid = r["field_id"]
        t = r["target_representation"]
        b = int(r["acquisition_block"])
        by_ft[(fid, t)][idx] += 1
        by_ftb[(fid, t, b)][idx] += 1
        raw_by_field[fid].append((t, idx, b))

    panel = []
    for f in fields:
        fid = f["field_id"]
        entry = {
            "field_id": fid,
            "source": f["source_representation"],
            "stratum": f["stratum"],
            "K": float(f["K"]),
            "r1": float(f["r1"]),
            "r2": float(f["r2"]),
            "targets": {},
            "raw": raw_by_field[fid],
        }
        blocks_raw = {}
        for t in TARGETS:
            entry["targets"][t] = _stats_from_counts(by_ft[(fid, t)])
            for b in (0, 1):
                blocks_raw[(t, b)] = _stats_from_counts(by_ftb[(fid, t, b)])
        entry["blocks"] = {
            f"{t}|b{b}": blocks_raw[(t, b)] for t in TARGETS for b in (0, 1)
        }
        panel.append(entry)
    return {"panel": panel}


def _panel_from_raw(panel_template: list[dict]) -> list[dict]:
    """Rebuild target/block aggregates from each field's raw tuples."""
    out = []
    for fr in panel_template:
        by_t = {t: np.zeros(3, dtype=int) for t in TARGETS}
        by_tb = {(t, b): np.zeros(3, dtype=int) for t in TARGETS for b in (0, 1)}
        for t, idx, b in fr["raw"]:
            by_t[t][idx] += 1
            by_tb[(t, b)][idx] += 1
        nr = {
            "field_id": fr["field_id"],
            "source": fr["source"],
            "stratum": fr["stratum"],
            "K": fr["K"],
            "r1": fr["r1"],
            "r2": fr["r2"],
            "raw": fr["raw"],
            "targets": {t: _stats_from_counts(by_t[t]) for t in TARGETS},
            "blocks": {
                f"{t}|b{b}": _stats_from_counts(by_tb[(t, b)])
                for t in TARGETS
                for b in (0, 1)
            },
        }
        out.append(nr)
    return out


def permute_target_labels_on_responses(panel: list[dict]) -> list[dict]:
    """Within each field, shuffle target labels across the 96 responses."""
    out = []
    for fr in panel:
        raw = list(fr["raw"])
        labels = [t for t, _, _ in raw]
        RNG.shuffle(labels)
        new_raw = [(labels[i], raw[i][1], raw[i][2]) for i in range(len(raw))]
        nr = dict(fr)
        nr["raw"] = new_raw
        out.append(nr)
    return _panel_from_raw(out)


def multinomial_deviance(panel: list[dict]) -> float:
    """Sum over fields of LR deviance: separate targets vs pooled multinomial."""
    total = 0.0
    for fr in panel:
        counts = np.stack(
            [np.asarray(fr["targets"][t]["n"], dtype=float) for t in TARGETS]
        )  # 3 x 3
        pooled = counts.sum(axis=0)
        n_tot = pooled.sum()
        if n_tot <= 0:
            continue
        p_pool = pooled / n_tot
        # saturated within each target
        ll_sat = 0.0
        ll_null = 0.0
        for r in range(3):
            n_r = counts[r].sum()
            if n_r <= 0:
                continue
            p_r = counts[r] / n_r
            for c in range(3):
                if counts[r, c] > 0:
                    ll_sat += counts[r, c] * np.log(p_r[c])
                if p_pool[c] > 0 and counts[r, c] > 0:
                    ll_null += counts[r, c] * np.log(p_pool[c])
        total += 2.0 * (ll_sat - ll_null)
    return float(total)


def global_target_test(panel: list[dict]) -> dict:
    # Primary: response-level field-blocked permutation of target labels,
    # statistic = mean pairwise TV (and report multinomial deviance too).
    obs_tv = target_perm_stat(panel)
    obs_dev = multinomial_deviance(panel)
    null_tv = np.empty(N_PERM)
    null_dev = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = permute_target_labels_on_responses(panel)
        null_tv[i] = target_perm_stat(perm)
        null_dev[i] = multinomial_deviance(perm)
    p_tv = float((np.sum(null_tv >= obs_tv) + 1) / (N_PERM + 1))
    p_dev = float((np.sum(null_dev >= obs_dev) + 1) / (N_PERM + 1))

    # Noise-floor paired field test: mean(between - within) > 0 via sign-flip
    diffs = []
    for fr in panel:
        bet = []
        wit = []
        ps = [np.asarray(fr["targets"][t]["p"], dtype=float) for t in TARGETS]
        for i in range(3):
            for j in range(i + 1, 3):
                bet.append(_tv(ps[i], ps[j]))
        for t in TARGETS:
            p0 = np.asarray(fr["blocks"][f"{t}|b0"]["p"], dtype=float)
            p1 = np.asarray(fr["blocks"][f"{t}|b1"]["p"], dtype=float)
            wit.append(_tv(p0, p1))
        diffs.append(float(np.mean(bet) - np.mean(wit)))
    diffs = np.asarray(diffs, dtype=float)
    obs_diff = float(np.mean(diffs))
    null_diff = np.empty(N_PERM)
    for i in range(N_PERM):
        signs = RNG.choice([-1.0, 1.0], size=len(diffs))
        null_diff[i] = float(np.mean(signs * diffs))
    p_noise = float((np.sum(null_diff >= obs_diff) + 1) / (N_PERM + 1))

    pair_boot = {}
    tvs = field_tvs(panel)
    for k, vals in tvs["pairwise_field_tv"].items():
        pair_boot[k] = cluster_bootstrap_mean(np.asarray(vals, dtype=float))
    return {
        "test": "field_blocked_response_permutation_mean_pairwise_TV",
        "n_perm": N_PERM,
        "observed_mean_pairwise_TV": obs_tv,
        "null_mean_TV": float(np.mean(null_tv)),
        "null_p95_TV": float(np.quantile(null_tv, 0.95)),
        "p_one_sided_TV": p_tv,
        "secondary_multinomial_deviance": {
            "observed": obs_dev,
            "null_mean": float(np.mean(null_dev)),
            "p_one_sided": p_dev,
        },
        "noise_floor_paired_test": {
            "observed_mean_between_minus_within": obs_diff,
            "p_one_sided": p_noise,
            "n_fields": int(len(diffs)),
        },
        "pairwise_bootstrap": pair_boot,
        "noise_floor": {
            "mean_between_target_TV": tvs["mean_between_target_tv"],
            "mean_within_target_block_TV": tvs["mean_within_block_tv"],
            "ratio": tvs["ratio_between_over_within"],
            "between_boot": cluster_bootstrap_mean(
                np.asarray(tvs["between_target_tv_all"], dtype=float),
                clusters=np.asarray(tvs["between_target_field_ix"]),
            ),
            "within_boot": cluster_bootstrap_mean(
                np.asarray(tvs["within_target_block_tv"], dtype=float),
                clusters=np.asarray(tvs["within_target_field_ix"]),
            ),
        },
        "field_tvs": tvs,
        # primary p used for verdict
        "p_one_sided": p_tv,
    }


def permute_targets_within_field(panel: list[dict]) -> list[dict]:
    """Deprecated alias retained for clarity — use response-level permute."""
    return permute_target_labels_on_responses(panel)


def field_tvs(panel: list[dict]) -> dict:
    out = {f"{a}__vs__{b}": [] for a, b in PAIRS}
    within_block = []
    between_target = []
    between_field_ix = []
    within_field_ix = []
    for field_ix, fr in enumerate(panel):
        for a, b in PAIRS:
            pa = np.asarray(fr["targets"][a]["p"], dtype=float)
            pb = np.asarray(fr["targets"][b]["p"], dtype=float)
            d = _tv(pa, pb)
            out[f"{a}__vs__{b}"].append(d)
            between_target.append(d)
            between_field_ix.append(field_ix)
        for t in TARGETS:
            p0 = np.asarray(fr["blocks"][f"{t}|b0"]["p"], dtype=float)
            p1 = np.asarray(fr["blocks"][f"{t}|b1"]["p"], dtype=float)
            within_block.append(_tv(p0, p1))
            within_field_ix.append(field_ix)
    return {
        "pairwise_field_tv": out,
        "mean_pairwise": {k: float(np.mean(v)) for k, v in out.items()},
        "between_target_tv_all": between_target,
        "within_target_block_tv": within_block,
        "between_target_field_ix": between_field_ix,
        "within_target_field_ix": within_field_ix,
        "mean_between_target_tv": float(np.mean(between_target)),
        "mean_within_block_tv": float(np.mean(within_block)),
        "ratio_between_over_within": float(
            np.mean(between_target) / max(np.mean(within_block), 1e-12)
        ),
    }


def _trace_token_totals(path: Path) -> dict:
    """Input and output token totals over every row of the acquisition trace."""
    ti = to = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            ti += int(row.get("input_tokens") or 0)
            to += int(row.get("output_tokens") or 0)
    return {"input": ti, "output": to}


def cluster_bootstrap_mean(
    values: np.ndarray, n_boot: int = N_BOOT, clusters: np.ndarray | None = None
) -> dict:
    """Percentile bootstrap of the mean, resampling clusters not elements.

    ``clusters`` gives the cluster label of each element (the physical field).
    When it is omitted the values are already one per field and each element is
    its own cluster.  When several values belong to one field, as they do for
    the noise-floor lists, the whole field must be resampled together: the
    values inside a field are not independent, and resampling them singly makes
    the interval too narrow.
    """
    values = np.asarray(values, dtype=float)
    if clusters is None:
        groups = [np.array([i]) for i in range(len(values))]
    else:
        clusters = np.asarray(clusters)
        groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    n_clusters = len(groups)
    # Dedicated stream: the cluster-aware path draws a different number of
    # variates than the flat path, and the module RNG is shared with the
    # permutation tests.  Using a separate generator keeps every other reported
    # quantity bit-identical when this function changes.
    rng = np.random.default_rng(20260724)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        means[i] = float(np.mean(values[np.concatenate([groups[j] for j in pick])]))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(lo), float(hi)],
        "n_fields": int(n_clusters),
        "n_values": int(len(values)),
        "n_clusters": int(n_clusters),
        "cluster_unit": "physical_field" if clusters is not None else "value",
    }


def target_perm_stat(panel: list[dict]) -> float:
    """Mean pairwise TV across fields (observed target assignment)."""
    tvs = []
    for fr in panel:
        ps = [np.asarray(fr["targets"][t]["p"], dtype=float) for t in TARGETS]
        for i in range(3):
            for j in range(i + 1, 3):
                tvs.append(_tv(ps[i], ps[j]))
    return float(np.mean(tvs))


def source_field_means(panel: list[dict]) -> dict:
    """Average a0/A/p over targets within each field, then by source."""
    by_source = {s: {"a0": [], "A": [], "p": []} for s in TARGETS}
    for fr in panel:
        a0s = [fr["targets"][t]["a0"] for t in TARGETS]
        As = [fr["targets"][t]["A"] for t in TARGETS]
        ps = [np.asarray(fr["targets"][t]["p"], dtype=float) for t in TARGETS]
        by_source[fr["source"]]["a0"].append(float(np.mean(a0s)))
        by_source[fr["source"]]["A"].append(float(np.mean(As)))
        by_source[fr["source"]]["p"].append(np.mean(np.stack(ps), axis=0))
    out = {}
    for s in TARGETS:
        ps = np.stack(by_source[s]["p"]) if by_source[s]["p"] else np.empty((0, 3))
        out[s] = {
            "n_fields": len(by_source[s]["a0"]),
            "mean_a0": float(np.mean(by_source[s]["a0"])),
            "mean_A": float(np.mean(by_source[s]["A"])),
            "mean_p": np.mean(ps, axis=0).tolist() if len(ps) else [np.nan] * 3,
        }
    return out


def source_dispersion(panel: list[dict]) -> float:
    """Between-source SSD of mean a0 (primary scalar for permutation)."""
    means = source_field_means(panel)
    a0s = np.array([means[s]["mean_a0"] for s in TARGETS])
    return float(np.sum((a0s - np.mean(a0s)) ** 2))


def blocked_source_perm(panel: list[dict]) -> list[dict]:
    """Permute source labels within stratum (K as soft covariate via stratum×K cells when size>=2)."""
    # Group indices by (stratum, K)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, fr in enumerate(panel):
        groups[(fr["stratum"], fr["K"])].append(i)
    # If a cell has size 1, fall back to stratum-only pool for that index
    stratum_groups: dict[str, list[int]] = defaultdict(list)
    for i, fr in enumerate(panel):
        stratum_groups[fr["stratum"]].append(i)

    new_sources = [fr["source"] for fr in panel]
    used = set()
    for key, idxs in groups.items():
        if len(idxs) >= 2:
            labels = [panel[i]["source"] for i in idxs]
            RNG.shuffle(labels)
            for i, lab in zip(idxs, labels):
                new_sources[i] = lab
                used.add(i)
    # Remaining size-1 cells: shuffle within stratum
    remaining_by_stratum: dict[str, list[int]] = defaultdict(list)
    for i, fr in enumerate(panel):
        if i not in used:
            remaining_by_stratum[fr["stratum"]].append(i)
    for st, idxs in remaining_by_stratum.items():
        labels = [panel[i]["source"] for i in idxs]
        if len(labels) >= 2:
            RNG.shuffle(labels)
            for i, lab in zip(idxs, labels):
                new_sources[i] = lab
        # size 1: leave as is (conservative)

    out = []
    for fr, src in zip(panel, new_sources):
        nr = dict(fr)
        nr["source"] = src
        out.append(nr)
    return out


def global_source_test(panel: list[dict]) -> dict:
    obs = source_dispersion(panel)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = source_dispersion(blocked_source_perm(panel))
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    means = source_field_means(panel)
    # Bootstrap CI for intervals vs moments mean a0 difference (field-level)
    mom = []
    cen = []
    itv = []
    for fr in panel:
        a0 = float(np.mean([fr["targets"][t]["a0"] for t in TARGETS]))
        if fr["source"] == "moments_m1_m3":
            mom.append(a0)
        elif fr["source"] == "centers_24_standard":
            cen.append(a0)
        else:
            itv.append(a0)
    # field-resample within source for contrast CI
    def boot_diff(a, b):
        diffs = []
        a = np.asarray(a)
        b = np.asarray(b)
        for _ in range(N_BOOT):
            aa = a[RNG.integers(0, len(a), size=len(a))]
            bb = b[RNG.integers(0, len(b), size=len(b))]
            diffs.append(np.mean(aa) - np.mean(bb))
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        return {
            "diff": float(np.mean(a) - np.mean(b)),
            "ci95": [float(lo), float(hi)],
        }

    return {
        "test": "stratum_x_K_blocked_permutation_SSD_mean_a0",
        "n_perm": N_PERM,
        "observed_SSD_mean_a0": obs,
        "null_mean": float(np.mean(null)),
        "p_one_sided": p,
        "source_means": means,
        "contrasts_a0": {
            "intervals_minus_moments": boot_diff(itv, mom),
            "intervals_minus_centers": boot_diff(itv, cen),
            "centers_minus_moments": boot_diff(cen, mom),
        },
        "adjustment": "permute source within (stratum,K) cells; size-1 cells fall back to stratum",
    }


def interaction_stat(panel: list[dict]) -> float:
    """SSD across sources of mean ΔA^{M-I} (primary interaction scalar)."""
    by_s = {s: [] for s in TARGETS}
    for fr in panel:
        dA = fr["targets"]["moments_m1_m3"]["A"] - fr["targets"]["intervals_24_decimal6"]["A"]
        by_s[fr["source"]].append(dA)
    means = np.array([np.mean(by_s[s]) for s in TARGETS])
    return float(np.sum((means - np.mean(means)) ** 2))


def interaction_contrasts(panel: list[dict]) -> list[dict]:
    rows = []
    for fr in panel:
        row = {
            "field_id": fr["field_id"],
            "source": fr["source"],
            "stratum": fr["stratum"],
            "K": fr["K"],
        }
        for a, b in PAIRS:
            row[f"dTV_{TARGET_SHORT[a]}{TARGET_SHORT[b]}"] = _tv(
                np.asarray(fr["targets"][a]["p"]),
                np.asarray(fr["targets"][b]["p"]),
            )
            row[f"dA_{TARGET_SHORT[a]}{TARGET_SHORT[b]}"] = (
                fr["targets"][a]["A"] - fr["targets"][b]["A"]
            )
            row[f"da0_{TARGET_SHORT[a]}{TARGET_SHORT[b]}"] = (
                fr["targets"][a]["a0"] - fr["targets"][b]["a0"]
            )
        rows.append(row)
    return rows


def global_interaction_test(panel: list[dict]) -> dict:
    obs = interaction_stat(panel)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = interaction_stat(blocked_source_perm(panel))
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    contrasts = interaction_contrasts(panel)
    # mean ΔA M-I by source + bootstrap
    by_s = {s: [] for s in TARGETS}
    for c in contrasts:
        by_s[c["source"]].append(c["dA_MI"])
    summary = {}
    for s in TARGETS:
        summary[s] = cluster_bootstrap_mean(np.asarray(by_s[s], dtype=float))
    return {
        "test": "stratum_x_K_blocked_permutation_SSD_mean_deltaA_MI",
        "n_perm": N_PERM,
        "observed_SSD_mean_deltaA_MI": obs,
        "null_mean": float(np.mean(null)),
        "p_one_sided": p,
        "deltaA_MI_by_source": summary,
        "contrasts": contrasts,
    }


def matrix_3x3(panel: list[dict]) -> dict:
    mat_A = {s: {} for s in TARGETS}
    mat_a0 = {s: {} for s in TARGETS}
    mat_p = {s: {} for s in TARGETS}
    for s in TARGETS:
        subset = [fr for fr in panel if fr["source"] == s]
        for t in TARGETS:
            As = [fr["targets"][t]["A"] for fr in subset]
            a0s = [fr["targets"][t]["a0"] for fr in subset]
            ps = np.stack([fr["targets"][t]["p"] for fr in subset])
            mat_A[s][t] = float(np.mean(As))
            mat_a0[s][t] = float(np.mean(a0s))
            mat_p[s][t] = np.mean(ps, axis=0).tolist()
    return {"A": mat_A, "a0": mat_a0, "p": mat_p}


def make_figures(panel: list[dict], target_res: dict, ix_res: dict, mat: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    tvs = target_res["field_tvs"]

    # Fig 6a
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xpos = [0, 1, 2]
    labels = ["M–C", "M–I", "C–I"]
    keys = [f"{a}__vs__{b}" for a, b in PAIRS]
    for i, k in enumerate(keys):
        vals = np.asarray(tvs["pairwise_field_tv"][k])
        ax.scatter(
            np.full_like(vals, xpos[i]) + RNG.normal(0, 0.04, size=len(vals)),
            vals,
            s=18,
            alpha=0.55,
            color="#2c5f7c",
            zorder=2,
        )
        boot = target_res["pairwise_bootstrap"][k]
        ax.errorbar(
            xpos[i],
            boot["mean"],
            yerr=[[boot["mean"] - boot["ci95"][0]], [boot["ci95"][1] - boot["mean"]]],
            fmt="o",
            color="#c44e52",
            ms=7,
            lw=1.5,
            capsize=4,
            zorder=3,
        )
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Total variation $d_{\mathrm{TV}}$")
    ax.set_title("Figure 6a — Target-pair TV across 48 fields")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6a_target_pair_tv.png", dpi=180)
    plt.close(fig)

    # Fig 6b heatmaps A and a0
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, key, title, cmap, vmin, vmax in [
        (axes[0], "A", r"Activity $A$", "YlOrRd", 0.7, 1.0),
        (axes[1], "a0", r"Signed mean $a_0$", "coolwarm", -0.5, 0.5),
    ]:
        data = np.array([[mat[key][s][t] for t in TARGETS] for s in TARGETS])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([TARGET_SHORT[t] for t in TARGETS])
        ax.set_yticklabels([TARGET_SHORT[s] for s in TARGETS])
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 6b — 3×3 action matrix (source × target)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig6b_action_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Fig 6c interaction contrasts ΔA M-I and Δa0 M-I by source
    contrasts = ix_res["contrasts"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, metric, title in [
        (axes[0], "dA_MI", r"$\Delta A^{M-I}$"),
        (axes[1], "da0_MI", r"$\Delta a_0^{M-I}$"),
    ]:
        for i, s in enumerate(TARGETS):
            vals = [c[metric] for c in contrasts if c["source"] == s]
            ax.scatter(
                np.full(len(vals), i) + RNG.normal(0, 0.05, size=len(vals)),
                vals,
                s=22,
                alpha=0.65,
                color="#2c5f7c",
            )
            ax.hlines(np.mean(vals), i - 0.25, i + 0.25, colors="#c44e52", lw=2)
        ax.axhline(0, color="0.5", lw=0.8, ls="--")
        ax.set_xticks(range(3))
        ax.set_xticklabels([TARGET_SHORT[s] for s in TARGETS])
        ax.set_xlabel("Source")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Figure 6c — Interaction contrasts by source", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig6c_interaction_contrasts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Fig 6d noise floor
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bt = np.asarray(tvs["between_target_tv_all"])
    wt = np.asarray(tvs["within_target_block_tv"])
    parts = ax.violinplot([wt, bt], positions=[0, 1], showmeans=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor("#7aa0b5")
        b.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "Within-target\nbetween-block TV",
            "Between-target\nTV",
        ]
    )
    ax.set_ylabel(r"$d_{\mathrm{TV}}$")
    ax.set_title("Figure 6d — Operator separation vs acquisition drift")
    ax.text(
        0.98,
        0.95,
        f"ratio = {tvs['ratio_between_over_within']:.1f}×",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6d_noise_floor.png", dpi=180)
    plt.close(fig)


def mechanistic_verdict(target_p: float, source_p: float, ix_p: float) -> dict:
    # Conservative alpha 0.05 for three primary globals (report raw; note multiplicity)
    target_ok = target_p < 0.05
    source_ok = source_p < 0.05
    ix_ok = ix_p < 0.05
    if target_ok and source_ok and ix_ok:
        claim = (
            "Observation representations change both microscopic response operators "
            "and the endogenous collective states on which those operators act, "
            "producing distinct collective phases through representation-dependent feedback."
        )
        feedback = "supported"
    elif target_ok and source_ok:
        claim = (
            "Observation representations change microscopic operators and endogenous "
            "field distributions; feedback interaction is not established at primary alpha."
        )
        feedback = "not_established"
    elif target_ok:
        claim = (
            "Same physical fields elicit representation-dependent operators "
            "(strong target main effect); endogenous source and feedback remain secondary."
        )
        feedback = "not_established"
    else:
        claim = "Primary target operator effect not established; revisit panel."
        feedback = "not_established"
    return {
        "operator_effect": "supported" if target_ok else "not_supported",
        "endogenous_manifold_effect": "supported" if source_ok else "suggested_ns_at_0.05",
        "representation_dependent_feedback": feedback,
        "claim_text": claim,
        "alpha_primary": 0.05,
        "multiplicity_note": (
            "Three primary global tests; pairwise decompositions are effect sizes "
            "(Holm only if elevated to secondary tests)."
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fields, rows, session = load_data()
    assert len(rows) == 4608
    assert sum(1 for r in rows if r.get("valid")) == 4608

    agg = aggregate(fields, rows)
    panel = agg["panel"]

    print("Running target global test...", flush=True)
    target_res = global_target_test(panel)
    print("Running source global test...", flush=True)
    source_res = global_source_test(panel)
    print("Running interaction global test...", flush=True)
    ix_res = global_interaction_test(panel)
    mat = matrix_3x3(panel)

    print("Making figures...", flush=True)
    make_figures(panel, target_res, ix_res, mat)

    verdict = mechanistic_verdict(
        target_res["p_one_sided"],
        source_res["p_one_sided"],
        ix_res["p_one_sided"],
    )
    # The three primary p values are carried inside the verdict block as well as
    # inside the per-test records, because the figures read them from here.
    verdict["source_p"] = source_res["p_one_sided"]
    verdict["interaction_p"] = ix_res["p_one_sided"]
    verdict["target_p"] = target_res["p_one_sided"]

    trace_tokens = _trace_token_totals(RUN_DIR / "trace.jsonl")
    trace_cost = (
        PRICE_USD_PER_INPUT_TOKEN * trace_tokens["input"]
        + PRICE_USD_PER_OUTPUT_TOKEN * trace_tokens["output"]
    )
    # Guard: the same two rates must reproduce the cost the runner recorded for
    # the session subset.  If a price constant is ever wrong this fails loudly
    # instead of writing a silently incorrect cost.
    session_cost = (
        PRICE_USD_PER_INPUT_TOKEN * session["actual_input_tokens_session"]
        + PRICE_USD_PER_OUTPUT_TOKEN * session["actual_output_tokens_session"]
    )
    if abs(session_cost - session["actual_cost_usd_session"]) > 1e-6:
        raise RuntimeError(
            "price constants do not reproduce the recorded session cost: "
            f"{session_cost} vs {session['actual_cost_usd_session']}"
        )

    # Drop heavy field lists from decision but keep contrasts CSV-like
    decision = {
        "status": "PAID_COMPLETE_PRIMARY_INFERENCE_V0_1",
        "inference_unit": "48_physical_fields",
        "acquisition": {
            "run_dir": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
            "n_calls": 4608,
            "n_valid": 4608,
            "n_rescued": 0,
            "n_unrecoverable": 0,
            "valid_rate": 1.0,
            "actual_cost_usd_session": session.get("actual_cost_usd_session"),
            "elapsed_seconds": session.get("elapsed_seconds"),
            "cost_ceiling_usd": 12.0,
            "preflight_tasks": 12,
            "preflight_note": "in-budget checkpoint; included in 4608 total",
            "actual_cost_usd_total_trace": round(trace_cost, 6),
            "actual_input_tokens_total": trace_tokens["input"],
            "actual_output_tokens_total": trace_tokens["output"],
        },
        "results_doc": "docs/MATCHED_REP_COLLECTIVE_REPLAY_RESULTS.md",
        "hashes": {
            "fields_sha256": _sha_file(FIELDS_PATH),
            "freeze_sha256": _sha_file(FREEZE_PATH),
            "trace_sha256": _sha_file(RUN_DIR / "trace.jsonl"),
            "protocol_sha256": _sha_file(RUN_DIR / "protocol.json"),
            "resolved_config_sha256": _sha_file(RUN_DIR / "resolved_config.json"),
        },
        "primary_global_tests": {
            "1_target_main_effect": {
                k: target_res[k]
                for k in (
                    "test",
                    "n_perm",
                    "observed_mean_pairwise_TV",
                    "null_mean_TV",
                    "null_p95_TV",
                    "p_one_sided_TV",
                    "p_one_sided",
                    "secondary_multinomial_deviance",
                    "noise_floor_paired_test",
                    "pairwise_bootstrap",
                    "noise_floor",
                )
            },
            "2_source_main_effect": {
                k: source_res[k]
                for k in (
                    "test",
                    "n_perm",
                    "observed_SSD_mean_a0",
                    "null_mean",
                    "p_one_sided",
                    "source_means",
                    "contrasts_a0",
                    "adjustment",
                )
            },
            "3_source_x_target_interaction": {
                k: ix_res[k]
                for k in (
                    "test",
                    "n_perm",
                    "observed_SSD_mean_deltaA_MI",
                    "null_mean",
                    "p_one_sided",
                    "deltaA_MI_by_source",
                )
            },
        },
        "matrix_3x3": mat,
        "mechanistic_verdict": verdict,
        "figures": {
            "fig6a": "analysis/matched_rep_collective/replay_primary/figures/fig6a_target_pair_tv.png",
            "fig6b": "analysis/matched_rep_collective/replay_primary/figures/fig6b_action_matrix.png",
            "fig6c": "analysis/matched_rep_collective/replay_primary/figures/fig6c_interaction_contrasts.png",
            "fig6d": "analysis/matched_rep_collective/replay_primary/figures/fig6d_noise_floor.png",
        },
        "exploratory_explicitly_deferred": [
            "surrogate refit",
            "stratum-wise multiplicity of cell p-values",
            "post-hoc field dropping",
            "second-model-family micro replication",
        ],
        "parent_outcome_A": "docs/MATCHED_REP_COLLECTIVE_RESULTS.md",
        "analysis_plan": "docs/MATCHED_REP_COLLECTIVE_REPLAY_ANALYSIS_PLAN.md",
    }

    # Persist field-level TV and contrasts for SI
    field_tv_rows = []
    for fr in panel:
        row = {
            "field_id": fr["field_id"],
            "source": fr["source"],
            "stratum": fr["stratum"],
            "K": fr["K"],
        }
        for a, b in PAIRS:
            row[f"dTV_{TARGET_SHORT[a]}{TARGET_SHORT[b]}"] = _tv(
                np.asarray(fr["targets"][a]["p"]),
                np.asarray(fr["targets"][b]["p"]),
            )
        field_tv_rows.append(row)

    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    (OUT / "field_pairwise_tv.json").write_text(
        json.dumps(field_tv_rows, indent=2), encoding="utf-8"
    )
    (OUT / "interaction_contrasts.json").write_text(
        json.dumps(ix_res["contrasts"], indent=2), encoding="utf-8"
    )
    # keep full target field tvs separately (lighter than embedding in decision)
    (OUT / "noise_floor_detail.json").write_text(
        json.dumps(
            {
                "between_target_tv": target_res["field_tvs"]["between_target_tv_all"],
                "within_block_tv": target_res["field_tvs"]["within_target_block_tv"],
            }
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "target_p": target_res["p_one_sided"],
        "source_p": source_res["p_one_sided"],
        "interaction_p": ix_res["p_one_sided"],
        "mean_TV": target_res["field_tvs"]["mean_pairwise"],
        "noise_ratio": target_res["noise_floor"]["ratio"],
        "verdict": verdict,
    }, indent=2))
    print(f"Wrote {OUT / 'decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
