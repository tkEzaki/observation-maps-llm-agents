"""Analyze the serialization-binding ladder: prompt length vs value-to-bin binding.

Four conditions on the frozen 48-field replay panel. All four serialize the
*same* 24 bin masses at the same precision, so the retained information is
identical along the whole ladder; they differ only in how each mass is bound to
its bin, and their prompt lengths are arranged so that length and binding make
opposite predictions:

    condition             binding                    prompt chars
    --------------------  -------------------------  ------------
    centers_compact       positional                          971
    centers_indexed_row   explicit_index                     1055
    centers_centered_row  explicit_centre                    1114
    centers_standard      explicit_index_and_centre          1429

The prespecified primary contrast pivots on ``centers_indexed_row``, the
shortest condition that binds every mass to an explicit label:

    Delta_binding = mean_field TV(centers_compact, centers_indexed_row)
                  - mean_field TV(centers_indexed_row, centers_standard)

The pivot sits 84 characters from the positional condition and 374 from the
fully labelled one, so a monotone prompt-length account predicts
Delta_binding < 0 while a binding account predicts Delta_binding > 0. The sign
alone separates them; no calibration between characters and response distance is
required, and the length gaps favour the length account a priori.

Delta_binding is judged on its field-cluster bootstrap interval, never on the
permutation p, whose null is the absence of any condition effect rather than
Delta_binding = 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.matched_rep_collective.run_replay_primary_inference import (  # noqa: E402
    N_PERM,
    _stats_from_counts,
    _tv,
    cluster_bootstrap_mean,
)
from circlemap.serialization_binding_controls import (  # noqa: E402
    SBC_ANCHORS,
    SBC_BINDING,
    SBC_FORM,
    SBC_NEW,
    SBC_VARIANTS,
)

RUN_ROOT = (
    ROOT
    / "runs"
    / "matched_rep_collective_sbc"
    / "matched-rep-sbc-v0.1_gpt-5.4-mini"
)
#: earlier serialization-length control acquisition, for the anchor
#: reproducibility check (secondary; skipped with a recorded note if absent)
SLC_RUN_ROOT = (
    ROOT
    / "runs"
    / "matched_rep_collective_slc"
    / "matched-rep-slc-v0.1_gpt-5.4-mini"
)
PROTOCOL_PATH = (
    ROOT / "experiments" / "stage_c" / "protocol_matched_rep_collective_sbc_v0_1.json"
)
FIELDS_PATH = ROOT / "analysis" / "matched_rep_collective" / "replay_fields_v0_1.json"
AUDIT_PATH = ROOT / "analysis" / "matched_rep_collective" / "sbc_information_audit.json"
OUT = ROOT / "analysis" / "matched_rep_collective" / "sbc_primary"

CONDITIONS = tuple(SBC_VARIANTS)
COND_INDEX = {c: i for i, c in enumerate(CONDITIONS)}
PAIRS = tuple(
    (CONDITIONS[i], CONDITIONS[j])
    for i in range(len(CONDITIONS))
    for j in range(i + 1, len(CONDITIONS))
)

#: primary contrast: the shortest explicitly bound condition against the two
#: anchors it might group with
PIVOT = "centers_indexed_row"
POSITIONAL_ANCHOR = "centers_compact"  # +84 chars from the pivot
LABELLED_ANCHOR = "centers_standard"  # -374 chars from the pivot
#: secondary contrast: same shape, near length-neutral pivot (143 / 315 chars)
PIVOT_SECONDARY = "centers_centered_row"

SEED = 20260729
RNG = np.random.default_rng(SEED)

#: The permutation p is reported but never decides the verdict. Same caveat as
#: the serialization-length control analysis.
PERMUTATION_CAVEAT = (
    "NOT the basis of the verdict. The permutation null is condition-label "
    "exchangeability, that is 'no condition effect at all'; rejecting it does "
    "not establish that the contrast itself differs from zero, and with a large "
    "global condition effect this p is small almost regardless of the contrast. "
    "The field-cluster bootstrap interval is the honest statement about the "
    "contrast, so a contrast whose interval covers zero is reported as "
    "directional and unresolved even when this p is small."
)


def _sha_file(path: Path) -> str:
    """Newline-normalized digest of a text artifact.

    Python's text writer translates \\n to \\r\\n on Windows, so the raw bytes of a
    JSON or JSONL artifact, and therefore its digest, depend on the platform
    that wrote it. Normalizing to LF before hashing keeps a recorded digest
    comparable across platforms while still catching any content change.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sign_test_p(n_pos: int, n_neg: int) -> float:
    """Exact two-sided sign test on the discordant pairs."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(n_pos, n_neg) + 1))
    return float(min(1.0, 2.0 * tail / 2**n))


def _pair_key(a: str, b: str) -> str:
    return f"{a}__vs__{b}"


def _canon_pair_key(a: str, b: str) -> str:
    """Pair key in the canonical condition order, so lookups never miss."""
    return _pair_key(*sorted((a, b), key=CONDITIONS.index))


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


# --------------------------------------------------------------------------
# acquisition discovery / loading
# --------------------------------------------------------------------------


def find_run_dir(explicit: Path | None = None) -> Path:
    """Newest timestamped acquisition, or the explicit --run-dir."""
    if explicit is not None:
        run_dir = Path(explicit).expanduser()
        if not (run_dir / "trace.jsonl").exists():
            raise SystemExit(
                "SBC_ACQUISITION_MISSING: no acquisition found at "
                f"{run_dir} (no trace.jsonl)."
            )
        return run_dir
    candidates = []
    if RUN_ROOT.is_dir():
        candidates = sorted(
            d for d in RUN_ROOT.iterdir() if d.is_dir() and (d / "trace.jsonl").exists()
        )
    if not candidates:
        raise SystemExit(
            "SBC_ACQUISITION_MISSING: no acquisition found under "
            f"{RUN_ROOT} — run experiments/stage_c/run_sbc_control.py first, "
            "or pass --run-dir."
        )
    return candidates[-1]


def find_slc_run_dir() -> Path | None:
    """Newest serialization-length control acquisition, or None."""
    if not SLC_RUN_ROOT.is_dir():
        return None
    candidates = sorted(
        d for d in SLC_RUN_ROOT.iterdir() if d.is_dir() and (d / "trace.jsonl").exists()
    )
    return candidates[-1] if candidates else None


def _load(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------


def build_panel(fields: list[dict], rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Per-field action probabilities by condition and by (condition, block).

    Returns the panel over usable fields (every condition observed at least
    once) and the list of field ids dropped for incompleteness.
    """
    by_fc: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=int)
    )
    by_fcb: dict[tuple[str, str, int], np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=int)
    )
    raw_by_field: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for r in rows:
        if not r.get("valid"):
            continue
        cond = r["target_representation"]
        if cond not in COND_INDEX:
            continue
        idx = {-1: 0, 0: 1, 1: 2}[int(r["action_value"])]
        fid = r["field_id"]
        block = int(r["acquisition_block"])
        by_fc[(fid, cond)][idx] += 1
        by_fcb[(fid, cond, block)][idx] += 1
        raw_by_field[fid].append((cond, idx, block))

    panel: list[dict] = []
    dropped: list[str] = []
    for f in fields:
        fid = f["field_id"]
        if any(by_fc[(fid, c)].sum() == 0 for c in CONDITIONS):
            dropped.append(fid)
            continue
        raw = raw_by_field[fid]
        entry = {
            "field_id": fid,
            "source": f["source_representation"],
            "stratum": f["stratum"],
            "K": float(f["K"]),
            "raw": raw,
            "cond_idx": np.asarray([COND_INDEX[c] for c, _, _ in raw], dtype=np.int64),
            "act_idx": np.asarray([i for _, i, _ in raw], dtype=np.int64),
            "conditions": {c: _stats_from_counts(by_fc[(fid, c)]) for c in CONDITIONS},
            "blocks": {
                f"{c}|b{b}": _stats_from_counts(by_fcb[(fid, c, b)])
                for c in CONDITIONS
                for b in (0, 1)
            },
        }
        panel.append(entry)
    return panel, dropped


def _p_matrix(cond_idx: np.ndarray, act_idx: np.ndarray) -> np.ndarray:
    """(n_conditions, 3) action probabilities from response-level index arrays."""
    counts = np.bincount(
        cond_idx * 3 + act_idx, minlength=len(CONDITIONS) * 3
    ).reshape(len(CONDITIONS), 3)
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(totals, 1)


def _pair_tvs_from_p(p: np.ndarray) -> dict[str, float]:
    return {
        _pair_key(a, b): _tv(p[COND_INDEX[a]], p[COND_INDEX[b]]) for a, b in PAIRS
    }


def observed_field_tvs(panel: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {_pair_key(a, b): [] for a, b in PAIRS}
    for fr in panel:
        for a, b in PAIRS:
            pa = np.asarray(fr["conditions"][a]["p"], dtype=float)
            pb = np.asarray(fr["conditions"][b]["p"], dtype=float)
            out[_pair_key(a, b)].append(_tv(pa, pb))
    return out


def block_noise_floor(panel: list[dict]) -> list[float]:
    """Within-condition between-block TV over all (field, condition) cells."""
    within = []
    for fr in panel:
        for c in CONDITIONS:
            p0 = np.asarray(fr["blocks"][f"{c}|b0"]["p"], dtype=float)
            p1 = np.asarray(fr["blocks"][f"{c}|b1"]["p"], dtype=float)
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                within.append(_tv(p0, p1))
    return within


# --------------------------------------------------------------------------
# permutation (condition label attached to each individual response, in field)
# --------------------------------------------------------------------------


def permute_stats(panel: list[dict]) -> tuple[float, float, float]:
    """One field-blocked permutation.

    Returns (Delta_binding, Delta_binding with the centred-row pivot, mean
    pairwise TV). Same exchangeability as ``analyze_slc_control.py``'s
    ``permute_stats``: the condition label carried by each individual response
    is shuffled inside its field, so the per-condition response counts are
    preserved and any condition structure is broken.
    """
    deltas = np.empty(len(panel))
    deltas_secondary = np.empty(len(panel))
    pair_means = np.empty(len(panel))
    i_piv = COND_INDEX[PIVOT]
    i_piv2 = COND_INDEX[PIVOT_SECONDARY]
    i_pos = COND_INDEX[POSITIONAL_ANCHOR]
    i_lab = COND_INDEX[LABELLED_ANCHOR]
    for k, fr in enumerate(panel):
        labels = RNG.permutation(fr["cond_idx"])
        p = _p_matrix(labels, fr["act_idx"])
        deltas[k] = _tv(p[i_pos], p[i_piv]) - _tv(p[i_piv], p[i_lab])
        deltas_secondary[k] = _tv(p[i_pos], p[i_piv2]) - _tv(p[i_piv2], p[i_lab])
        pair_means[k] = float(
            np.mean([_tv(p[COND_INDEX[a]], p[COND_INDEX[b]]) for a, b in PAIRS])
        )
    return (
        float(np.mean(deltas)),
        float(np.mean(deltas_secondary)),
        float(np.mean(pair_means)),
    )


def _two_sided_from_tails(null: np.ndarray, obs: float) -> dict:
    """Plus-one estimator tails and the doubled two-sided p on the sign."""
    p_upper = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    p_lower = float((np.sum(null <= obs) + 1) / (len(null) + 1))
    return {
        "p_one_sided_upper_tail": p_upper,
        "p_one_sided_lower_tail": p_lower,
        "p_two_sided": float(min(1.0, 2.0 * min(p_upper, p_lower))),
        "null_mean": float(np.mean(null)),
        "null_p2p5": float(np.quantile(null, 0.025)),
        "null_p97p5": float(np.quantile(null, 0.975)),
        "n_perm": int(len(null)),
        "estimator": "plus_one",
        "caveat": PERMUTATION_CAVEAT,
    }


# --------------------------------------------------------------------------
# Spearman (numpy only; the project carries no scipy dependency)
# --------------------------------------------------------------------------


def _rank_average_ties(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = float(np.mean(ranks[order[i : j + 1]]))
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rank_average_ties(x)
    ry = _rank_average_ties(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt(np.sum(rx**2) * np.sum(ry**2)))
    if den <= 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / den)


def _json_float(value: float) -> float | None:
    """NaN is not JSON; a constant predictor has no defined rank correlation."""
    return None if not math.isfinite(value) else float(value)


# --------------------------------------------------------------------------
# secondary blocks
# --------------------------------------------------------------------------


def pivot_contrast(
    pivot: str,
    field_tvs: dict[str, list[float]],
    pair_mean: dict[str, float],
    prompt_chars: dict[str, int],
) -> dict:
    """Delta = TV(positional anchor, pivot) - TV(pivot, labelled anchor).

    Positive means the pivot groups with the fully labelled anchor (binding);
    negative means it groups with the positional anchor (length), which is what
    the character gaps predict for both pivots.
    """
    k_pos = _canon_pair_key(POSITIONAL_ANCHOR, pivot)
    k_lab = _canon_pair_key(pivot, LABELLED_ANCHOR)
    tv_pos = np.asarray(field_tvs[k_pos], dtype=float)
    tv_lab = np.asarray(field_tvs[k_lab], dtype=float)
    delta_field = tv_pos - tv_lab
    delta_obs = float(np.mean(delta_field))
    boot = cluster_bootstrap_mean(delta_field)
    ci = boot["ci95"]
    n_pos = int(np.sum(delta_field > 0))
    n_neg = int(np.sum(delta_field < 0))
    gap_pos = abs(prompt_chars[pivot] - prompt_chars[POSITIONAL_ANCHOR])
    gap_lab = abs(prompt_chars[pivot] - prompt_chars[LABELLED_ANCHOR])
    return {
        "pivot": pivot,
        "positional_anchor": POSITIONAL_ANCHOR,
        "labelled_anchor": LABELLED_ANCHOR,
        "pair_pivot_to_positional": k_pos,
        "pair_pivot_to_labelled": k_lab,
        "char_gap_pivot_to_positional": gap_pos,
        "char_gap_pivot_to_labelled": gap_lab,
        "length_account_predicts": "negative" if gap_pos < gap_lab else "positive",
        "binding_account_predicts": "positive",
        "mean_TV_pivot_to_positional": pair_mean[k_pos],
        "mean_TV_pivot_to_labelled": pair_mean[k_lab],
        "delta": delta_obs,
        "delta_bootstrap": boot,
        "delta_ci_excludes_zero": bool(ci[0] > 0.0 or ci[1] < 0.0),
        "favours": (
            "binding" if delta_obs > 0 else "length" if delta_obs < 0 else "neither"
        ),
        "exact_sign_test": {
            "test": "exact two-sided sign test over the discordant fields",
            "n_discordant": n_pos + n_neg,
            "n_favouring_binding": n_pos,
            "n_favouring_length": n_neg,
            "p_two_sided": _sign_test_p(n_pos, n_neg),
        },
    }


def monotonicity_from_anchor(
    anchor: str,
    field_tvs: dict[str, list[float]],
    pair_mean: dict[str, float],
    pair_boot: dict[str, dict],
    prompt_chars: dict[str, int],
    within_mean: float,
) -> dict:
    """Are the distances from ``anchor`` monotone in prompt-length gap?

    The required ordering is derived from ``prompt_chars``, never hard-coded:
    under any monotone increasing function of prompt length, the condition
    FURTHER from the anchor in characters must be at least as far in response.
    Each required inequality is then tested in the anchored paired form of
    ``analyze_slc_control.py``: the same physical field supplies both distances,
    and a strictly positive excess of the nearer over the further condition
    refutes the monotone length account.
    """
    others = [c for c in CONDITIONS if c != anchor]
    # descending character gap: the required non-increasing order of TV
    by_gap = sorted(
        others, key=lambda c: (-abs(prompt_chars[c] - prompt_chars[anchor]), c)
    )
    distances = []
    for c in by_gap:
        key = _canon_pair_key(anchor, c)
        distances.append(
            {
                "condition": c,
                "pair": key,
                "prompt_chars": prompt_chars[c],
                "char_gap_from_anchor": abs(prompt_chars[c] - prompt_chars[anchor]),
                "mean_TV": pair_mean[key],
                "bootstrap": pair_boot[key],
                "ratio_to_block_noise": pair_mean[key] / max(within_mean, 1e-12),
            }
        )
    observed_by_tv = sorted(others, key=lambda c: (-pair_mean[_canon_pair_key(anchor, c)], c))
    tv_sequence = [d["mean_TV"] for d in distances]
    holds = all(
        tv_sequence[i] >= tv_sequence[i + 1] for i in range(len(tv_sequence) - 1)
    )

    comparisons = {}
    for i in range(len(by_gap)):
        for j in range(i + 1, len(by_gap)):
            further, nearer = by_gap[i], by_gap[j]  # by_gap is descending in gap
            k_far = _canon_pair_key(anchor, further)
            k_near = _canon_pair_key(anchor, nearer)
            excess = np.asarray(field_tvs[k_near], dtype=float) - np.asarray(
                field_tvs[k_far], dtype=float
            )
            boot = cluster_bootstrap_mean(excess)
            n_pos = int(np.sum(excess > 0))
            n_neg = int(np.sum(excess < 0))
            comparisons[f"{nearer}__nearer_than__{further}"] = {
                "nearer_in_length": nearer,
                "nearer_char_gap": abs(prompt_chars[nearer] - prompt_chars[anchor]),
                "nearer_mean_TV": pair_mean[k_near],
                "further_in_length": further,
                "further_char_gap": abs(prompt_chars[further] - prompt_chars[anchor]),
                "further_mean_TV": pair_mean[k_far],
                "required_inequality": (
                    f"TV({anchor}, {further}) >= TV({anchor}, {nearer})"
                ),
                "paired_excess_near_minus_far": boot,
                "exact_sign_test": {
                    "n_discordant": n_pos + n_neg,
                    "n_near_larger": n_pos,
                    "n_far_larger": n_neg,
                    "p_two_sided": _sign_test_p(n_pos, n_neg),
                },
                "monotone_length_violated": bool(boot["ci95"][0] > 0.0),
            }

    return {
        "test": (
            "paired within-field comparison of the information-identical "
            "conditions against one anchor; any monotone increasing function of "
            "prompt length requires the condition further from the anchor in "
            "characters to be at least as distant in response"
        ),
        "prespecified_as": "secondary_contrasts[1]",
        "paired_implementation_post_hoc": True,
        "anchor": anchor,
        "anchor_prompt_chars": prompt_chars[anchor],
        "distances_ordered_by_char_gap_descending": distances,
        "required_order_by_length": by_gap,
        "required_ordering_statement": " >= ".join(
            f"TV({anchor}, {c})" for c in by_gap
        ),
        "observed_order_by_mean_TV": observed_by_tv,
        "required_ordering_holds": bool(holds),
        "comparisons": comparisons,
        "monotone_length_account_refuted": any(
            v["monotone_length_violated"] for v in comparisons.values()
        ),
    }


def anchor_reproducibility(panel: list[dict], within_boot: dict) -> dict:
    """Per-field TV between this run's anchors and the earlier SLC acquisition."""
    slc_dir = find_slc_run_dir()
    if slc_dir is None:
        return {
            "status": "SLC_ACQUISITION_ABSENT",
            "note": (
                "No serialization-length control acquisition found under "
                f"{_repo_relative(SLC_RUN_ROOT)}; the run-to-run reproducibility "
                "check of the two re-acquired anchors is skipped."
            ),
            "anchors": list(SBC_ANCHORS),
        }
    slc_rows = _load(slc_dir)
    slc_counts: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=int)
    )
    for r in slc_rows:
        if not r.get("valid"):
            continue
        cond = r["target_representation"]
        if cond not in SBC_ANCHORS:
            continue
        idx = {-1: 0, 0: 1, 1: 2}[int(r["action_value"])]
        slc_counts[(r["field_id"], cond)][idx] += 1

    out: dict = {
        "status": "COMPARED",
        "slc_run_dir": _repo_relative(slc_dir),
        "slc_trace_sha256": _sha_file(slc_dir / "trace.jsonl"),
        "block_noise_floor": within_boot,
        "anchors": {},
    }
    for anchor in SBC_ANCHORS:
        tvs = []
        for fr in panel:
            counts = slc_counts.get((fr["field_id"], anchor))
            if counts is None or counts.sum() == 0:
                continue
            p_slc = counts.astype(float) / counts.sum()
            p_sbc = np.asarray(fr["conditions"][anchor]["p"], dtype=float)
            tvs.append(_tv(p_sbc, p_slc))
        if not tvs:
            out["anchors"][anchor] = {
                "status": "NO_OVERLAPPING_FIELDS",
                "n_fields": 0,
            }
            continue
        arr = np.asarray(tvs, dtype=float)
        boot = cluster_bootstrap_mean(arr)
        mean_tv = float(np.mean(arr))
        out["anchors"][anchor] = {
            "status": "COMPARED",
            "n_fields": len(tvs),
            "mean_between_run_TV": mean_tv,
            "bootstrap": boot,
            "ratio_to_block_noise": mean_tv / max(within_boot["mean"], 1e-12),
            "agrees_within_block_noise": bool(
                boot["ci95"][0] <= within_boot["ci95"][1]
            ),
        }
    compared = [
        v for v in out["anchors"].values() if v.get("status") == "COMPARED"
    ]
    out["all_anchors_agree_within_block_noise"] = bool(
        compared and all(v["agrees_within_block_noise"] for v in compared)
    )
    out["note"] = (
        "Per-field total-variation distance between the two acquisitions' action "
        "distributions for each re-acquired anchor, judged against the "
        "within-run between-block noise floor: agreement means the "
        "between-run distance is not resolvably larger than the noise floor "
        "(its bootstrap interval reaches the floor's interval)."
    )
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="acquisition directory (default: newest under runs/matched_rep_collective_sbc)",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT, help="output directory for decision.json"
    )
    args = parser.parse_args(argv)

    run_dir = find_run_dir(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    prompt_chars = protocol["prompt_chars_on_frozen_panel"]
    fields = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))["fields"]
    rows = _load(run_dir)
    session_path = run_dir / "session_summary.json"
    session = (
        json.loads(session_path.read_text(encoding="utf-8"))
        if session_path.exists()
        else {}
    )

    n_calls = len(rows)
    n_valid = sum(1 for r in rows if r.get("valid"))

    # --- 1. per-field panel ----------------------------------------------
    panel, dropped = build_panel(fields, rows)
    if not panel:
        raise SystemExit(
            "SBC_ACQUISITION_MISSING: trace at "
            f"{run_dir} carries no valid responses for the four ladder conditions."
        )

    # --- 2. six pairwise per-field TVs -----------------------------------
    field_tvs = observed_field_tvs(panel)
    pair_boot = {
        k: cluster_bootstrap_mean(np.asarray(v, dtype=float))
        for k, v in field_tvs.items()
    }
    pair_mean = {k: float(np.mean(v)) for k, v in field_tvs.items()}

    # --- 3. noise floor ---------------------------------------------------
    within = block_noise_floor(panel)
    within_boot = cluster_bootstrap_mean(np.asarray(within, dtype=float))
    within_mean = float(np.mean(within))

    # --- 4. primary prespecified contrast ---------------------------------
    primary = pivot_contrast(PIVOT, field_tvs, pair_mean, prompt_chars)
    delta_obs = primary["delta"]
    delta_ci = primary["delta_bootstrap"]["ci95"]

    # --- 5a. secondary: centred-row pivot ---------------------------------
    secondary_pivot = pivot_contrast(
        PIVOT_SECONDARY, field_tvs, pair_mean, prompt_chars
    )

    all_pair_mean_obs = float(np.mean([v for vals in field_tvs.values() for v in vals]))

    print(f"Running {N_PERM} field-blocked condition-label permutations...", flush=True)
    null_delta = np.empty(N_PERM)
    null_delta_secondary = np.empty(N_PERM)
    null_global = np.empty(N_PERM)
    for i in range(N_PERM):
        null_delta[i], null_delta_secondary[i], null_global[i] = permute_stats(panel)
    primary["permutation"] = {
        "test": "field_blocked_permutation_of_condition_labels_on_responses",
        "two_sided_on": "sign of Delta_binding",
        **_two_sided_from_tails(null_delta, delta_obs),
    }
    secondary_pivot["permutation"] = {
        "test": "field_blocked_permutation_of_condition_labels_on_responses",
        "two_sided_on": "sign of Delta_binding (centred-row pivot)",
        **_two_sided_from_tails(null_delta_secondary, secondary_pivot["delta"]),
    }
    p_global = float((np.sum(null_global >= all_pair_mean_obs) + 1) / (N_PERM + 1))

    # --- 5b. secondary: monotonicity from the fully labelled anchor -------
    monotone = monotonicity_from_anchor(
        LABELLED_ANCHOR, field_tvs, pair_mean, pair_boot, prompt_chars, within_mean
    )

    # --- 5c. secondary: run-to-run reproducibility of the two anchors -----
    repro = anchor_reproducibility(panel, within_boot)

    # --- 5d. secondary: rank correlation of the six pairwise mean TVs -----
    predictor_rows = []
    for a, b in PAIRS:
        key = _pair_key(a, b)
        predictor_rows.append(
            {
                "pair": key,
                "mean_TV": pair_mean[key],
                "abs_prompt_char_diff": abs(prompt_chars[a] - prompt_chars[b]),
                "binding_mismatch": int(SBC_BINDING[a] != SBC_BINDING[b]),
                "explicit_binding_mismatch": int(
                    (SBC_BINDING[a] == "positional") != (SBC_BINDING[b] == "positional")
                ),
            }
        )
    tv_vec = np.asarray([r["mean_TV"] for r in predictor_rows], dtype=float)
    rho = {
        name: spearman(
            tv_vec, np.asarray([r[name] for r in predictor_rows], dtype=float)
        )
        for name in (
            "abs_prompt_char_diff",
            "binding_mismatch",
            "explicit_binding_mismatch",
        )
    }
    finite = {k: abs(v) for k, v in rho.items() if math.isfinite(v)}
    if not finite:
        better = "undefined"
    else:
        best = max(finite.values())
        winners = sorted(k for k, v in finite.items() if v == best)
        better = winners[0] if len(winners) == 1 else "tie"

    # --- 5e. the design is a crossed 2x2, so decompose by which factor differs
    #
    # This reframing is post hoc. The conditions cross two binary features:
    #   does the prompt print the numeric bin centre next to each mass?
    #   does it print the bin index next to each mass?
    # The prespecified primary contrast was framed on the index feature, which
    # this decomposition shows to be the weak one, so the contrast turned out
    # not to be diagnostic. Both main effects are carried equally by a crossed
    # design, so reading them is analysis of the design rather than a search.
    PRINTS_CENTRE = {
        "centers_compact": False,
        "centers_indexed_row": False,
        "centers_centered_row": True,
        "centers_standard": True,
    }
    PRINTS_INDEX = {
        "centers_compact": False,
        "centers_indexed_row": True,
        "centers_centered_row": False,
        "centers_standard": True,
    }
    cells: dict[str, list[tuple[str, str]]] = {}
    for a, b in PAIRS:
        dc = PRINTS_CENTRE[a] != PRINTS_CENTRE[b]
        di = PRINTS_INDEX[a] != PRINTS_INDEX[b]
        name = "both" if (dc and di) else "centre_only" if dc else "index_only"
        cells.setdefault(name, []).append((a, b))

    cell_report = {}
    cell_mean_field = {}
    for name in ("index_only", "centre_only", "both"):
        members = cells[name]
        stacked = np.mean(
            [np.asarray(field_tvs[_canon_pair_key(a, b)], dtype=float) for a, b in members],
            axis=0,
        )
        cell_mean_field[name] = stacked
        cell_report[name] = {
            "pairs": [
                {
                    "pair": _canon_pair_key(a, b),
                    "abs_prompt_char_diff": abs(prompt_chars[a] - prompt_chars[b]),
                    "mean_TV": pair_mean[_canon_pair_key(a, b)],
                }
                for a, b in members
            ],
            "cell_bootstrap": cluster_bootstrap_mean(stacked),
            "ratio_to_block_noise": float(np.mean(stacked)) / max(within_mean, 1e-12),
        }

    factor_diff = cell_mean_field["centre_only"] - cell_mean_field["index_only"]
    factor_boot = cluster_bootstrap_mean(factor_diff)
    n_pos = int(np.sum(factor_diff > 0))
    n_neg = int(np.sum(factor_diff < 0))

    # Within a cell the two pairs differ greatly in character gap while holding
    # the factor pattern fixed, so a length account predicts a large within-cell
    # difference and a factor account predicts none.
    length_insensitivity = {}
    for name in ("index_only", "centre_only", "both"):
        (a1, b1), (a2, b2) = cells[name]
        g1 = abs(prompt_chars[a1] - prompt_chars[b1])
        g2 = abs(prompt_chars[a2] - prompt_chars[b2])
        if g1 > g2:
            (a1, b1), (a2, b2), (g1, g2) = (a2, b2), (a1, b1), (g2, g1)
        k1, k2 = _canon_pair_key(a1, b1), _canon_pair_key(a2, b2)
        diff = np.asarray(field_tvs[k2], dtype=float) - np.asarray(
            field_tvs[k1], dtype=float
        )
        boot = cluster_bootstrap_mean(diff)
        length_insensitivity[name] = {
            "narrow_gap_pair": {"pair": k1, "char_gap": g1, "mean_TV": pair_mean[k1]},
            "wide_gap_pair": {"pair": k2, "char_gap": g2, "mean_TV": pair_mean[k2]},
            "char_gap_ratio": g2 / max(g1, 1),
            "paired_difference_wide_minus_narrow": boot,
            "difference_ci_covers_zero": bool(
                boot["ci95"][0] <= 0.0 <= boot["ci95"][1]
            ),
        }

    factorial = {
        "reframing": "post hoc",
        "features": {
            "prints_bin_centre": PRINTS_CENTRE,
            "prints_bin_index": PRINTS_INDEX,
        },
        "cells": cell_report,
        "centre_only_minus_index_only": {
            "difference": float(np.mean(factor_diff)),
            "bootstrap": factor_boot,
            "exact_sign_test": {
                "n_discordant": n_pos + n_neg,
                "n_centre_larger": n_pos,
                "n_index_larger": n_neg,
                "p_two_sided": _sign_test_p(n_pos, n_neg),
            },
            "ci_excludes_zero": bool(
                factor_boot["ci95"][0] > 0.0 or factor_boot["ci95"][1] < 0.0
            ),
        },
        "within_cell_length_insensitivity": length_insensitivity,
        "length_insensitive_in_every_cell": all(
            v["difference_ci_covers_zero"] for v in length_insensitivity.values()
        ),
        "note": (
            "Distance depends on which feature differs, not on how many "
            "characters differ. Within each cell the character gap varies by a "
            "factor of 2.6 to 7.8 while the response distance does not move "
            "resolvably."
        ),
    }

    # Which of the two pivot contrasts could tell length and centre printing
    # apart at all: a contrast is diagnostic only if the two accounts predict
    # opposite signs for it.
    def _diagnostic(pair_near: tuple[str, str], pair_far: tuple[str, str]) -> dict:
        ka, kb = _canon_pair_key(*pair_near), _canon_pair_key(*pair_far)
        ga = abs(prompt_chars[pair_near[0]] - prompt_chars[pair_near[1]])
        gb = abs(prompt_chars[pair_far[0]] - prompt_chars[pair_far[1]])
        fa = "index_only" if PRINTS_CENTRE[pair_near[0]] == PRINTS_CENTRE[
            pair_near[1]
        ] else "centre_only"
        length_sign = "negative" if ga < gb else "positive"
        centre_sign = "negative" if fa == "index_only" else "positive"
        return {
            "first_term": {"pair": ka, "char_gap": ga, "feature_that_differs": fa},
            "second_term": {"pair": kb, "char_gap": gb},
            "length_account_predicts": length_sign,
            "centre_printing_account_predicts": centre_sign,
            "diagnostic": length_sign != centre_sign,
        }

    primary["diagnosticity"] = _diagnostic(
        (POSITIONAL_ANCHOR, PIVOT), (PIVOT, LABELLED_ANCHOR)
    )
    secondary_pivot["diagnosticity"] = _diagnostic(
        (POSITIONAL_ANCHOR, "centers_centered_row"),
        ("centers_centered_row", LABELLED_ANCHOR),
    )

    # --- 6. verdict -------------------------------------------------------
    #
    # Delta_binding is judged on its bootstrap interval, not on the permutation
    # p: the permutation null is condition-label exchangeability, that is "no
    # condition effect at all", and rejecting it says nothing about whether
    # Delta_binding itself differs from zero.
    delta_excludes_zero = primary["delta_ci_excludes_zero"]
    ordering_holds = monotone["required_ordering_holds"]
    ordering_clause = (
        " The three distances from the fully labelled condition are "
        + ("monotone" if ordering_holds else "NOT monotone")
        + " in prompt length: taking the other conditions in order of decreasing "
        "character gap ("
        + ", ".join(
            str(d["char_gap_from_anchor"])
            for d in monotone["distances_ordered_by_char_gap_descending"]
        )
        + " characters), where a monotone length account requires "
        "non-increasing distances, the mean TVs are "
        + ", ".join(
            f"{d['mean_TV']:.3f}"
            for d in monotone["distances_ordered_by_char_gap_descending"]
        )
        + "."
    )

    # The prespecified contrast is only informative about length versus centre
    # printing if the two accounts predict opposite signs for it. Where it is
    # not diagnostic, the verdict must come from the factorial decomposition
    # instead of pretending the contrast settled something.
    primary_diagnostic = bool(primary["diagnosticity"]["diagnostic"])
    centre_effect = factorial["centre_only_minus_index_only"]
    centre_resolved = bool(centre_effect["ci_excludes_zero"]) and (
        centre_effect["difference"] > 0
    )
    length_insensitive = bool(factorial["length_insensitive_in_every_cell"])

    if not primary_diagnostic:
        verdict = (
            "CENTRE_PRINTING_OVER_LENGTH"
            if (centre_resolved and length_insensitive)
            else "PRIMARY_CONTRAST_NOT_DIAGNOSTIC"
        )
        cells = factorial["cells"]
        claim = (
            "The prespecified contrast is not diagnostic here. Its first term is "
            f"an {primary['diagnosticity']['first_term']['feature_that_differs']} "
            "change and its second a "
            f"{'centre_only' if primary['diagnosticity']['first_term']['feature_that_differs'] == 'index_only' else 'index_only'} "
            "change, and the character gaps run the same way, so a prompt-length "
            "account and a centre-printing account predict the same sign for it "
            f"(observed Delta_binding = {delta_obs:+.4f}, 95% CI "
            f"[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]). The design is a crossed 2x2, "
            "so the verdict comes from decomposing the six distances by which "
            "feature differs. Changing only whether the bin index is printed moved "
            f"the operator by {cells['index_only']['cell_bootstrap']['mean']:.3f} "
            f"({cells['index_only']['ratio_to_block_noise']:.1f} times block noise); "
            "changing only whether the numeric bin centre is printed moved it by "
            f"{cells['centre_only']['cell_bootstrap']['mean']:.3f} "
            f"({cells['centre_only']['ratio_to_block_noise']:.1f} times); changing both, "
            f"{cells['both']['cell_bootstrap']['mean']:.3f}. The centre feature exceeds "
            f"the index feature by {centre_effect['difference']:+.3f} (95% CI "
            f"[{centre_effect['bootstrap']['ci95'][0]:+.3f}, "
            f"{centre_effect['bootstrap']['ci95'][1]:+.3f}], exact sign test p = "
            f"{centre_effect['exact_sign_test']['p_two_sided']:.4f}). Within every cell "
            "the character gap varies by a factor of 2.6 to 7.8 while the response "
            "distance does not move resolvably, so which feature differs predicts the "
            "distance and how many characters differ does not."
            + ordering_clause
        )
    elif delta_obs > 0 and delta_excludes_zero:
        verdict = "BINDING_OVER_LENGTH"
        claim = (
            "Response distances follow the binding of each value to its bin, not "
            f"prompt length: the pivot {PIVOT}, which sits "
            f"{primary['char_gap_pivot_to_positional']} characters from "
            f"{POSITIONAL_ANCHOR} and "
            f"{primary['char_gap_pivot_to_labelled']} from {LABELLED_ANCHOR}, is "
            f"nevertheless closer in response to {LABELLED_ANCHOR} "
            f"({primary['mean_TV_pivot_to_labelled']:.3f}) than to "
            f"{POSITIONAL_ANCHOR} ({primary['mean_TV_pivot_to_positional']:.3f}); "
            f"Delta_binding = {delta_obs:+.4f} with 95% CI "
            f"[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}], which excludes zero."
            + ordering_clause
        )
    elif delta_obs < 0 and delta_excludes_zero:
        verdict = "LENGTH_OVER_BINDING"
        claim = (
            "Response distances follow prompt length rather than the binding of "
            f"each value to its bin: the pivot {PIVOT} groups with the "
            f"positional condition it is {primary['char_gap_pivot_to_positional']} "
            "characters from rather than with the fully labelled condition it is "
            f"{primary['char_gap_pivot_to_labelled']} characters from; "
            f"Delta_binding = {delta_obs:+.4f} with 95% CI "
            f"[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}], which excludes zero."
            + ordering_clause
        )
    else:
        verdict = "UNRESOLVED"
        direction = (
            "binding" if delta_obs > 0 else "length" if delta_obs < 0 else "neither"
        )
        claim = (
            "Neither account is established by this design. Delta_binding = "
            f"{delta_obs:+.4f} with 95% CI [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}], "
            "which covers zero, so the contrast is "
            f"{'directional towards ' + direction if direction != 'neither' else 'flat'} "
            "and plainly unresolved; the permutation p is not read as support for "
            f"Delta_binding itself (p_two_sided = "
            f"{primary['permutation']['p_two_sided']:.4f})."
            + ordering_clause
        )

    decision = {
        "status": "SBC_CONTROL_PRIMARY_V0_1",
        "inference_unit": "physical_fields",
        "run_dir": _repo_relative(run_dir),
        "acquisition": {
            "n_calls": n_calls,
            "n_valid": n_valid,
            "valid_rate": n_valid / max(n_calls, 1),
            "expected_calls": int(protocol["expected_calls"]),
            "matches_expected_calls": n_calls == int(protocol["expected_calls"]),
            "actual_cost_usd_session": session.get("actual_cost_usd_session"),
            "elapsed_seconds": session.get("elapsed_seconds"),
            "n_fields_used": len(panel),
            "fields_dropped_incomplete": dropped,
        },
        "conditions": {
            c: {
                "binding": SBC_BINDING[c],
                "form": SBC_FORM[c],
                "prompt_chars": prompt_chars[c],
                "role": "anchor" if c in SBC_ANCHORS else "new_rung",
            }
            for c in CONDITIONS
        },
        "condition_roles": {
            "reacquired_anchors": list(SBC_ANCHORS),
            "new_rungs": list(SBC_NEW),
        },
        "hashes": {
            "information_audit_sha256": _sha_file(AUDIT_PATH),
            "protocol_sha256": _sha_file(PROTOCOL_PATH),
            "fields_sha256": _sha_file(FIELDS_PATH),
            "trace_sha256": _sha_file(run_dir / "trace.jsonl"),
        },
        "inference_settings": {
            "n_perm": N_PERM,
            "rng_seed": SEED,
            "test": "field_blocked_permutation_of_condition_labels_on_responses",
            "alpha_primary": 0.05,
            "bootstrap_rng": (
                "field-cluster bootstrap from "
                "analysis.matched_rep_collective.run_replay_primary_inference "
                "(module RNG seed 20260724, 5000 resamples)"
            ),
            "permutation_note": (
                "One set of permutation draws serves the primary Delta_binding "
                "test, the centred-row pivot contrast and the global "
                "any-difference test; all three share the same null of "
                "condition-label exchangeability within a field."
            ),
        },
        "pairwise_bootstrap": pair_boot,
        "pairwise_mean_TV": pair_mean,
        "noise_floor": {
            "mean_within_condition_between_block_TV": within_mean,
            "within_boot": within_boot,
            "mean_between_condition_TV": all_pair_mean_obs,
            "ratio_between_over_within": all_pair_mean_obs / max(within_mean, 1e-12),
            "n_cells": len(within),
        },
        "primary_contrast": {
            "name": protocol["primary_contrast"],
            "definition": protocol["primary_contrast_definition"],
            "interpretation_rule": protocol["primary_interpretation_rule"],
            "decided_on": "delta_bootstrap.ci95",
            **primary,
        },
        "secondary": {
            "factorial_decomposition": factorial,
            "centred_row_pivot_contrast": {
                "note": (
                    "Same contrast with centers_centered_row as pivot, whose "
                    "character gaps are nearly balanced, so it asks only whether "
                    "an explicitly bound single row groups with the labelled or "
                    "with the positional condition."
                ),
                **secondary_pivot,
            },
            "monotonicity_from_labelled_anchor": monotone,
            "anchor_reproducibility_vs_slc": repro,
            "predictor_rank_correlation": {
                "rows": predictor_rows,
                "spearman_rho_abs_prompt_char_diff": _json_float(
                    rho["abs_prompt_char_diff"]
                ),
                "spearman_rho_binding_mismatch": _json_float(rho["binding_mismatch"]),
                "spearman_rho_explicit_binding_mismatch": _json_float(
                    rho["explicit_binding_mismatch"]
                ),
                "better_predictor": better,
                "n_pairs": len(predictor_rows),
                "note": (
                    "Spearman on six condition pairs; implemented in numpy "
                    "(average ranks for ties), no scipy dependency. All four "
                    "conditions carry a different value of SBC_BINDING, so "
                    "binding_mismatch is 1 for every pair: it is constant, has no "
                    "rank variance, and its correlation is undefined (null). That "
                    "is why explicit_binding_mismatch, 1 when exactly one member "
                    "of the pair is positionally bound, is reported as well: it is "
                    "the axis the ladder was built to test."
                ),
            },
            "global_any_condition_difference": {
                "test": "field_blocked_condition_label_permutation_mean_pairwise_TV",
                "observed_mean_pairwise_TV": all_pair_mean_obs,
                "null_mean": float(np.mean(null_global)),
                "null_p95": float(np.quantile(null_global, 0.95)),
                "p_one_sided": p_global,
                "n_perm": N_PERM,
            },
        },
        "verdict": verdict,
        "claim": claim,
        "primary_interpretation_rule": protocol["primary_interpretation_rule"],
        "scope_note": protocol["scope_note"],
    }

    (out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "run_dir": decision["run_dir"],
                "n_valid": n_valid,
                "valid_rate": decision["acquisition"]["valid_rate"],
                "mean_TV": pair_mean,
                "within_block_TV": within_mean,
                "delta_binding": delta_obs,
                "delta_binding_ci95": delta_ci,
                "delta_binding_p_two_sided": primary["permutation"]["p_two_sided"],
                "delta_binding_sign_test_p": primary["exact_sign_test"]["p_two_sided"],
                "favours": primary["favours"],
                "delta_centred_row_pivot": secondary_pivot["delta"],
                "delta_centred_row_pivot_ci95": secondary_pivot["delta_bootstrap"][
                    "ci95"
                ],
                "monotone_ordering_holds": monotone["required_ordering_holds"],
                "monotone_length_account_refuted": monotone[
                    "monotone_length_account_refuted"
                ],
                "anchor_reproducibility": repro.get(
                    "all_anchors_agree_within_block_noise", repro.get("status")
                ),
                "p_any_condition_difference": p_global,
                "spearman_length": _json_float(rho["abs_prompt_char_diff"]),
                "spearman_binding_mismatch": _json_float(rho["binding_mismatch"]),
                "spearman_explicit_binding": _json_float(
                    rho["explicit_binding_mismatch"]
                ),
                "better_predictor": better,
                "verdict": verdict,
                "claim": claim,
                "cost": session.get("actual_cost_usd_session"),
            },
            indent=2,
        )
    )
    print(f"Wrote {out_dir / 'decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
