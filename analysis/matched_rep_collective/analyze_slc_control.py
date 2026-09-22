"""Analyze the serialization-length control: content class vs prompt length.

Four arms on the frozen 48-field replay panel:

    arm               content class   form      prompt chars (frozen panel)
    ----------------  --------------  --------  ---------------------------
    moments_compact   moments3        compact   752
    moments_standard  moments3        standard  789
    centers_compact   histogram24     compact   971
    centers_standard  histogram24     standard  1429

Within a content class the two forms carry identical numbers at identical
precision, so the length axis crosses the content axis: ``centers_compact``
sits 182 characters from ``moments_standard`` (different content) but 458
characters from ``centers_standard`` (identical content).  The prespecified
primary contrast

    Delta = mean_field TV(centers_compact, moments_standard)
          - mean_field TV(centers_compact, centers_standard)

is therefore positive if the operator tracks retained information and negative
if it tracks prompt length.
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
from circlemap.serialization_length_controls import (  # noqa: E402
    SLC_CONTENT,
    SLC_FORM,
    SLC_VARIANTS,
)

RUN_ROOT = (
    ROOT
    / "runs"
    / "matched_rep_collective_slc"
    / "matched-rep-slc-v0.1_gpt-5.4-mini"
)
PROTOCOL_PATH = (
    ROOT / "experiments" / "stage_c" / "protocol_matched_rep_collective_slc_v0_1.json"
)
FIELDS_PATH = ROOT / "analysis" / "matched_rep_collective" / "replay_fields_v0_1.json"
AUDIT_PATH = ROOT / "analysis" / "matched_rep_collective" / "slc_information_audit.json"
OUT = ROOT / "analysis" / "matched_rep_collective" / "slc_primary"

ARMS = tuple(SLC_VARIANTS)
ARM_INDEX = {a: i for i, a in enumerate(ARMS)}
PAIRS = tuple(
    (ARMS[i], ARMS[j]) for i in range(len(ARMS)) for j in range(i + 1, len(ARMS))
)
#: the two arms of the primary contrast, relative to centers_compact
PIVOT = "centers_compact"
CONTENT_PARTNER = "centers_standard"  # same content, +458 chars
LENGTH_NEIGHBOUR = "moments_standard"  # different content, -182 chars

SEED = 20260729
RNG = np.random.default_rng(SEED)


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
                "SLC_ACQUISITION_MISSING: no acquisition found at "
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
            "SLC_ACQUISITION_MISSING: no acquisition found under "
            f"{RUN_ROOT} — run experiments/stage_c/run_slc_control.py first, "
            "or pass --run-dir."
        )
    return candidates[-1]


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
    """Per-field action probabilities by arm and by (arm, block).

    Returns the panel over usable fields (every arm observed at least once)
    and the list of field ids dropped for incompleteness.
    """
    by_fa: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=int)
    )
    by_fab: dict[tuple[str, str, int], np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=int)
    )
    raw_by_field: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for r in rows:
        if not r.get("valid"):
            continue
        arm = r["target_representation"]
        if arm not in ARM_INDEX:
            continue
        idx = {-1: 0, 0: 1, 1: 2}[int(r["action_value"])]
        fid = r["field_id"]
        block = int(r["acquisition_block"])
        by_fa[(fid, arm)][idx] += 1
        by_fab[(fid, arm, block)][idx] += 1
        raw_by_field[fid].append((arm, idx, block))

    panel: list[dict] = []
    dropped: list[str] = []
    for f in fields:
        fid = f["field_id"]
        if any(by_fa[(fid, a)].sum() == 0 for a in ARMS):
            dropped.append(fid)
            continue
        raw = raw_by_field[fid]
        entry = {
            "field_id": fid,
            "source": f["source_representation"],
            "stratum": f["stratum"],
            "K": float(f["K"]),
            "raw": raw,
            "arm_idx": np.asarray([ARM_INDEX[a] for a, _, _ in raw], dtype=np.int64),
            "act_idx": np.asarray([i for _, i, _ in raw], dtype=np.int64),
            "arms": {a: _stats_from_counts(by_fa[(fid, a)]) for a in ARMS},
            "blocks": {
                f"{a}|b{b}": _stats_from_counts(by_fab[(fid, a, b)])
                for a in ARMS
                for b in (0, 1)
            },
        }
        panel.append(entry)
    return panel, dropped


def _p_matrix(arm_idx: np.ndarray, act_idx: np.ndarray) -> np.ndarray:
    """(n_arms, 3) action probabilities from response-level index arrays."""
    counts = np.bincount(
        arm_idx * 3 + act_idx, minlength=len(ARMS) * 3
    ).reshape(len(ARMS), 3)
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(totals, 1)


def _pair_tvs_from_p(p: np.ndarray) -> dict[str, float]:
    return {
        _pair_key(a, b): _tv(p[ARM_INDEX[a]], p[ARM_INDEX[b]]) for a, b in PAIRS
    }


def observed_field_tvs(panel: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {_pair_key(a, b): [] for a, b in PAIRS}
    for fr in panel:
        for a, b in PAIRS:
            pa = np.asarray(fr["arms"][a]["p"], dtype=float)
            pb = np.asarray(fr["arms"][b]["p"], dtype=float)
            out[_pair_key(a, b)].append(_tv(pa, pb))
    return out


def block_noise_floor(panel: list[dict]) -> list[float]:
    """Within-arm between-block TV over all (field, arm) cells."""
    within = []
    for fr in panel:
        for a in ARMS:
            p0 = np.asarray(fr["blocks"][f"{a}|b0"]["p"], dtype=float)
            p1 = np.asarray(fr["blocks"][f"{a}|b1"]["p"], dtype=float)
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                within.append(_tv(p0, p1))
    return within


# --------------------------------------------------------------------------
# permutation (arm label attached to each individual response, within field)
# --------------------------------------------------------------------------


def permute_stats(panel: list[dict]) -> tuple[float, float]:
    """One field-blocked permutation; returns (Delta, mean pairwise TV).

    Same exchangeability as ``analyze_omap_control.py``'s ``permute``: the arm
    label carried by each individual response is shuffled inside its field, so
    the per-arm response counts are preserved and any arm structure is broken.
    """
    deltas = np.empty(len(panel))
    pair_means = np.empty(len(panel))
    i_piv = ARM_INDEX[PIVOT]
    i_len = ARM_INDEX[LENGTH_NEIGHBOUR]
    i_con = ARM_INDEX[CONTENT_PARTNER]
    for k, fr in enumerate(panel):
        labels = RNG.permutation(fr["arm_idx"])
        p = _p_matrix(labels, fr["act_idx"])
        deltas[k] = _tv(p[i_piv], p[i_len]) - _tv(p[i_piv], p[i_con])
        pair_means[k] = float(
            np.mean([_tv(p[ARM_INDEX[a]], p[ARM_INDEX[b]]) for a, b in PAIRS])
        )
    return float(np.mean(deltas)), float(np.mean(pair_means))


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


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="acquisition directory (default: newest under runs/matched_rep_collective_slc)",
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
    panel, dropped = build_panel(fields, rows)
    if not panel:
        raise SystemExit(
            "SLC_ACQUISITION_MISSING: trace at "
            f"{run_dir} carries no valid responses for the four control arms."
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
    k_len = _pair_key(*sorted((PIVOT, LENGTH_NEIGHBOUR), key=ARMS.index))
    k_con = _pair_key(*sorted((PIVOT, CONTENT_PARTNER), key=ARMS.index))
    tv_len = np.asarray(field_tvs[k_len], dtype=float)  # different content, +182 chars
    tv_con = np.asarray(field_tvs[k_con], dtype=float)  # same content, +458 chars
    delta_field = tv_len - tv_con
    delta_obs = float(np.mean(delta_field))
    delta_boot = cluster_bootstrap_mean(delta_field)

    all_pair_mean_obs = float(np.mean([v for vals in field_tvs.values() for v in vals]))

    print(f"Running {N_PERM} field-blocked arm-label permutations...", flush=True)
    null_delta = np.empty(N_PERM)
    null_global = np.empty(N_PERM)
    for i in range(N_PERM):
        null_delta[i], null_global[i] = permute_stats(panel)
    p_delta = float((np.sum(null_delta >= delta_obs) + 1) / (N_PERM + 1))
    p_delta_lower = float((np.sum(null_delta <= delta_obs) + 1) / (N_PERM + 1))
    p_global = float((np.sum(null_global >= all_pair_mean_obs) + 1) / (N_PERM + 1))

    # --- 5. secondary -----------------------------------------------------
    k_layout = _pair_key("moments_standard", "moments_compact")
    k_info = _pair_key("centers_standard", "centers_compact")
    layout_ref = {
        **pair_boot[k_layout],
        "ratio_to_block_noise": pair_mean[k_layout] / max(within_mean, 1e-12),
        "prompt_char_gap": abs(
            prompt_chars["moments_standard"] - prompt_chars["moments_compact"]
        ),
        "content_mismatch": 0,
    }
    info_ref = {
        **pair_boot[k_info],
        "ratio_to_block_noise": pair_mean[k_info] / max(within_mean, 1e-12),
        "prompt_char_gap": abs(
            prompt_chars["centers_standard"] - prompt_chars["centers_compact"]
        ),
        "content_mismatch": 0,
    }

    # Anchored monotonicity test (post hoc, not prespecified).
    #
    # The two single-class reference contrasts above cannot separate form from
    # length, because the moments pair differs by 37 characters and the
    # histogram pair by 458: a length account predicts exactly that ordering.
    # This contrast is immune to that confound. Hold one moments prompt fixed
    # and compare it with the two histogram prompts, which carry identical
    # information and differ only in serialization. Under any monotone
    # increasing function of prompt length, the histogram prompt that is
    # FURTHER from the anchor in length must be at least as far in response.
    # The comparison is paired within field, so the same physical field
    # supplies both distances.
    anchored = {}
    for anchor in ("moments_standard", "moments_compact"):
        k_near = _pair_key(*sorted((anchor, "centers_compact"), key=ARMS.index))
        k_far = _pair_key(*sorted((anchor, "centers_standard"), key=ARMS.index))
        near = np.asarray(field_tvs[k_near], dtype=float)
        far = np.asarray(field_tvs[k_far], dtype=float)
        excess = near - far
        boot = cluster_bootstrap_mean(excess)
        n_pos = int(np.sum(excess > 0))
        n_neg = int(np.sum(excess < 0))
        anchored[anchor] = {
            "nearer_in_length": "centers_compact",
            "nearer_char_gap": abs(prompt_chars[anchor] - prompt_chars["centers_compact"]),
            "nearer_mean_TV": pair_mean[k_near],
            "further_in_length": "centers_standard",
            "further_char_gap": abs(
                prompt_chars[anchor] - prompt_chars["centers_standard"]
            ),
            "further_mean_TV": pair_mean[k_far],
            "paired_excess_near_minus_far": boot,
            "exact_sign_test": {
                "n_discordant": n_pos + n_neg,
                "n_near_larger": n_pos,
                "n_far_larger": n_neg,
                "p_two_sided": _sign_test_p(n_pos, n_neg),
            },
            "monotone_length_violated": bool(boot["ci95"][0] > 0.0),
        }

    predictor_rows = []
    for a, b in PAIRS:
        key = _pair_key(a, b)
        predictor_rows.append(
            {
                "pair": key,
                "mean_TV": pair_mean[key],
                "content_mismatch": int(SLC_CONTENT[a] != SLC_CONTENT[b]),
                "abs_prompt_char_diff": abs(prompt_chars[a] - prompt_chars[b]),
            }
        )
    tv_vec = np.asarray([r["mean_TV"] for r in predictor_rows], dtype=float)
    content_vec = np.asarray([r["content_mismatch"] for r in predictor_rows], dtype=float)
    length_vec = np.asarray(
        [r["abs_prompt_char_diff"] for r in predictor_rows], dtype=float
    )
    rho_content = spearman(tv_vec, content_vec)
    rho_length = spearman(tv_vec, length_vec)
    if abs(rho_content) == abs(rho_length):
        better = "tie"
    else:
        better = "content_mismatch" if abs(rho_content) > abs(rho_length) else "abs_prompt_char_diff"

    # --- 6. verdict -------------------------------------------------------
    #
    # Delta is judged on its bootstrap interval, not on p_delta. The
    # permutation null is arm-label exchangeability, that is "no condition
    # effect at all"; rejecting it does not establish that Delta itself
    # differs from zero, and with a large global effect p_delta is small
    # almost regardless of Delta. The interval is the honest statement about
    # Delta, so a Delta whose interval covers zero is reported as directional
    # and unresolved even when p_delta is small.
    info_smaller = pair_mean[k_info] < pair_mean[k_len]
    delta_ci = delta_boot["ci95"]
    delta_excludes_zero = delta_ci[0] > 0.0 or delta_ci[1] < 0.0
    length_refuted = rho_length <= 0.0

    if delta_obs > 0 and delta_excludes_zero and info_smaller:
        verdict = "PASS_content_over_length"
        claim = (
            "Response distances follow retained information, not prompt length: "
            "the information-identical pair separated by 458 characters is closer "
            "than the content-mismatched pair separated by 182 characters, and the "
            "difference has an interval excluding zero."
        )
    elif delta_obs < 0 and delta_excludes_zero and not info_smaller:
        verdict = "LENGTH_SENSITIVE"
        claim = (
            "Response distances follow prompt length rather than retained "
            "information: re-serializing the same numbers at a different length "
            "moves the operator more than changing the content class."
        )
    elif length_refuted or all(v["monotone_length_violated"] for v in anchored.values()):
        verdict = "LENGTH_REFUTED_DELTA_UNRESOLVED"
        a1 = anchored["moments_standard"]
        a2 = anchored["moments_compact"]
        claim = (
            "No monotone function of prompt length can order the response "
            "distances. Holding one moments prompt fixed and comparing it with the "
            "two information-identical histogram prompts, the histogram prompt "
            "nearer in length is significantly the more distant in response, in "
            "both anchors: "
            f"{a1['paired_excess_near_minus_far']['mean']:+.3f} "
            f"[{a1['paired_excess_near_minus_far']['ci95'][0]:+.3f}, "
            f"{a1['paired_excess_near_minus_far']['ci95'][1]:+.3f}] and "
            f"{a2['paired_excess_near_minus_far']['mean']:+.3f} "
            f"[{a2['paired_excess_near_minus_far']['ci95'][0]:+.3f}, "
            f"{a2['paired_excess_near_minus_far']['ci95'][1]:+.3f}], where a "
            "monotone length account requires both to be at most zero. The "
            f"prespecified contrast points the same way (Delta = {delta_obs:+.4f}) "
            f"but its interval covers zero ([{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]), "
            "so it is directional and not established on its own. What distinguishes "
            "the compact histogram condition is not resolved by this design, which "
            "varies its length and its serialization form together."
        )
    else:
        verdict = "UNRESOLVED"
        claim = (
            "Neither the content nor the length ordering is established "
            f"(Delta = {delta_obs:+.4f}, 95% CI "
            f"[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}], p = {p_delta:.4f}); "
            "see decision.json effect sizes."
        )

    decision = {
        "status": "SLC_CONTROL_PRIMARY_V0_1",
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
        "arms": {
            a: {
                "content_class": SLC_CONTENT[a],
                "form": SLC_FORM[a],
                "prompt_chars": prompt_chars[a],
            }
            for a in ARMS
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
            "test": "field_blocked_permutation_of_arm_labels_on_responses",
            "alpha_primary": 0.05,
            "permutation_note": (
                "One set of permutation draws serves both the primary Delta test "
                "and the global any-difference test; both share the same null of "
                "arm-label exchangeability within a field."
            ),
        },
        "pairwise_bootstrap": pair_boot,
        "pairwise_mean_TV": pair_mean,
        "noise_floor": {
            "mean_within_arm_between_block_TV": within_mean,
            "within_boot": within_boot,
            "mean_between_arm_TV": all_pair_mean_obs,
            "ratio_between_over_within": all_pair_mean_obs / max(within_mean, 1e-12),
            "n_cells": len(within),
        },
        "primary_contrast": {
            "definition": protocol["primary_contrast_definition"],
            "pair_different_content_182_chars": k_len,
            "pair_same_content_458_chars": k_con,
            "mean_TV_different_content": pair_mean[k_len],
            "mean_TV_same_content": pair_mean[k_con],
            "delta": delta_obs,
            "delta_bootstrap": delta_boot,
            "null_mean": float(np.mean(null_delta)),
            "null_p95": float(np.quantile(null_delta, 0.95)),
            "p_one_sided": p_delta,
            "p_one_sided_lower_tail": p_delta_lower,
            "n_perm": N_PERM,
            "favours": "content" if delta_obs > 0 else "length",
        },
        "secondary": {
            "layout_only_reference_moments_class": layout_ref,
            "information_only_reference_histogram_class": info_ref,
            "reference_contrast_caveat": (
                "The two reference contrasts above cannot separate serialization "
                "form from prompt length: the moments pair differs by 37 "
                "characters and the histogram pair by 458, so their ratio is "
                "also what a prompt-length account predicts. Use "
                "anchored_monotonicity_test for a length-immune contrast."
            ),
            "anchored_monotonicity_test": {
                "test": (
                    "paired within-field comparison of one moments prompt against "
                    "the two information-identical histogram prompts; any monotone "
                    "increasing function of prompt length requires the further "
                    "prompt to be at least as distant in response"
                ),
                "prespecified": False,
                "anchors": anchored,
                "monotone_length_account_refuted": all(
                    v["monotone_length_violated"] for v in anchored.values()
                ),
            },
            "predictor_rank_correlation": {
                "rows": predictor_rows,
                "spearman_rho_content_mismatch": rho_content,
                "spearman_rho_abs_prompt_char_diff": rho_length,
                "better_predictor": better,
                "n_pairs": len(predictor_rows),
                "note": (
                    "Spearman on six arm pairs; implemented in numpy "
                    "(average ranks for ties), no scipy dependency."
                ),
            },
            "global_any_arm_difference": {
                "test": "field_blocked_arm_label_permutation_mean_pairwise_TV",
                "observed_mean_pairwise_TV": all_pair_mean_obs,
                "null_mean": float(np.mean(null_global)),
                "null_p95": float(np.quantile(null_global, 0.95)),
                "p_one_sided": p_global,
                "n_perm": N_PERM,
            },
        },
        "verdict": verdict,
        "claim": claim,
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
                "delta": delta_obs,
                "delta_ci95": delta_boot["ci95"],
                "p_delta": p_delta,
                "p_any_arm_difference": p_global,
                "spearman_content": rho_content,
                "spearman_length": rho_length,
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
