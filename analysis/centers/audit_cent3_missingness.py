"""CENT-4A: parse / missingness audit for CENT-3 traces (offline).

Notes on recoverable raw text:
  Current CENT-3 traces set raw_response=None on all 73 invalids because
  run_task discarded failed backend text. Token counters show each failed
  attempt consumed max_output_tokens, so the backend returned *something*,
  but the exact string is not recoverable without re-acquisition.
  Future runs retain last_raw_response (see experiments/.../run.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.centers.cent3_io import (  # noqa: E402
    BRANCH,
    field_feature_row,
    latest_cent3_runs,
    load_catalog,
)
from circlemap.response import ACTION_VALUES, parse_social_action  # noqa: E402

ACTION_RE = re.compile(
    r"(advance|stay|retard)",
    flags=re.IGNORECASE,
)


def classify_raw(raw: str | None, errors: list[str]) -> dict:
    """Deterministic taxonomy; applied uniformly when raw text exists."""
    if raw is None:
        return {
            "class": "raw_unavailable",
            "semantic_action": None,
            "would_relaxed_parse": False,
            "detail": "trace discarded failed raw_response",
        }
    text = raw if isinstance(raw, str) else str(raw)
    if text.strip() == "":
        return {
            "class": "empty_output",
            "semantic_action": None,
            "would_relaxed_parse": False,
            "detail": "empty string",
        }
    try:
        parse_social_action(text)
        return {
            "class": "strict_ok_but_marked_invalid",
            "semantic_action": None,
            "would_relaxed_parse": True,
            "detail": "strict parse succeeds offline (trace inconsistency)",
        }
    except ValueError:
        pass

    # Truncation / prose / multi-action heuristics (uniform rules).
    actions = [m.group(1).lower() for m in ACTION_RE.finditer(text)]
    unique = []
    for a in actions:
        if a not in unique:
            unique.append(a)

    relaxed_label = None
    would_relaxed = False
    # Deterministic relaxed rule: single JSON-ish object with only social_action,
    # allowing surrounding whitespace/markdown fences; or bare action token.
    stripped = text.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidate = fence.group(1) if fence else stripped
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict) and "social_action" in payload:
            label = payload["social_action"]
            if isinstance(label, str) and label.lower() in ACTION_VALUES:
                relaxed_label = label.lower()
                would_relaxed = True
    except json.JSONDecodeError:
        if len(unique) == 1 and re.fullmatch(
            r'["\']?(advance|stay|retard)["\']?',
            stripped,
            flags=re.IGNORECASE,
        ):
            relaxed_label = unique[0]
            would_relaxed = True

    if len(unique) > 1:
        cls = "multiple_actions"
    elif would_relaxed and relaxed_label is not None:
        cls = "semantic_ok_format_fail"
    elif actions and not would_relaxed:
        cls = "prose_with_action_mention"
    elif len(text) >= 40 or "\n" in text:
        cls = "prose_or_explanation"
    else:
        cls = "malformed_or_truncated"

    err_join = " | ".join(errors)
    if "JSON" in err_join and cls == "malformed_or_truncated":
        detail = "strict JSON failure"
    else:
        detail = err_join[:200]

    return {
        "class": cls,
        "semantic_action": relaxed_label,
        "would_relaxed_parse": would_relaxed,
        "detail": detail,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir

    load_catalog(branch / "cent3_stimulus_catalog_v0.json")
    fields = json.loads((branch / "cent3_frozen_fields_v0.json").read_text(encoding="utf-8"))[
        "fields"
    ]
    lock = json.loads((branch / "cent3_prospective_lock_v0.json").read_text(encoding="utf-8"))
    lock_by = {row["field_id"]: row for row in lock["rows"]}
    field_by = {f["field_id"]: f for f in fields}

    run_dirs = latest_cent3_runs()
    if len(run_dirs) < 2:
        raise SystemExit(f"need 2 cent3 runs, found {run_dirs}")

    # Per call records
    call_rows = []
    class_counter: Counter[str] = Counter()
    semantic_rescue = 0
    for run_dir in run_dirs:
        block = "block_1" if "block_1" in run_dir.parent.name else "block_2"
        with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                fid = rec["profile"]
                meta = classify_raw(rec.get("raw_response"), list(rec.get("errors") or []))
                if not rec.get("valid"):
                    class_counter[meta["class"]] += 1
                    if meta["would_relaxed_parse"]:
                        semantic_rescue += 1
                call_rows.append(
                    {
                        "field_id": fid,
                        "block": block,
                        "valid_strict": bool(rec.get("valid")),
                        "action_label_strict": rec.get("action_label"),
                        "raw_response": rec.get("raw_response"),
                        "attempts": rec.get("attempts"),
                        "input_tokens": rec.get("input_tokens"),
                        "output_tokens": rec.get("output_tokens"),
                        "errors": " || ".join(rec.get("errors") or []),
                        "invalid_class": meta["class"] if not rec.get("valid") else "valid",
                        "semantic_action_relaxed": meta["semantic_action"],
                        "would_relaxed_parse": meta["would_relaxed_parse"],
                        "class_detail": meta["detail"],
                    }
                )

    # Field × block rates
    rate_rows = []
    for fid, field in field_by.items():
        x, unresolved, peer, sparsity = field_feature_row(field)
        locked = lock_by[fid]
        for block in ("block_1", "block_2"):
            subset = [r for r in call_rows if r["field_id"] == fid and r["block"] == block]
            n = len(subset)
            n_inv = sum(1 for r in subset if not r["valid_strict"])
            rate_rows.append(
                {
                    "field_id": fid,
                    "block": block,
                    "bucket": field["bucket"],
                    "role": field["role"],
                    "peer_count": peer,
                    "bin_sparsity": sparsity,
                    "n_calls": n,
                    "n_invalid": n_inv,
                    "invalid_rate": n_inv / n if n else float("nan"),
                    "n_valid_strict": n - n_inv,
                    "R_pre": locked["R_pre"],
                    "p_stay_pre": locked["p_stay_pre"],
                    "p_minus_pre": locked["p_minus_pre"],
                    "p_plus_pre": locked["p_plus_pre"],
                    "abs_z1": locked["abs_z1"],
                    "d_shape": locked["d_shape"],
                    "d_native": locked["d_native"],
                    "unresolved_near_zero": int(unresolved),
                }
            )

    # Correlations (field-pooled across blocks)
    pooled = {}
    for row in rate_rows:
        fid = row["field_id"]
        slot = pooled.setdefault(
            fid,
            {
                "n_calls": 0,
                "n_invalid": 0,
                "R_pre": row["R_pre"],
                "bin_sparsity": row["bin_sparsity"],
                "peer_count": row["peer_count"],
                "p_stay_pre": row["p_stay_pre"],
            },
        )
        slot["n_calls"] += row["n_calls"]
        slot["n_invalid"] += row["n_invalid"]
    fids = sorted(pooled)
    inv_rate = np.asarray(
        [pooled[f]["n_invalid"] / max(pooled[f]["n_calls"], 1) for f in fids],
        dtype=np.float64,
    )
    r_pre = np.asarray([pooled[f]["R_pre"] for f in fids], dtype=np.float64)
    spars = np.asarray([pooled[f]["bin_sparsity"] for f in fids], dtype=np.float64)
    stay_pre = np.asarray([pooled[f]["p_stay_pre"] for f in fids], dtype=np.float64)
    peer = np.asarray([pooled[f]["peer_count"] for f in fids], dtype=np.float64)

    sparse_ctrl = pooled.get("cent_ctrl_sparse_antipodal", {})
    sparse_valid = sparse_ctrl.get("n_calls", 0) - sparse_ctrl.get("n_invalid", 0)

    summary = {
        "status": "cent4a_missingness_audit",
        "n_calls": len(call_rows),
        "n_invalid_strict": sum(1 for r in call_rows if not r["valid_strict"]),
        "invalid_class_counts": dict(class_counter),
        "semantic_format_fail_recoverable_now": semantic_rescue,
        "raw_text_recoverability": {
            "cent3_traces": "none — all invalids have raw_response=null",
            "evidence_of_nonzero_output": (
                "all 73 invalids have output_tokens=48 (=3×max_output_tokens=16) "
                "and input_tokens=1536 (=3×512), so attempts reached the backend"
            ),
            "pipeline_fix": (
                "run_task now retains last_raw_response on failure for future runs"
            ),
            "reparse_policy": (
                "If raw text becomes available, apply classify_raw uniformly to all "
                "fields; keep strict counts alongside any relaxed counts; never "
                "hand-rescue a single control."
            ),
        },
        "correlations_field_level": {
            "spearman_invalid_rate_vs_R_pre": _spearman(inv_rate, r_pre),
            "spearman_invalid_rate_vs_bin_sparsity": _spearman(inv_rate, spars),
            "spearman_invalid_rate_vs_p_stay_pre": _spearman(inv_rate, stay_pre),
            "spearman_invalid_rate_vs_peer_count": _spearman(inv_rate, peer),
            "mean_invalid_rate_peer8": float(
                np.mean(inv_rate[peer == 8]) if np.any(peer == 8) else float("nan")
            ),
            "mean_invalid_rate_peer16": float(
                np.mean(inv_rate[peer == 16]) if np.any(peer == 16) else float("nan")
            ),
        },
        "sparse_antipodal_control": {
            "n_valid_strict": int(sparse_valid),
            "n_calls": int(sparse_ctrl.get("n_calls", 0)),
            "phenotype_gate_decidable": bool(sparse_valid >= 12),
            "policy": (
                "keep field; mark phenotype freeze gate undecidable if n_valid < 12"
            ),
        },
        "top_invalid_fields": sorted(
            [
                {
                    "field_id": f,
                    "invalid_rate": pooled[f]["n_invalid"] / max(pooled[f]["n_calls"], 1),
                    "n_invalid": pooled[f]["n_invalid"],
                    "n_calls": pooled[f]["n_calls"],
                    "R_pre": pooled[f]["R_pre"],
                    "bin_sparsity": pooled[f]["bin_sparsity"],
                    "peer_count": pooled[f]["peer_count"],
                }
                for f in fids
            ],
            key=lambda r: (-r["invalid_rate"], r["field_id"]),
        )[:12],
    }

    out_dir = branch
    with (out_dir / "cent4a_invalid_by_field_block.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rate_rows)

    # Only write invalid call detail (full 1152 is large but ok; keep all for audit)
    with (out_dir / "cent4a_call_parse_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(call_rows[0].keys()))
        writer.writeheader()
        writer.writerows(call_rows)

    (out_dir / "cent4a_missingness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
