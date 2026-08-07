"""Offline downsample of GPT replay to choose R1 n_response."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.matched_rep_collective.run_replay_primary_inference import (  # noqa: E402
    TARGETS,
    _tv,
    multinomial_deviance,
    target_perm_stat,
)

RUN = (
    ROOT
    / "runs/matched_rep_collective_replay"
    / "matched-rep-collective-3x3-replay-v0.1_1e976b79b71d"
    / "20260724T114515Z"
)
FIELDS = ROOT / "analysis/matched_rep_collective/replay_fields_v0_1.json"
OUT = ROOT / "analysis/matched_rep_collective/r1_downsample"
NS = (8, 12, 16, 24, 32)
N_MONTE = 80
N_PERM = 400
RNG = np.random.default_rng(202607241)


def _load() -> tuple[list[dict], dict]:
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["fields"]
    by = defaultdict(list)
    with (RUN / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if not r.get("valid"):
                continue
            key = (r["field_id"], r["target_representation"], int(r["acquisition_block"]))
            by[key].append(int(r["action_value"]))
    return fields, by


def _panel_from_subsample(fields, by, n_per_cond: int, rng: np.random.Generator):
    """n_per_cond total responses per field×target, split across 2 blocks."""
    assert n_per_cond % 2 == 0
    per_block = n_per_cond // 2
    panel = []
    for f in fields:
        fid = f["field_id"]
        targets = {}
        blocks = {}
        raw = []
        for t in TARGETS:
            counts = np.zeros(3, dtype=int)
            for b in (0, 1):
                pool = by[(fid, t, b)]
                take = min(per_block, len(pool))
                idx = rng.choice(len(pool), size=take, replace=False)
                chosen = [pool[i] for i in idx]
                bc = np.zeros(3, dtype=int)
                for v in chosen:
                    j = {-1: 0, 0: 1, 1: 2}[v]
                    counts[j] += 1
                    bc[j] += 1
                    raw.append((t, j, b))
                p = bc / bc.sum() if bc.sum() else np.full(3, np.nan)
                blocks[f"{t}|b{b}"] = {
                    "n": bc.tolist(),
                    "p": p.tolist(),
                    "a0": float((-1) * p[0] + p[2]) if np.all(np.isfinite(p)) else float("nan"),
                    "A": float(p[0] + p[2]) if np.all(np.isfinite(p)) else float("nan"),
                }
            p = counts / counts.sum()
            targets[t] = {
                "n": counts.tolist(),
                "p": p.tolist(),
                "a0": float((-1) * p[0] + p[2]),
                "A": float(p[0] + p[2]),
            }
        panel.append(
            {
                "field_id": fid,
                "source": f["source_representation"],
                "stratum": f["stratum"],
                "K": float(f["K"]),
                "targets": targets,
                "blocks": blocks,
                "raw": raw,
            }
        )
    return panel


def permute_targets(panel, rng):
    out = []
    for fr in panel:
        raw = list(fr["raw"])
        labels = [t for t, _, _ in raw]
        rng.shuffle(labels)
        new_raw = [(labels[i], raw[i][1], raw[i][2]) for i in range(len(raw))]
        # rebuild
        by_t = {t: np.zeros(3, dtype=int) for t in TARGETS}
        by_tb = {(t, b): np.zeros(3, dtype=int) for t in TARGETS for b in (0, 1)}
        for t, j, b in new_raw:
            by_t[t][j] += 1
            by_tb[(t, b)][j] += 1
        targets = {}
        blocks = {}
        for t in TARGETS:
            c = by_t[t]
            p = c / c.sum()
            targets[t] = {
                "n": c.tolist(),
                "p": p.tolist(),
                "a0": float((-1) * p[0] + p[2]),
                "A": float(p[0] + p[2]),
            }
            for b in (0, 1):
                bc = by_tb[(t, b)]
                pb = bc / bc.sum() if bc.sum() else np.full(3, np.nan)
                blocks[f"{t}|b{b}"] = {"n": bc.tolist(), "p": pb.tolist()}
        out.append({**fr, "targets": targets, "blocks": blocks, "raw": new_raw})
    return out


def noise_ratio(panel):
    bet, wit = [], []
    for fr in panel:
        ps = [np.asarray(fr["targets"][t]["p"]) for t in TARGETS]
        for i in range(3):
            for j in range(i + 1, 3):
                bet.append(_tv(ps[i], ps[j]))
        for t in TARGETS:
            p0 = np.asarray(fr["blocks"][f"{t}|b0"]["p"])
            p1 = np.asarray(fr["blocks"][f"{t}|b1"]["p"])
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                wit.append(_tv(p0, p1))
    return float(np.mean(bet) / max(np.mean(wit), 1e-12)), float(np.mean(bet)), float(np.mean(wit))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fields, by = _load()
    rows = []
    for n in NS:
        dets = []
        ratios = []
        mean_tvs = []
        ci_widths = []
        for m in range(N_MONTE):
            rng = np.random.default_rng(int(RNG.integers(1, 2**31 - 1)))
            panel = _panel_from_subsample(fields, by, n, rng)
            obs = target_perm_stat(panel)
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                null[i] = target_perm_stat(permute_targets(panel, rng))
            p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
            dets.append(p < 0.05)
            ratio, mb, mw = noise_ratio(panel)
            ratios.append(ratio)
            mean_tvs.append(mb)
            # crude CI width via field bootstrap of pairwise mean TV
            field_means = []
            for fr in panel:
                ps = [np.asarray(fr["targets"][t]["p"]) for t in TARGETS]
                tvs = [_tv(ps[i], ps[j]) for i in range(3) for j in range(i + 1, 3)]
                field_means.append(np.mean(tvs))
            field_means = np.asarray(field_means)
            boots = [
                float(np.mean(field_means[rng.integers(0, 48, size=48)]))
                for _ in range(300)
            ]
            ci_widths.append(float(np.quantile(boots, 0.975) - np.quantile(boots, 0.025)))
        row = {
            "n_response": n,
            "n_calls": 48 * 3 * n,
            "detection_rate_p05": float(np.mean(dets)),
            "mean_noise_ratio": float(np.mean(ratios)),
            "mean_between_TV": float(np.mean(mean_tvs)),
            "mean_ci_width_pairwise_TV": float(np.mean(ci_widths)),
            "n_monte": N_MONTE,
            "n_perm": N_PERM,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    # Recommend smallest n with detection_rate>=0.95 and mean_noise_ratio>=2
    rec = None
    for row in rows:
        if row["detection_rate_p05"] >= 0.95 and row["mean_noise_ratio"] >= 2.0:
            rec = row["n_response"]
            break
    if rec is None:
        rec = 32
    report = {
        "status": "r1_downsample_v0_1",
        "recommendation_n_response": rec,
        "recommendation_blocks": f"{rec // 2}x2",
        "recommendation_calls_per_model": 48 * 3 * rec,
        "rows": rows,
        "note": (
            "Offline GPT-replay subsample; choose smallest n with "
            "detection_rate>=0.95 and mean between/within TV ratio>=2."
        ),
    }
    (OUT / "downsample_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("RECOMMEND", rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
