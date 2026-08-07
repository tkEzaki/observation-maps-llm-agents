"""Export offline figure assets: nulls, fieldwise TVs, prompts, R1 strips.

Recomputes field-blocked permutation nulls from existing traces (no API calls).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.matched_rep_collective.run_replay_primary_inference import (  # noqa: E402
    N_PERM,
    TARGETS,
    _stats_from_counts,
    _tv,
    aggregate,
    permute_target_labels_on_responses,
    target_perm_stat,
)
from analysis.matched_rep_collective.run_replay_primary_inference import RNG as REPLAY_RNG  # noqa: E402

OUT = ROOT / "analysis" / "figure_data" / "data"
OUT.mkdir(parents=True, exist_ok=True)

REPLAY_RUN = (
    ROOT
    / "runs/matched_rep_collective_replay"
    / "matched-rep-collective-3x3-replay-v0.1_1e976b79b71d"
    / "20260724T114515Z"
)
OMAP_RUN = (
    ROOT
    / "runs/matched_rep_collective_omap"
    / "matched-rep-omap-control-v0.1-gpt-5.4-mini"
)
# fix path
OMAP_RUN = (
    ROOT
    / "runs/matched_rep_collective_omap"
    / "matched-rep-omap-control-v0.1_gpt-5.4-mini"
    / "20260724T161234Z"
)
FIELDS = ROOT / "analysis/matched_rep_collective/replay_fields_v0_1.json"
VARIANTS = (
    "moments_original",
    "moments_reformatted",
    "moments_length_matched",
)
OMAP_RNG = np.random.default_rng(20260725)


def _load_trace(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def export_replay_null() -> None:
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["fields"]
    rows = _load_trace(REPLAY_RUN)
    panel = aggregate(fields, rows)["panel"]
    obs = target_perm_stat(panel)
    null_tv = np.empty(N_PERM)
    for i in range(N_PERM):
        null_tv[i] = target_perm_stat(permute_target_labels_on_responses(panel))
    np.save(OUT / "replay_null_tv.npy", null_tv)
    # per-field between vs within mean
    between = []
    within = []
    field_ids = []
    for fr in panel:
        ps = [np.asarray(fr["targets"][t]["p"], dtype=float) for t in TARGETS]
        bet = [_tv(ps[i], ps[j]) for i in range(3) for j in range(i + 1, 3)]
        wit = []
        for t in TARGETS:
            p0 = np.asarray(fr["blocks"][f"{t}|b0"]["p"], dtype=float)
            p1 = np.asarray(fr["blocks"][f"{t}|b1"]["p"], dtype=float)
            wit.append(_tv(p0, p1))
        between.append(float(np.mean(bet)))
        within.append(float(np.mean(wit)))
        field_ids.append(fr["field_id"])
    pd.DataFrame(
        {"field_id": field_ids, "between_target_TV": between, "within_block_TV": within}
    ).to_csv(OUT / "replay_field_noise_pairs.csv", index=False)
    meta = {
        "observed_mean_pairwise_TV": float(obs),
        "null_mean": float(np.mean(null_tv)),
        "null_p95": float(np.quantile(null_tv, 0.95)),
        "n_perm": int(N_PERM),
        "p_one_sided": float((np.sum(null_tv >= obs) + 1) / (N_PERM + 1)),
    }
    (OUT / "replay_null_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("replay null", meta)


def export_omap() -> None:
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["fields"]
    omap_rows = _load_trace(OMAP_RUN)
    by_ft = defaultdict(lambda: np.zeros(3, dtype=int))
    by_ftb = defaultdict(lambda: np.zeros(3, dtype=int))
    raw_by: dict[str, list] = defaultdict(list)
    for r in omap_rows:
        if not r.get("valid"):
            continue
        idx = {-1: 0, 0: 1, 1: 2}[int(r["action_value"])]
        fid = r["field_id"]
        t = r["target_representation"]
        b = int(r["acquisition_block"])
        by_ft[(fid, t)][idx] += 1
        by_ftb[(fid, t, b)][idx] += 1
        raw_by[fid].append((t, idx, b))

    panel = []
    for f in fields:
        fid = f["field_id"]
        entry = {
            "field_id": fid,
            "raw": raw_by[fid],
            "targets": {},
            "blocks": {},
        }
        for t in VARIANTS:
            entry["targets"][t] = _stats_from_counts(by_ft[(fid, t)])
            for b in (0, 1):
                entry["blocks"][f"{t}|b{b}"] = _stats_from_counts(by_ftb[(fid, t, b)])
        panel.append(entry)

    def mean_pairwise(panel_local):
        tvs = []
        for fr in panel_local:
            ps = [np.asarray(fr["targets"][t]["p"]) for t in VARIANTS]
            for i in range(3):
                for j in range(i + 1, 3):
                    tvs.append(_tv(ps[i], ps[j]))
        return float(np.mean(tvs)), tvs

    def permute(panel_local):
        out = []
        for fr in panel_local:
            raw = list(fr["raw"])
            labels = [t for t, _, _ in raw]
            OMAP_RNG.shuffle(labels)
            new_raw = [(labels[i], raw[i][1], raw[i][2]) for i in range(len(raw))]
            by_t = {t: np.zeros(3, dtype=int) for t in VARIANTS}
            by_tb = {(t, b): np.zeros(3, dtype=int) for t in VARIANTS for b in (0, 1)}
            for t, j, b in new_raw:
                by_t[t][j] += 1
                by_tb[(t, b)][j] += 1
            targets = {t: _stats_from_counts(by_t[t]) for t in VARIANTS}
            blocks = {
                f"{t}|b{b}": _stats_from_counts(by_tb[(t, b)])
                for t in VARIANTS
                for b in (0, 1)
            }
            out.append({**fr, "targets": targets, "blocks": blocks, "raw": new_raw})
        return out

    obs, _ = mean_pairwise(panel)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = mean_pairwise(permute(panel))[0]
    np.save(OUT / "omap_null_tv.npy", null)

    rows = []
    for fr in panel:
        o = np.asarray(fr["targets"]["moments_original"]["p"])
        r = np.asarray(fr["targets"]["moments_reformatted"]["p"])
        L = np.asarray(fr["targets"]["moments_length_matched"]["p"])
        rows.append(
            {
                "field_id": fr["field_id"],
                "dTV_orig_reformat": _tv(o, r),
                "dTV_orig_pad": _tv(o, L),
                "dTV_reformat_pad": _tv(r, L),
                "p_orig": o.tolist(),
                "p_reformat": r.tolist(),
                "p_pad": L.tolist(),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "omap_field_pairwise_tv.csv", index=False)
    meta = {
        "observed_mean_pairwise_TV": float(obs),
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "n_perm": int(N_PERM),
        "p_one_sided": float((np.sum(null >= obs) + 1) / (N_PERM + 1)),
    }
    (OUT / "omap_null_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("omap null", meta)


def export_r1_fieldwise() -> None:
    paths = {
        "gpt": ROOT
        / "analysis/matched_rep_collective/replay_primary/field_pairwise_tv.json",
        "claude": ROOT
        / "runs/matched_rep_collective_r1"
        / "matched-rep-r1-cross-model-v0.1_claude_claude-haiku-4-5-20251001"
        / "20260724T142109Z"
        / "primary_3x3_report.json",
        "gemini": ROOT
        / "runs/matched_rep_collective_r1"
        / "matched-rep-r1-cross-model-v0.1_gemini_gemini-3.5-flash"
        / "20260724T142109Z"
        / "primary_3x3_report.json",
    }
    rows = []
    # GPT from field_pairwise_tv
    gpt = json.loads(paths["gpt"].read_text(encoding="utf-8"))
    for f in gpt:
        mean_tv = (f["dTV_MC"] + f["dTV_MI"] + f["dTV_CI"]) / 3
        rows.append(
            {
                "family": "gpt",
                "field_id": f["field_id"],
                "stratum": f.get("stratum"),
                "mean_pairwise_TV": mean_tv,
                "dTV_MC": f["dTV_MC"],
                "dTV_MI": f["dTV_MI"],
                "dTV_CI": f["dTV_CI"],
            }
        )
    for fam in ("claude", "gemini"):
        report = json.loads(paths[fam].read_text(encoding="utf-8"))
        for fr in report["field_rows"]:
            pw = fr["pairwise"]
            # keys may vary
            vals = [v["d_TV"] for v in pw.values()]
            rows.append(
                {
                    "family": fam,
                    "field_id": fr["field_id"],
                    "stratum": fr.get("stratum"),
                    "mean_pairwise_TV": float(np.mean(vals)),
                    "dTV_MC": None,
                    "dTV_MI": None,
                    "dTV_CI": None,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "r1_fieldwise_mean_tv.csv", index=False)
    print("r1 fieldwise", len(rows))


def export_prompts() -> None:
    """Extract one shared-field prompt excerpt per representation from Stage B."""
    stim_paths = sorted(Path(ROOT / "runs/stimulus_manifold").rglob("stimuli.jsonl"))
    want = {
        "moments_m1_m3": None,
        "centers_24_standard": None,
        "intervals_24_decimal6": None,
    }
    chosen_offset = {}
    profile = "unimodal_k9"
    for path in stim_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                o = json.loads(line)
                rep = o.get("representation")
                if rep in want and want[rep] is None and o.get("profile") == profile:
                    want[rep] = o["serialized_prompt"]
                    chosen_offset[rep] = o.get("offset_index")
        if all(want.values()):
            break
    # GUARD - do not remove.
    #
    # The selection above matches on `profile` only and takes the *first*
    # record it meets per representation. Every profile is swept over many
    # rotations, so the three excerpts it picked came from three DIFFERENT
    # rotations of the field (offset_index 30 / 4 / 29) while main Figure 2
    # titled panel a "Same field" and panel b "Three observations of the same
    # field". The excerpts already written to
    # data/prompt_excerpts_unimodal.json - and therefore Figure 2 in
    # the legacy figure outputs - carry that defect (see DEPRECATED.md).
    #
    # The corrected selection is
    # analysis/figures_main/fig2_micro.py::_shared_field_records: it
    # pins all three representations to a single offset_index and then proves
    # physical identity (generating parameters, a derived physical_id, and the
    # 24-bin mass recovered from each serialisation) rather than trusting the
    # index label. See analysis/figure_data/DEPRECATED.md.
    #
    # Fail loudly here rather than silently regenerate mismatched excerpts.
    offsets = {rep: chosen_offset.get(rep) for rep in want if want[rep]}
    assert len(set(offsets.values())) == 1, (
        "export_prompts picked three different rotations of "
        f"{profile!r}: offset_index by representation {offsets}. Figure 2 "
        "claims one physical field observed three ways, so all three records "
        "must share one offset_index. Use the corrected selection in "
        "analysis/figures_main/fig2_micro.py::_shared_field_records "
        "instead of regenerating data/prompt_excerpts_unimodal.json from this "
        "exporter (see analysis/figure_data/DEPRECATED.md)."
    )
    excerpts = {}
    for rep, text in want.items():
        if not text:
            continue
        lines = text.strip().splitlines()
        head = lines[:5]
        data = [ln for ln in lines[5:] if ln.strip()][:10]
        excerpts[rep] = "\n".join(head + ["…"] + data)
        excerpts[rep + "__n_chars"] = len(text)
    (OUT / "prompt_excerpts_unimodal.json").write_text(
        json.dumps(excerpts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("prompts", [k for k in excerpts if not k.endswith("__n_chars")])


def main() -> int:
    print("Exporting figure data to", OUT)
    export_replay_null()
    export_omap()
    export_r1_fieldwise()
    export_prompts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
