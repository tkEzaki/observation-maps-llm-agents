"""Select median-TV stratum anchors from frozen multi-family replay (no API)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.matched_rep_collective.run_replay_primary_inference import (  # noqa: E402
    TARGETS,
    _tv,
    aggregate,
)

OUT = ROOT / "analysis" / "matched_rep_collective" / "anchors"
FIELDS = ROOT / "analysis/matched_rep_collective/replay_fields_v0_1.json"
STRATA = [
    "early_transient",
    "pre_onset",
    "post_onset",
    "ordered_state",
    "negative_K",
    "high_r2",
]
RUNS = {
    "gpt": ROOT
    / "runs/matched_rep_collective_replay"
    / "matched-rep-collective-3x3-replay-v0.1_1e976b79b71d"
    / "20260724T114515Z",
    "claude": ROOT
    / "runs/matched_rep_collective_r1"
    / "matched-rep-r1-cross-model-v0.1_claude_claude-haiku-4-5-20251001"
    / "20260724T142109Z",
    "gemini": ROOT
    / "runs/matched_rep_collective_r1"
    / "matched-rep-r1-cross-model-v0.1_gemini_gemini-3.5-flash"
    / "20260724T142109Z",
}
SHORT = {
    "moments_m1_m3": "M",
    "centers_24_standard": "C",
    "intervals_24_decimal6": "I",
}


def _load_rows(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def field_mean_tv(fr: dict) -> float:
    ps = [np.asarray(fr["targets"][t]["p"], dtype=float) for t in TARGETS]
    tvs = [_tv(ps[i], ps[j]) for i in range(3) for j in range(i + 1, 3)]
    return float(np.mean(tvs))


def select_median_anchors(fields: list[dict], gpt_panel: list[dict]) -> list[dict]:
    """Preregistered rule: within each stratum, pick field with median GPT mean TV.

    Ties: lowest field_id lexicographically.
    Physical fields are NOT inspected visually.
    """
    by_stratum: dict[str, list[tuple[float, str, dict, dict]]] = defaultdict(list)
    meta = {f["field_id"]: f for f in fields}
    for fr in gpt_panel:
        fid = fr["field_id"]
        st = meta[fid]["stratum"]
        by_stratum[st].append((field_mean_tv(fr), fid, meta[fid], fr))
    selected = []
    for st in STRATA:
        pool = sorted(by_stratum[st], key=lambda x: (x[0], x[1]))
        if not pool:
            continue
        # median index (lower median for even n)
        idx = (len(pool) - 1) // 2
        tv, fid, meta_f, fr = pool[idx]
        selected.append(
            {
                "stratum": st,
                "field_id": fid,
                "selection_rule": "median_gpt_mean_pairwise_TV_within_stratum",
                "gpt_mean_pairwise_TV": tv,
                "K": meta_f["K"],
                "source_representation": meta_f["source_representation"],
                "physical_hash": meta_f["physical_hash"],
                "r1": meta_f["r1"],
                "r2": meta_f["r2"],
                "pool_n": len(pool),
                "pool_tvs": [p[0] for p in pool],
            }
        )
    return selected


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["fields"]
    panels = {
        fam: aggregate(fields, _load_rows(run))["panel"] for fam, run in RUNS.items()
    }
    # index by field_id
    by_id = {
        fam: {fr["field_id"]: fr for fr in panel} for fam, panel in panels.items()
    }
    anchors = select_median_anchors(fields, panels["gpt"])

    # Attach trinomials for all families
    for a in anchors:
        fid = a["field_id"]
        a["responses"] = {}
        for fam in RUNS:
            fr = by_id[fam][fid]
            a["responses"][fam] = {
                t: {
                    "p": fr["targets"][t]["p"],
                    "A": fr["targets"][t]["A"],
                    "a0": fr["targets"][t]["a0"],
                    "n": fr["targets"][t]["n"],
                }
                for t in TARGETS
            }

    (OUT / "anchor_selection_v0_1.json").write_text(
        json.dumps(
            {
                "status": "anchors_v0_1_unpaid",
                "rule": "median_gpt_mean_pairwise_TV_within_each_frozen_stratum",
                "tie_break": "lowest_field_id",
                "n_anchors": len(anchors),
                "anchors": anchors,
                "note": "No additional paid calls; drawn from frozen GPT/Claude/Gemini traces.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Figure: per-stratum stacked bars of p for each family×target
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    colors = {"M": "#2c5f7c", "C": "#c44e52", "I": "#6a9c78"}
    for ax, a in zip(axes.ravel(), anchors):
        fams = ["gpt", "claude", "gemini"]
        x = np.arange(len(fams))
        width = 0.25
        for i, t in enumerate(TARGETS):
            # show activity A as bar height alternative? User asked trinomial.
            # Plot p_stay as main readable channel + markers for signed a0
            stays = [a["responses"][f][t]["p"][1] for f in fams]
            ax.bar(
                x + (i - 1) * width,
                stays,
                width=width,
                color=colors[SHORT[t]],
                alpha=0.85,
                label=SHORT[t] if ax is axes[0, 0] else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["GPT", "Claude", "Gemini"], fontsize=8)
        ax.set_title(f"{a['stratum']}\n{a['field_id']}", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel(r"$p_{\mathrm{stay}}$")
    axes[1, 0].set_ylabel(r"$p_{\mathrm{stay}}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Anchor phenotypes — median-TV field per stratum (unpaid reuse)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_anchors_pstay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Second figure: mean pairwise TV for each anchor×family
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(anchors))
    width = 0.25
    for i, fam in enumerate(["gpt", "claude", "gemini"]):
        vals = []
        for a in anchors:
            ps = [
                np.asarray(a["responses"][fam][t]["p"], dtype=float) for t in TARGETS
            ]
            vals.append(float(np.mean([_tv(ps[j], ps[k]) for j in range(3) for k in range(j + 1, 3)])))
        ax.bar(x + (i - 1) * width, vals, width=width, label=fam)
    ax.set_xticks(x)
    ax.set_xticklabels([a["stratum"].replace("_", "\n") for a in anchors], fontsize=8)
    ax.set_ylabel("mean pairwise TV")
    ax.set_title("Anchor fields — operator separation by family")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_anchors_tv.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"n_anchors": len(anchors), "ids": [a["field_id"] for a in anchors]}, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
