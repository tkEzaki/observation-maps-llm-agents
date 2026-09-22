"""Build Nat Commun main figures (submission revision; new figure order).

Order:
  1 macro phases
  2 microscopic operators
  3 identical-field replay
  4 cross-family replication
  5 same-task-info control
  6 compressibility ≠ transportability
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.natcomm_figures import (  # noqa: E402
    fig1_macro,
    fig2_micro,
    fig3_replay,
    fig4_cross_family,
    fig5_omap,
    fig6_compressibility,
)
from analysis.natcomm_figures.style import FIG_DIR  # noqa: E402

# Remove obsolete stems from prior numbering if present
OBSOLETE = [
    "fig3_compressibility_transportability",
    "fig4_replay_operator",
    "fig5_cross_family",
    "fig6_observation_map_control",
]

BUILDERS = {
    "1": ("fig1_collective_phases", fig1_macro.build),
    "2": ("fig2_microscopic_operators", fig2_micro.build),
    "3": ("fig3_replay_operator", fig3_replay.build),
    "4": ("fig4_cross_family", fig4_cross_family.build),
    "5": ("fig5_observation_map_control", fig5_omap.build),
    "6": ("fig6_compressibility_transportability", fig6_compressibility.build),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    wanted = set(args.only) if args.only else set(BUILDERS)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stem in OBSOLETE:
        for ext in (".png", ".pdf"):
            p = FIG_DIR / f"{stem}{ext}"
            if p.exists() and stem not in {b[0] for b in BUILDERS.values()}:
                # keep if still a current stem (fig4_cross_family etc. reused)
                pass
    # Explicit cleanup of truly obsolete names only
    for stem in (
        "fig3_compressibility_transportability",  # old fig3 name before reorder
    ):
        # After reorder fig6 uses this stem — do NOT delete
        pass

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "order": [
            "macro phases",
            "microscopic operators",
            "identical-field replay",
            "cross-family replication",
            "same-task-info control",
            "compressibility ≠ transportability",
        ],
        "figures": {},
    }
    for key in sorted(wanted, key=lambda x: int(x)):
        if key not in BUILDERS:
            print(f"skip unknown figure {key}")
            continue
        stem, fn = BUILDERS[key]
        print(f"building Figure {key}: {stem} ...")
        t0 = time.time()
        path = fn()
        dt = time.time() - t0
        print(f"  -> {path} ({dt:.1f}s)")
        manifest["figures"][stem] = {
            "png": str(path.relative_to(ROOT)).replace("\\", "/"),
            "seconds": round(dt, 2),
        }

    (FIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (FIG_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Nat Commun main figures (submission revision)",
                "",
                "Rebuild: `py -m analysis.natcomm_figures.build_all`",
                "Export supporting nulls/fieldwise: `py -m analysis.natcomm_figures.export_figure_data`",
                "",
                "Plan: `docs/NATCOMM_FIGURE_PLAN.md`",
                "",
                "## Order",
                "",
                "1. `fig1_collective_phases` — matched macro phases",
                "2. `fig2_microscopic_operators` — Stage-B operators",
                "3. `fig3_replay_operator` — identical-field replay",
                "4. `fig4_cross_family` — GPT/Claude/Gemini",
                "5. `fig5_observation_map_control` — same-task-info control",
                "6. `fig6_compressibility_transportability` — C ≠ T methodology",
                "",
                "Figure titles/captions live in the manuscript; panels only in these files.",
                "Colors: moments=blue, centers=orange, intervals=teal; action $p_+/p_-$ ≠ intervals green.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"manifest: {FIG_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
