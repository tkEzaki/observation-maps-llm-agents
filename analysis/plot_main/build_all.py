"""Build the redesigned main-text figures.

Usage:
  py -m analysis.plot_main.build_all [--only 1 2 ...]

Outputs to the earlier figure layout (PNG 450 dpi + vector PDF, true 180 mm width).
Reads the same source data as analysis.figure_assets.
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

from analysis.plot_main import (  # noqa: E402
    fig1_macro,
    fig2_micro,
    fig3_replay,
    fig4_cross_family,
    fig5_omap,
    fig6_compressibility,
)
from analysis.plot_main.style import FIG_DIR  # noqa: E402

BUILDERS = {
    "1": ("fig1_collective_phases", fig1_macro.build),
    "2": ("fig2_microscopic_operators", fig2_micro.build),
    "3": ("fig3_replay_operator", fig3_replay.build),
    "4": ("fig4_cross_family", fig4_cross_family.build),
    "5": ("fig5_observation_map_control", fig5_omap.build),
    "6": ("fig6_compressibility_transportability", fig6_compressibility.build),
}

README = """# Main figures - current redesign

Layout redesign of `the earlier figure layout` (same data, same panel content,
same colours). Rebuild: `py -m analysis.plot_main.build_all`

## What changed vs the earlier layout

- **Exact print geometry.** Every figure is built on a fixed 180 mm canvas and
  saved without `bbox_inches="tight"`. The earlier set cropped to the tight
  bbox and so shipped at 169-206 mm; fig6 exceeded the 180 mm NC maximum.
- **Density.** Panels are placed in millimetres from the top-left via
  `mm_axes`, not nested GridSpecs. The 15-25 % dead bands between panel rows
  are gone; heights are now 100-130 mm instead of 118-160 mm.
- **Alignment.** Panel letters are anchored in figure coordinates, so letters
  in a row share a baseline and panels in a column share a left edge
  regardless of tick-label width.
- **Optical balance.** Schematic panels were compressed and the area returned
  to the panels carrying the evidence.

## Design spec

- 180 mm (7.087 in) double-column width; heights 100-130 mm, ceiling 170 mm.
- Typography: Arial/Helvetica (Liberation Sans as a metric-compatible Linux
  substitute), 5.2-8 pt at final size; panel letters bold lowercase.
- Colours: Okabe-Ito colourblind-safe palette, unchanged from the earlier layout
  (moments #0072B2 - centers #E69F00 - intervals #009E73).
  Action trinomial p-/p0/p+ = #AA3377 / #E3E3E3 / #004488.
- Marks: 0.55 pt axes, direct labels over legend boxes, auto-contrast heatmap
  cell text, deterministic swarms instead of random jitter.
- PNG 450 dpi + editable-text PDF (fonttype 42).



## Order

1. `fig1_collective_phases` - observation maps select collective outcomes (GPT)
2. `fig2_microscopic_operators` - controlled fields elicit different operators
3. `fig3_replay_operator` - cross-encoding replay on identical endogenous fields
4. `fig4_cross_family` - model-family-specific collective outcomes (Claude R2
   macro + GPT-Claude matched map + GPT/Claude/Gemini microscopic replication)
5. `fig5_observation_map_control` - same-task-information control
6. `fig6_compressibility_transportability` - compressibility != transportability

Panel content follows the frozen panel plan (2026-07-25).
Figure 4 was rebuilt on the Claude macroscopic R2 acquisition
(`runs/stage_c/r2_claude_macro/.../r2_inference/`); its previous cross-family
content is now panel f.

Figure titles/captions live in the manuscript; panels only in these files.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    wanted = set(args.only) if args.only else set(BUILDERS)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": "main_redesign",
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
    (FIG_DIR / "README.md").write_text(README, encoding="utf-8")
    print(f"manifest: {FIG_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
