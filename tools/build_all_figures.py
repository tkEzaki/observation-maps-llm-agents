#!/usr/bin/env python3
"""Rebuild every manuscript and supplementary figure.

Each figure module exposes ``build()``, which reads the artefacts under
``analysis/`` and writes a PDF and a PNG into ``figures/``.  Run
``tools/restore_frozen_results.py`` first, or the inputs will be missing.

    python tools/build_all_figures.py             # everything
    python tools/build_all_figures.py fig3 figS16 # selected modules
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN = ["fig1_macro", "fig2_micro", "fig3_replay", "fig4_cross_family",
        "fig5_omap", "fig6_compressibility"]
SI = [f"figS{n:02d}" for n in range(1, 24)] + ["make_surrogate_schematic"]

# These modules re-derive per-call statistics from the large microscopic sweeps
# rather than from a stored summary.  Their raw responses are tracked in this
# repository under runs/, so nothing extra has to be downloaded; the check
# below stays in place for a shallow or partial checkout.
ARCHIVE_DEPENDENT = {
    "figS02": "stage_c, stimulus_manifold and matched_rep_collective_r1 traces",
    "figS03": "response_law_representation traces",
    "figS04": "response_law traces",
    "figS06": "antipodal_weight traces",
}


def archive_present() -> bool:
    """True when at least one large-sweep trace has been unpacked."""
    for name in ("stage_c", "stimulus_manifold", "response_law",
                 "response_law_representation", "antipodal_weight"):
        if any((ROOT / "runs" / name).rglob("trace.jsonl")):
            return True
    return False


def modules() -> list[str]:
    return ([f"analysis.figures_main.{m}" for m in MAIN]
            + [f"analysis.figures_si.{m}" for m in SI])


def main(argv: list[str]) -> int:
    wanted = argv[1:]
    have_archive = archive_present()
    ok = failed = skipped = 0
    for name in modules():
        short = name.rsplit(".", 1)[1]
        if wanted and not any(w in short for w in wanted):
            continue
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError:
            print(f"skip  {short}  (module not present)")
            skipped += 1
            continue
        build = getattr(mod, "build", None)
        if build is None:
            print(f"skip  {short}  (no build())")
            skipped += 1
            continue
        try:
            out = build()
            print(f"ok    {short}  -> {Path(out).name}")
            ok += 1
        except Exception as exc:                      # noqa: BLE001
            if short in ARCHIVE_DEPENDENT and not have_archive:
                print(f"skip  {short}  needs {ARCHIVE_DEPENDENT[short]} "
                      f"(see DATA_ARCHIVE_MANIFEST.json)")
                skipped += 1
                continue
            print(f"FAIL  {short}  {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            failed += 1
    print(f"\n{ok} built, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
