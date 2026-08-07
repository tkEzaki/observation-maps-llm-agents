"""Build every Supplementary figure that has a module, in order.

Modules are imported lazily so the set can be built while it is still being
written: a figure with no module yet is reported as pending rather than
crashing the run.

    py -m analysis.figures_si.build_all            # all
    py -m analysis.figures_si.build_all 12 13 20   # a subset
"""

from __future__ import annotations

import importlib
import json
import sys
import time
import traceback
from pathlib import Path

from analysis.figures_si.style import FIG_DIR

ORDER = [f"figS{n:02d}" for n in range(1, 24)]


def _build_one(mod_name: str):
    mod = importlib.import_module(f"analysis.figures_si.{mod_name}")
    return mod.build()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    wanted = ORDER
    if argv:
        keep = {f"figS{int(a):02d}" for a in argv}
        wanted = [m for m in ORDER if m in keep]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    built, pending, failed = [], [], []
    for mod_name in wanted:
        try:
            importlib.import_module(f"analysis.figures_si.{mod_name}")
        except ModuleNotFoundError:
            pending.append(mod_name)
            continue
        print(f"building {mod_name} ...", flush=True)
        t0 = time.time()
        try:
            path = _build_one(mod_name)
        except Exception:
            failed.append(mod_name)
            print(f"  !! {mod_name} FAILED\n{traceback.format_exc()}", flush=True)
            continue
        built.append(mod_name)
        print(f"  -> {path} ({time.time() - t0:.1f}s)", flush=True)

    pages = sorted(p.name for p in FIG_DIR.glob("figS*.pdf"))
    manifest = {
        "variant": "si",
        "plan": "frozen SI panel plan (2026-07-25)",
        "width_mm": 180.0,
        "png_dpi": 450,
        "pdf_fonttype": 42,
        "built": built,
        "pending": pending,
        "failed": failed,
        "pages": pages,
    }
    (FIG_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nbuilt {len(built)} | pending {len(pending)} | failed {len(failed)}")
    if pending:
        print("pending:", " ".join(pending))
    if failed:
        print("failed :", " ".join(failed))
    print("manifest:", FIG_DIR / "manifest.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
