"""Run per-representation --estimate-only for matched collective cost card."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = [
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_v0_1_moments.json",
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_v0_1_centers.json",
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_v0_1_intervals.json",
]
OUT = ROOT / "analysis" / "matched_rep_collective" / "estimate_only_log.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--input-price-per-million", type=float, default=0.75)
    parser.add_argument("--output-price-per-million", type=float, default=4.5)
    parser.add_argument(
        "--backend",
        choices=("openai", "mock"),
        default="openai",
        help="openai for real token estimate; mock still exercises the path",
    )
    args = parser.parse_args(argv)

    entries = []
    for proto_path in PROTOCOLS:
        cmd = [
            sys.executable,
            "-m",
            "experiments.stage_c.run_collective",
            "--protocol",
            str(proto_path),
            "--backend",
            args.backend,
            "--estimate-only",
            "--input-price-per-million",
            str(args.input_price_per_million),
            "--output-price-per-million",
            str(args.output_price_per_million),
        ]
        if args.backend == "openai":
            cmd.extend(["--model", args.model])
        print(">>", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(proc.returncode)
        proto = json.loads(proto_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "representation": proto["representation"],
                "protocol": str(proto_path.relative_to(ROOT)).replace("\\", "/"),
                "expected_calls": proto["expected_calls"],
                "stdout": proc.stdout,
                "model": args.model if args.backend == "openai" else "mock",
                "backend": args.backend,
            }
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_expected_calls": sum(e["expected_calls"] for e in entries),
        "entries": entries,
        "note": "Paste est. base USD into docs/MATCHED_REP_COLLECTIVE_COST_CARD.md",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
