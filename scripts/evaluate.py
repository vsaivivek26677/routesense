"""Backtest a region: train before --split, test after; writes eval/report.{json,md}.

    python scripts/evaluate.py --region dev_city
"""
from __future__ import annotations

import argparse
import json

from routesense.config import PROCESSED_DIR
from routesense.evaluation import run, to_markdown
from routesense.pipeline import load_inputs
from routesense.sim.params import SimParams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="dev_city")
    ap.add_argument("--split", default="2026-07-01")
    ap.add_argument("--impact-trips", type=int, default=1500)
    args = ap.parse_args()

    region_dir = PROCESSED_DIR / args.region
    seg, weather, sim = load_inputs(region_dir)
    report = run(seg, sim, weather, SimParams(), split=args.split, impact_trips=args.impact_trips)
    out = region_dir / "eval"
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
