"""Run the fleet simulator on a region prepared by feasibility_check.py.

    python scripts/simulate.py --region blr_core
"""
from __future__ import annotations

import argparse
import json
import time

import geopandas as gpd
import pandas as pd

from routesense.config import PROCESSED_DIR, REGIONS
from routesense.sim.params import SimParams
from routesense.sim.simulator import calibration_report, simulate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="blr_core", choices=sorted(REGIONS))
    ap.add_argument("--vehicles", type=int, default=None, help="override SimParams.n_vehicles")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    region = REGIONS[args.region]
    params = SimParams()
    if args.vehicles:
        params.n_vehicles = args.vehicles
    if args.seed is not None:
        params.seed = args.seed

    out = PROCESSED_DIR / region.name
    seg = gpd.read_parquet(out / "segments.parquet")
    weather = pd.read_parquet(out / "weather.parquet")

    t = time.perf_counter()
    res = simulate(seg, weather, params)
    print(f"Simulated in {time.perf_counter() - t:.0f}s")
    res.save(out / "sim")
    print(json.dumps(calibration_report(res), indent=2, default=str))
    print(f"Saved to {out / 'sim'}")


if __name__ == "__main__":
    main()
