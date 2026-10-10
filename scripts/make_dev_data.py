"""Build the SYNTHETIC dev city dataset (for development while real data is unavailable).

Writes the same files as feasibility_check.py + simulate.py, under data/processed/dev_city.

    python scripts/make_dev_data.py
"""
from __future__ import annotations

import json
import time

from routesense.config import PROCESSED_DIR
from routesense.devcity import dev_city, dev_weather
from routesense.ingest.osm import build_segments, summarize_segments
from routesense.sim.params import SimParams
from routesense.sim.simulator import calibration_report, simulate


def main() -> None:
    params = SimParams()
    out = PROCESSED_DIR / "dev_city"
    out.mkdir(parents=True, exist_ok=True)

    seg = build_segments(dev_city())
    seg.to_parquet(out / "segments.parquet", index=False)
    weather = dev_weather(params.start_date, params.end_date)
    weather.to_parquet(out / "weather.parquet", index=False)
    print(json.dumps(summarize_segments(seg), indent=2))

    t = time.perf_counter()
    res = simulate(seg, weather, params)
    res.save(out / "sim")
    print(f"Simulated in {time.perf_counter() - t:.0f}s")
    print(json.dumps(calibration_report(res), indent=2, default=str))


if __name__ == "__main__":
    main()
