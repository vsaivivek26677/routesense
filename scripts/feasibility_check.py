"""Data feasibility check: can we get the real road network and weather we need?

Downloads (and caches) the OSM drive network and Open-Meteo hourly weather for a
region, builds the segment table, and prints headline numbers.

    python scripts/feasibility_check.py --region blr_core
"""
from __future__ import annotations

import argparse
import json

from routesense.config import PROCESSED_DIR, RAW_DIR, REGIONS
from routesense.ingest.osm import build_segments, load_graph, summarize_segments
from routesense.ingest.weather import fetch_hourly_weather
from routesense.sim.params import SimParams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="blr_core", choices=sorted(REGIONS))
    args = ap.parse_args()
    region = REGIONS[args.region]
    params = SimParams()

    print(f"[1/3] Road network for {region.name} {region.bbox} ...")
    G = load_graph(region, RAW_DIR)
    print(f"      {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("[2/3] Segment features ...")
    seg = build_segments(G)
    out = PROCESSED_DIR / region.name
    out.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(out / "segments.parquet", index=False)
    print(json.dumps(summarize_segments(seg), indent=2))

    print(f"[3/3] Hourly weather {params.start_date} .. {params.end_date} ...")
    lat, lon = region.center
    w = fetch_hourly_weather(lat, lon, params.start_date, params.end_date, RAW_DIR)
    w.to_parquet(out / "weather.parquet", index=False)
    monthly = w.set_index("ts")["precip_mm"].resample("MS").agg(["sum", lambda s: (s > 0.5).mean() * 100])
    monthly.columns = ["precip_mm", "rainy_hours_pct"]
    print(monthly.round(1).to_string())

    print(f"\nOK. Segments and weather saved to {out}")


if __name__ == "__main__":
    main()
