"""Build everything the app serves for one region, from its processed data.

Input  (data/processed/<region>/): segments.parquet, weather.parquet, sim/
Output (data/processed/<region>/artifacts/): model, risk table, hotspots, zones,
alerts, analytics and meta.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from routesense.analytics import area_of, depot_nodes, weekly_table
from routesense.ingest.osm import node_table
from routesense.models.emerging import DailyCounts, active_alerts
from routesense.models.hotspots import detect_hotspots, zones
from routesense.models.risk import RiskModel, exposure_table, save_table
from routesense.routing.engine import congestion_profile, fleet_level_thresholds
from routesense.sim.simulator import SimResult

SYNTHETIC_REGIONS = {"dev_city"}


def load_inputs(region_dir: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame, SimResult]:
    region_dir = Path(region_dir)
    return (
        gpd.read_parquet(region_dir / "segments.parquet"),
        pd.read_parquet(region_dir / "weather.parquet"),
        SimResult.load(region_dir / "sim"),
    )


def build(region_dir: Path, log=print) -> dict:
    region_dir = Path(region_dir)
    out = region_dir / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    seg, weather, sim = load_inputs(region_dir)
    as_of = sim.events["ts"].max()
    t0 = time.perf_counter()

    def step(msg):
        log(f"[{time.perf_counter() - t0:5.1f}s] {msg}")

    step("exposure table")
    ex = exposure_table(seg, sim.trips, sim.trip_segments, sim.events, weather)
    step("risk model")
    model = RiskModel.fit(seg, ex)
    model.save(out / "model.joblib")
    table = model.risk_table(seg)
    save_table(table, out / "risk_table.npz")

    step("hotspots")
    hot = detect_hotspots(sim.events, seg, sim.trip_segments, as_of=as_of)
    hot.to_parquet(out / "hotspots.parquet", index=False)
    zones(hot).to_parquet(out / "zones.parquet", index=False)

    step("emerging alerts")
    alerts = active_alerts(DailyCounts(seg, sim.trip_segments, sim.events), as_of)
    alerts.assign(event_mix=alerts["event_mix"].map(json.dumps)).to_parquet(out / "alerts.parquet", index=False)

    step("analytics")
    weekly_table(seg, sim.trips, sim.trip_segments, sim.events).to_parquet(out / "analytics.parquet", index=False)

    step("meta")
    nodes = node_table(seg)
    bbox = [float(x) for x in seg.total_bounds]
    depots = depot_nodes(sim.trips)
    dest_counts = sim.trips.loc[sim.trips["origin_node"].isin(depots.values), "dest_node"].value_counts().head(40)
    places = [
        {"id": f"depot-{int(d) + 1}", "label": f"Depot {int(d) + 1}", "kind": "depot",
         "lat": float(nodes.loc[n, "lat"]), "lon": float(nodes.loc[n, "lon"])}
        for d, n in depots.items()
    ] + [
        {"id": f"stop-{i + 1}", "label": f"Customer {i + 1} ({area_of([nodes.loc[n, 'lat']], [nodes.loc[n, 'lon']], bbox)[0]})",
         "kind": "customer", "lat": float(nodes.loc[n, "lat"]), "lon": float(nodes.loc[n, "lon"])}
        for i, n in enumerate(dest_counts.index)
    ]
    meta = {
        "region": region_dir.name,
        "synthetic_roads": region_dir.name in SYNTHETIC_REGIONS,
        "as_of": as_of.isoformat(),
        "data_start": sim.trips["depart_ts"].min().isoformat(),
        "bbox": bbox,
        "center": [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
        "counts": {"segments": len(seg), "trips": len(sim.trips), "events": len(sim.events),
                   "vehicle_km": round(float(ex["exposure_km"].sum()))},
        "congestion": [round(float(x), 4) for x in congestion_profile(sim.trips, sim.trip_segments, seg)],
        "level_thresholds": [float(x) for x in fleet_level_thresholds(table, ex)],
        "places": places,
        "model": {"alpha": {g: round(a, 3) for g, a in model.alpha.items()}, "features": model.features,
                  "data_sparse_segments_pct": round(100 * float(table.data_sparse.mean()), 1),
                  "never_driven_segments_pct": round(100 * float((model.exposure_km == 0).mean()), 1)},
        "n_hotspots": int(len(hot)),
        "n_alerts": int(len(alerts)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    step(f"done -> {out}")
    return meta
