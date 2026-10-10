"""Run the whole real-data pipeline for a region and write a summary to reports/<region>/.

    python scripts/run_region.py --region blr_core                  # download, simulate, build, evaluate
    python scripts/run_region.py --region blr_core --download-only  # just fetch roads + weather into snapshots/

Only the download step needs internet (overpass-api.de, archive-api.open-meteo.com).
Roads and weather are also copied to snapshots/<region>/ (tracked in git), so another
machine can run the remaining steps without internet.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import geopandas as gpd
import pandas as pd

from routesense.config import PROCESSED_DIR, PROJECT_DIR, RAW_DIR, REGIONS, Region
from routesense.evaluation import run, to_markdown
from routesense.ingest.osm import build_segments, load_graph, summarize_segments
from routesense.ingest.weather import fetch_hourly_weather
from routesense.pipeline import build
from routesense.reference import MORTH_2022
from routesense.sim.params import SimParams
from routesense.sim.simulator import calibration_report, simulate

SNAPSHOTS = PROJECT_DIR / "snapshots"
REPORTS = PROJECT_DIR / "reports"


def log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:6.0f}s] {msg}", flush=True)


def ensure_inputs(region, params: SimParams, t0: float) -> None:
    out = PROCESSED_DIR / region.name
    snap = SNAPSHOTS / region.name
    out.mkdir(parents=True, exist_ok=True)
    for name in ("segments.parquet", "weather.parquet"):
        if not (out / name).exists() and (snap / name).exists():
            shutil.copy(snap / name, out / name)
            log(f"using snapshot {snap / name}", t0)
    if not (out / "segments.parquet").exists():
        log(f"downloading OSM drive network for {region.bbox}", t0)
        seg = build_segments(load_graph(region, RAW_DIR))
        seg.to_parquet(out / "segments.parquet", index=False)
    if not (out / "weather.parquet").exists():
        log("downloading Open-Meteo hourly weather", t0)
        lat, lon = region.center
        fetch_hourly_weather(lat, lon, params.start_date, params.end_date, RAW_DIR).to_parquet(
            out / "weather.parquet", index=False)
    snap.mkdir(parents=True, exist_ok=True)
    for name in ("segments.parquet", "weather.parquet"):
        if not (snap / name).exists():
            shutil.copy(out / name, snap / name)


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join(rows)


def render_report(summary: dict, report: dict) -> str:
    roads, calib, morth = summary["roads"], summary["simulation"], summary["morth_2022_reference"]
    monthly = pd.DataFrame(summary["weather_by_month"]).set_index("month")
    slots = calib["serious_event_share_by_slot_pct"]
    ref = morth["accident_share_by_slot_pct"]
    model = summary["model"]
    return (
        f"# {summary['region']}: real roads and weather, simulated fleet\n\n"
        f"Roads (OpenStreetMap): {roads['segments']:,} directed segments, {roads['road_km']:,} km; "
        f"lighting tagged on {roads['lit_tag_coverage_pct']}% of segments. "
        f"{model['never_driven_segments_pct']}% of segments were never driven by the simulated fleet; "
        f"{model['data_sparse_segments_pct']}% have fewer than 30 fleet passes.\n\n"
        "Weather (Open-Meteo), by month:\n\n" + md_table(monthly) + "\n\n"
        "## Simulator check against MoRTH 2022 (all-India, all traffic)\n\n"
        "| Slot | Simulated fleet serious events | MoRTH accidents |\n|---|---|---|\n"
        + "".join(f"| {s} | {v}% | {str(ref[s]) + '%' if s in ref else ''} |\n" for s, v in slots.items())
        + "\nMoRTH shares mix traffic volume and risk for all vehicles; the fleet's hours differ (trucks drive at night).\n\n"
        + to_markdown(report).replace("# Backtest", "## Backtest", 1)
    )


def resolve_region(name: str) -> Region:
    """A configured region, or (for already-prepared data such as dev_city) one from its segments."""
    if name in REGIONS:
        return REGIONS[name]
    for d in (PROCESSED_DIR / name, SNAPSHOTS / name):
        if (d / "segments.parquet").exists():
            return Region(name, tuple(float(x) for x in gpd.read_parquet(d / "segments.parquet").total_bounds))
    raise SystemExit(f"unknown region {name!r}: add it to routesense/config.py REGIONS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="blr_core")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--impact-trips", type=int, default=1500)
    ap.add_argument("--report-only", action="store_true", help="re-render reports/<region>/report.md from its JSON")
    args = ap.parse_args()
    if args.report_only:
        rep = REPORTS / args.region
        summary = json.loads((rep / "summary.json").read_text())
        (rep / "report.md").write_text(render_report(summary, json.loads((rep / "report.json").read_text())))
        print((rep / "report.md").read_text())
        return
    region, params, t0 = resolve_region(args.region), SimParams(), time.perf_counter()
    out = PROCESSED_DIR / region.name

    ensure_inputs(region, params, t0)
    seg = gpd.read_parquet(out / "segments.parquet")
    weather = pd.read_parquet(out / "weather.parquet")
    roads = summarize_segments(seg)
    log(f"roads: {roads['segments']:,} segments, {roads['road_km']:,} km", t0)
    if args.download_only:
        print(f"Done. Commit and push snapshots/{region.name}/ to run the rest elsewhere.")
        return

    log("simulating fleet telematics", t0)
    sim = simulate(seg, weather, params)
    sim.save(out / "sim")
    calib = calibration_report(sim)
    log(f"{calib['trips']:,} trips, {calib['events']:,} events", t0)

    log("building model, hotspots, alerts, analytics", t0)
    meta = build(out, log=lambda m: None)

    log("evaluating (backtest + impact)", t0)
    report = run(seg, sim, weather, params, impact_trips=args.impact_trips)
    (out / "eval").mkdir(exist_ok=True)
    (out / "eval" / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "eval" / "report.md").write_text(to_markdown(report))

    rep = REPORTS / region.name
    rep.mkdir(parents=True, exist_ok=True)
    monthly = weather.set_index("ts")["precip_mm"].resample("MS").agg(["sum", lambda s: (s >= 0.5).mean() * 100])
    monthly.columns = ["precip_mm", "wet_hours_pct"]
    monthly.index = monthly.index.strftime("%Y-%m")
    summary = {"region": region.name, "bbox": region.bbox, "roads": roads,
               "weather_by_month": monthly.round(1).reset_index(names="month").to_dict("records"),
               "simulation": calib, "morth_2022_reference": MORTH_2022, "model": meta["model"],
               "n_hotspots": meta["n_hotspots"], "n_alerts": meta["n_alerts"]}
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (rep / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (rep / "report.md").write_text(render_report(summary, report))
    log(f"done: {rep / 'report.md'}", t0)
    print((rep / "report.md").read_text())


if __name__ == "__main__":
    main()
