"""HTTP API for the dispatcher UI.

    uvicorn --factory routesense.api.app:app_from_env   (ROUTESENSE_REGION_DIR=data/processed/<region>)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from routesense.analytics import DIMENSIONS, summary, trends
from routesense.briefing import LANGUAGES, _llm_available, briefing
from routesense.explain import Explainer
from routesense.models.risk import RiskModel, band_of_hour, load_table
from routesense.routing.engine import Conditions, RouteEngine
from routesense.schema import VEHICLE_TYPES

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
VehicleType = Literal["truck", "lcv", "two_wheeler"]


# ---- request bodies ------------------------------------------------------------------
class LatLon(BaseModel):
    lat: float
    lon: float


class TripConditions(BaseModel):
    depart: datetime
    rain: bool = False
    vehicle_type: VehicleType = "lcv"
    risk_aversion: float = Field(0.0, ge=0, le=3)

    def conditions(self) -> Conditions:
        return Conditions(_local(self.depart), self.rain, self.vehicle_type, self.risk_aversion)


class RouteRequest(TripConditions):
    origin: LatLon
    dest: LatLon


class DepartureRequest(RouteRequest):
    before_min: int = Field(60, ge=0, le=240)
    after_min: int = Field(120, ge=0, le=360)
    step_min: int = Field(15, ge=5, le=60)
    arrive_by: datetime | None = None


class BriefingRequest(TripConditions):
    segment_ids: list[int]
    language: Literal["en", "hi", "kn"] = "en"


class DispatchRequest(RouteRequest):
    segment_ids: list[int]
    option_label: str
    recommended_label: str
    followed_recommendation: bool
    risk: float
    risk_lo: float
    risk_hi: float
    level: str
    duration_s: float
    distance_km: float
    fastest_risk: float
    fastest_duration_s: float
    near_miss_per_1k_trips: float = 0.0
    accidents_per_1k_trips: float = 0.0
    driver: str | None = None
    vehicle: str | None = None


class OutcomeRequest(BaseModel):
    harsh_events: int = Field(0, ge=0)
    near_misses: int = Field(0, ge=0)
    accidents: int = Field(0, ge=0)


def _local(ts: datetime) -> pd.Timestamp:
    """Naive Asia/Kolkata timestamp (the convention used throughout the project)."""
    t = pd.Timestamp(ts)
    return t.tz_convert("Asia/Kolkata").tz_localize(None) if t.tzinfo else t


def _ints(values) -> list[int]:
    return [int(v) for v in values]


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


# ---- services ----------------------------------------------------------------------
class Services:
    def __init__(self, region_dir: Path):
        a = Path(region_dir) / "artifacts"
        if not (a / "meta.json").exists():
            raise FileNotFoundError(f"no artifacts in {a}; run scripts/build.py first")
        self.region_dir = Path(region_dir)
        self.meta = json.loads((a / "meta.json").read_text())
        self.segments = gpd.read_parquet(Path(region_dir) / "segments.parquet")
        self.model = RiskModel.load(a / "model.joblib")
        self.table = load_table(a / "risk_table.npz")
        self.hotspots = pd.read_parquet(a / "hotspots.parquet")
        self.zones = pd.read_parquet(a / "zones.parquet")
        self.alerts = pd.read_parquet(a / "alerts.parquet")
        self.alerts["event_mix"] = self.alerts["event_mix"].map(json.loads)
        self.alerts["segment_ids"] = self.alerts["segment_ids"].map(_ints)
        self.alerts["names"] = self.alerts["names"].map(list)
        self.hotspots["segment_ids"] = self.hotspots["segment_ids"].map(_ints)
        self.weekly = pd.read_parquet(a / "analytics.parquet")
        self.engine = RouteEngine(self.segments, self.table, np.array(self.meta["congestion"]),
                                  level_thresholds=np.array(self.meta["level_thresholds"]))
        self.explainer = Explainer(self.segments, self.model, self.table, self.hotspots, self.alerts)
        report = Path(region_dir) / "eval" / "report.json"
        self.eval = json.loads(report.read_text()) if report.exists() else None
        self.link = np.minimum(self.segments["u"], self.segments["v"]).astype(str) + "-" + \
            np.maximum(self.segments["u"], self.segments["v"]).astype(str)
        self.db = _DispatchLog(Path(region_dir) / "dispatch.sqlite")

    def snap(self, p: LatLon) -> int:
        return self.engine.nearest_node(p.lat, p.lon)

    def with_reasons(self, option: dict, cond: Conditions) -> dict:
        band = int(band_of_hour(cond.depart.hour))
        for t in option["top_segments"]:
            t["reasons"] = self.explainer.reasons(t["segment_id"], band, int(cond.rain), cond.vehicle, limit=3)
        return option


class _DispatchLog:
    SCHEMA = """CREATE TABLE IF NOT EXISTS dispatches (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, payload TEXT,
        outcome TEXT, outcome_at TEXT)"""

    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute(self.SCHEMA)
            self.conn.commit()

    def add(self, payload: dict) -> int:
        with self.lock:
            cur = self.conn.execute("INSERT INTO dispatches (created_at, payload) VALUES (?, ?)",
                                    (datetime.now().isoformat(timespec="seconds"), json.dumps(payload, default=str)))
            self.conn.commit()
            return int(cur.lastrowid)

    def set_outcome(self, did: int, outcome: dict) -> bool:
        with self.lock:
            cur = self.conn.execute("UPDATE dispatches SET outcome = ?, outcome_at = ? WHERE id = ?",
                                    (json.dumps(outcome), datetime.now().isoformat(timespec="seconds"), did))
            self.conn.commit()
            return cur.rowcount == 1

    def list(self, limit: int = 100) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, created_at, payload, outcome, outcome_at FROM dispatches ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"id": r[0], "created_at": r[1], **json.loads(r[2]),
                 "outcome": json.loads(r[3]) if r[3] else None, "outcome_at": r[4]} for r in rows]


# ---- app ---------------------------------------------------------------------------
def create_app(region_dir: Path) -> FastAPI:
    s = Services(region_dir)
    app = FastAPI(title="RouteSense", version="0.1.0")

    @app.get("/api/meta")
    def meta():
        ev = s.eval
        return {
            **s.meta,
            "vehicle_types": list(VEHICLE_TYPES),
            "languages": LANGUAGES,
            "llm_briefings": _llm_available(),
            "eval_summary": None if ev is None else {
                "split": ev["split"],
                "trip_auc": ev["trip_backtest"]["auc_full_model"],
                "trip_auc_distance": ev["trip_backtest"]["auc_distance_only"],
                "top10_lift": ev["segment_backtest"]["full_model"]["top10pct_km_lift"],
                "impact": ev["impact"],
                "emerging": ev["emerging_detection"],
            },
        }

    @lru_cache(maxsize=64)
    def _heatmap(hour: int, rain: bool, vehicle_type: str, limit: int, min_level: str) -> dict:
        cond = Conditions(pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour), rain, vehicle_type)
        band = int(band_of_hour(hour))
        per_km = s.table.mean[:, band, int(rain), cond.vehicle]
        th = s.engine.level_thresholds
        df = pd.DataFrame({"sid": np.arange(len(per_km)), "r": per_km, "link": s.link})
        floor = th[["moderate", "high", "very high"].index(min_level)]
        df = df[df["r"] >= floor].sort_values("r", ascending=False).drop_duplicates("link").head(limit)
        return {
            "thresholds": [float(x) for x in th],
            "segments": [
                {"id": int(i), "r": round(float(r), 4), "level": s.engine.level(float(r)),
                 "sparse": bool(s.table.data_sparse[i]), "name": s.engine.names[i],
                 "coords": [[round(y, 5), round(x, 5)] for x, y in s.engine.coords[i]]}
                for i, r in zip(df["sid"], df["r"])
            ],
        }

    @app.get("/api/heatmap")
    def heatmap(depart: datetime, rain: bool = False, vehicle_type: VehicleType = "lcv",
                limit: int = Query(4000, ge=1, le=20000),
                min_level: Literal["moderate", "high", "very high"] = "high"):
        return _heatmap(_local(depart).hour, rain, vehicle_type, limit, min_level)

    @app.get("/api/segment/{sid}")
    def segment(sid: int, depart: datetime, rain: bool = False, vehicle_type: VehicleType = "lcv"):
        if not 0 <= sid < len(s.segments):
            raise HTTPException(404, "unknown segment")
        cond = Conditions(_local(depart), rain, vehicle_type)
        band = int(band_of_hour(cond.depart.hour))
        r = float(s.table.mean[sid, band, int(rain), cond.vehicle])
        row = s.segments.iloc[sid]
        return {
            "id": sid, "name": row["name"] if pd.notna(row["name"]) else None, "road_class": row["road_class"],
            "risk_per_km": r, "level": s.engine.level(r), "data_sparse": bool(s.table.data_sparse[sid]),
            "observed": {g: int(s.model.observed[g][sid]) for g in s.model.observed},
            "vehicle_km": round(float(s.model.exposure_km[sid]), 1),
            "reasons": s.explainer.reasons(sid, band, int(rain), cond.vehicle, limit=5),
        }

    @app.get("/api/hotspots")
    def hotspots(q: str | None = None, kind: Literal["intersection", "road"] | None = None,
                 limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        h = s.hotspots
        if kind:
            h = h[h["kind"] == kind]
        if q:
            h = h[h["name"].fillna("").str.contains(q, case=False, regex=False) | (h["hotspot_id"].astype(str) == q)]
        return {"total": int(len(h)), "items": _records(h.iloc[offset:offset + limit])}

    @app.get("/api/zones")
    def zones():
        return _records(s.zones.assign(names=s.zones["names"].map(list)))

    @app.get("/api/alerts")
    def alerts():
        return _records(s.alerts)

    @app.post("/api/route/options")
    def route_options(req: RouteRequest):
        cond = req.conditions()
        try:
            opts = s.engine.options(s.snap(req.origin), s.snap(req.dest), cond)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"options": [s.with_reasons(o, cond) for o in opts]}

    @app.post("/api/route/departure")
    def route_departure(req: DepartureRequest):
        adv = s.engine.departure_advice(
            s.snap(req.origin), s.snap(req.dest), req.conditions(), req.before_min, req.after_min, req.step_min,
            _local(req.arrive_by) if req.arrive_by else None)
        return json.loads(json.dumps(adv, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)))

    @app.post("/api/briefing")
    def make_briefing(req: BriefingRequest):
        if not req.segment_ids or max(req.segment_ids) >= len(s.segments) or min(req.segment_ids) < 0:
            raise HTTPException(422, "invalid segment_ids")
        cond = req.conditions()
        option = s.engine.score(np.array(req.segment_ids), cond)
        return briefing(option, cond, s.explainer, req.language)

    @app.post("/api/dispatch")
    def dispatch(req: DispatchRequest):
        payload = req.model_dump(mode="json")
        payload["risk_change_vs_fastest_pct"] = round(100 * (req.risk / req.fastest_risk - 1), 1) if req.fastest_risk else 0.0
        return {"id": s.db.add(payload)}

    @app.get("/api/dispatch")
    def dispatch_log(limit: int = Query(100, ge=1, le=1000)):
        return s.db.list(limit)

    @app.post("/api/dispatch/{did}/outcome")
    def dispatch_outcome(did: int, req: OutcomeRequest):
        if not s.db.set_outcome(did, req.model_dump()):
            raise HTTPException(404, "unknown dispatch")
        return {"ok": True}

    @app.get("/api/dispatch/summary")
    def dispatch_summary():
        rows = s.db.list(1000)
        if not rows:
            return {"dispatches": 0}
        done = [r for r in rows if r["outcome"]]
        return {
            "dispatches": len(rows),
            "followed_recommendation_pct": round(100 * np.mean([r["followed_recommendation"] for r in rows]), 1),
            "avg_risk_change_vs_fastest_pct": round(float(np.mean([r["risk_change_vs_fastest_pct"] for r in rows])), 1),
            "avg_extra_min": round(float(np.mean([(r["duration_s"] - r["fastest_duration_s"]) / 60 for r in rows])), 1),
            "completed": len(done),
            "completed_predicted_serious": round(sum(
                (r["near_miss_per_1k_trips"] + r["accidents_per_1k_trips"]) / 1000 for r in done), 2),
            "completed_reported_serious": int(sum(r["outcome"]["near_misses"] + r["outcome"]["accidents"] for r in done)),
            "completed_reported": {k: int(sum(r["outcome"][k] for r in done)) for k in ("harsh_events", "near_misses", "accidents")},
        }

    @app.get("/api/analytics/summary")
    def analytics_summary():
        return summary(s.weekly)

    @app.get("/api/analytics/trends")
    def analytics_trends(dimension: str = "road_class", metric: Literal["serious", "harsh", "near_miss", "accident"] = "serious"):
        if dimension not in DIMENSIONS:
            raise HTTPException(422, f"dimension must be one of {DIMENSIONS}")
        return trends(s.weekly, dimension, metric)

    @app.get("/api/eval")
    def evaluation():
        if s.eval is None:
            raise HTTPException(404, "no evaluation report; run scripts/evaluate.py")
        return s.eval

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def app_from_env() -> FastAPI:
    return create_app(Path(os.environ.get("ROUTESENSE_REGION_DIR", "data/processed/dev_city")))
