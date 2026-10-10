import pandas as pd
import pytest
from fastapi.testclient import TestClient

from routesense.api.app import create_app


@pytest.fixture(scope="module")
def client(region_dir):
    return TestClient(create_app(region_dir))


@pytest.fixture(scope="module")
def trip(client):
    meta = client.get("/api/meta").json()
    depot = next(p for p in meta["places"] if p["kind"] == "depot")
    stop = next(p for p in meta["places"] if p["kind"] == "customer")
    return {"origin": {"lat": depot["lat"], "lon": depot["lon"]}, "dest": {"lat": stop["lat"], "lon": stop["lon"]},
            "depart": "2026-08-20T18:15:00", "rain": True, "vehicle_type": "truck"}


def test_meta(client):
    m = client.get("/api/meta").json()
    assert m["vehicle_types"] == ["truck", "lcv", "two_wheeler"]
    assert m["eval_summary"]["trip_auc"] > 0.5
    assert len(m["congestion"]) == 24


def test_heatmap(client):
    h = client.get("/api/heatmap", params={"depart": "2026-08-20T19:00:00", "rain": True, "limit": 200,
                                           "min_level": "moderate"}).json()
    assert 0 < len(h["segments"]) <= 200
    risks = [x["r"] for x in h["segments"]]
    assert risks == sorted(risks, reverse=True) and min(risks) >= h["thresholds"][0]
    high = client.get("/api/heatmap", params={"depart": "2026-08-20T19:00:00", "rain": True}).json()
    assert all(x["level"] in ("high", "very high") for x in high["segments"])
    assert all(len(x["coords"]) >= 2 for x in h["segments"])


def test_route_options_and_briefing(client, trip):
    r = client.post("/api/route/options", json=trip)
    assert r.status_code == 200
    opts = r.json()["options"]
    assert opts[0]["label"] == "Fastest" and sum(o["recommended"] for o in opts) == 1
    assert "reasons" in opts[0]["top_segments"][0]
    b = client.post("/api/briefing", json={**{k: trip[k] for k in ("depart", "rain", "vehicle_type")},
                                           "segment_ids": opts[0]["segment_ids"], "language": "en"}).json()
    assert b["source"] in ("template", "llm") and b["text"]
    assert client.post("/api/briefing", json={**{k: trip[k] for k in ("depart", "rain", "vehicle_type")},
                                              "segment_ids": [10**9]}).status_code == 422


def test_departure_advice(client, trip):
    adv = client.post("/api/route/departure", json={**trip, "before_min": 30, "after_min": 60, "step_min": 30}).json()
    assert [s["offset_min"] for s in adv["slots"]] == [-30, 0, 30, 60]
    pd.Timestamp(adv["best"]["depart"])  # ISO timestamp
    assert adv["advice"]


def test_segment_detail(client):
    d = client.get("/api/segment/0", params={"depart": "2026-08-20T21:00:00"}).json()
    assert d["level"] in ("low", "moderate", "high", "very high") and isinstance(d["reasons"], list)
    assert client.get("/api/segment/999999", params={"depart": "2026-08-20T21:00:00"}).status_code == 404


def test_hotspots_search_and_alerts(client):
    allh = client.get("/api/hotspots", params={"limit": 5}).json()
    assert allh["total"] >= len(allh["items"]) and all(isinstance(i["segment_ids"], list) for i in allh["items"])
    name = next(i["name"] for i in allh["items"] if i["name"])
    found = client.get("/api/hotspots", params={"q": name.split()[0]}).json()
    assert found["total"] >= 1
    assert isinstance(client.get("/api/alerts").json(), list)
    assert isinstance(client.get("/api/zones").json(), list)


def test_dispatch_log_roundtrip(client, trip):
    opt = client.post("/api/route/options", json=trip).json()["options"][0]
    body = {**trip, "segment_ids": opt["segment_ids"], "option_label": opt["label"], "recommended_label": "Fastest",
            "followed_recommendation": True, "risk": opt["risk"], "risk_lo": opt["risk_lo"], "risk_hi": opt["risk_hi"],
            "level": opt["level"], "duration_s": opt["duration_s"], "distance_km": opt["distance_km"],
            "fastest_risk": opt["risk"], "fastest_duration_s": opt["duration_s"],
            "near_miss_per_1k_trips": opt["near_miss_per_1k_trips"], "accidents_per_1k_trips": opt["accidents_per_1k_trips"]}
    did = client.post("/api/dispatch", json=body).json()["id"]
    assert client.post(f"/api/dispatch/{did}/outcome", json={"near_misses": 1}).json() == {"ok": True}
    assert client.post("/api/dispatch/99999/outcome", json={}).status_code == 404
    log = client.get("/api/dispatch").json()
    assert log[0]["id"] == did and log[0]["outcome"]["near_misses"] == 1
    summ = client.get("/api/dispatch/summary").json()
    assert summ["dispatches"] >= 1 and summ["completed_reported_serious"] >= 1


def test_analytics(client):
    assert client.get("/api/analytics/summary").json()["vehicle_km"] > 0
    t = client.get("/api/analytics/trends", params={"dimension": "driver_cohort"}).json()
    assert set(t) == {"novice", "regular", "experienced"}
    assert client.get("/api/analytics/trends", params={"dimension": "nope"}).status_code == 422
    assert client.get("/api/eval").status_code == 200
