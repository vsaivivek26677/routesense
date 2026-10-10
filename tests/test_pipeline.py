import json

import pandas as pd
import pytest

from routesense.analytics import DIMENSIONS, area_of, summary, trends


def test_build_writes_artifacts(region_dir):
    a = region_dir / "artifacts"
    for name in ("model.joblib", "risk_table.npz", "hotspots.parquet", "zones.parquet",
                 "alerts.parquet", "analytics.parquet", "meta.json"):
        assert (a / name).exists(), name
    meta = json.loads((a / "meta.json").read_text())
    assert len(meta["congestion"]) == 24 and len(meta["level_thresholds"]) == 3
    assert any(p["kind"] == "depot" for p in meta["places"]) and any(p["kind"] == "customer" for p in meta["places"])
    alerts = pd.read_parquet(a / "alerts.parquet")
    assert all(isinstance(json.loads(m), dict) for m in alerts["event_mix"])


def test_area_of_corners():
    bbox = (0.0, 0.0, 3.0, 3.0)
    assert list(area_of([0.1, 2.9, 1.5], [0.1, 2.9, 1.5], bbox)) == ["South-West", "North-East", "Central"]


def test_trends_and_summary(region_dir):
    weekly = pd.read_parquet(region_dir / "artifacts" / "analytics.parquet")
    assert set(weekly["dimension"]) == set(DIMENSIONS)
    for dim in DIMENSIONS:
        t = trends(weekly, dim)
        assert t and all(len(series) > 4 for series in t.values())
    s = summary(weekly)
    assert s["vehicle_km"] > 0 and s["serious_per_100km_last_4_weeks"] > 0
    with pytest.raises(ValueError):
        trends(weekly, "colour")
