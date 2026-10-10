import numpy as np
import pandas as pd
import pytest

from routesense.geo import junction_zones
from routesense.models.emerging import (
    DailyCounts, EmergingParams, _bh, active_alerts, detect, group_alerts, replay, score_detection,
)


@pytest.fixture(scope="module")
def daily(city_segments, city_sim):
    _, sim = city_sim
    return DailyCounts(city_segments, sim.trip_segments, sim.events)


def test_bh_adjustment():
    q = _bh(np.array([0.01, 0.04, 0.03, 0.5]))
    # sorted p: 0.01, 0.03, 0.04, 0.5 -> raw 0.04, 0.06, 0.0533, 0.5 -> running min from the top
    np.testing.assert_allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
    assert (_bh(np.array([0.2, 0.2])) == 0.2).all()


def test_junction_zones_include_touching_and_nearby(city_segments):
    nodes, zones = junction_zones(city_segments, radius_m=75)
    assert zones.shape == (len(nodes), len(city_segments))
    assert set(np.unique(zones.data)) == {1.0}
    u = city_segments["u"].to_numpy()
    node = nodes[0]
    zone = set(zones[0].indices)
    assert set(np.flatnonzero(u == node)) <= zone
    bigger = junction_zones(city_segments, radius_m=300)[1]
    assert bigger.nnz > zones.nnz


def test_area_counts_are_zone_sums(city_sim, daily):
    _, sim = city_sim
    i = int(np.argmax(np.asarray(daily.incidence.sum(axis=1)).ravel()))
    zone = set(daily.node_segments[i])
    t, o = daily.window(0, len(daily.days) - 1)
    lead = sim.events[sim.events["type"].isin(["harsh_brake", "harsh_turn", "near_miss", "accident"])]
    assert t[i] == sim.trip_segments["segment_id"].isin(zone).sum()
    assert o[i] == lead["segment_id"].isin(zone).sum()


def test_no_alerts_before_any_hazard(city_sim, daily):
    _, sim = city_sim
    first_hazard = sim.emerging["start_date"].min()
    flagged = detect(daily, first_hazard - pd.Timedelta(days=1))
    assert len(flagged) <= 2  # FDR-controlled: at most a stray false alarm


def test_detects_most_injected_hazards(city_sim, daily):
    _, sim = city_sim
    first, last = sim.emerging["start_date"].min(), daily.days[-1]
    score = score_detection(replay(daily, first, last), sim.emerging, first, last, daily)
    assert score["detected"] >= score["hazards"] / 2
    assert score["median_delay_days"] <= 14
    assert score["false_alarms_per_week"] < 1
    assert len(score["per_hazard"]) == score["hazards"]
    import json
    json.dumps(score, allow_nan=False)  # valid JSON: undetected hazards have delay None, not NaN


def test_traffic_floor_limits_testing(city_sim, daily):
    _, sim = city_sim
    day = daily.days[-1]
    strict = detect(daily, day, EmergingParams(min_recent_passes=10**9))
    assert strict.empty


def test_alerts_group_neighbouring_areas(city_sim, daily):
    last = daily.days[-1]
    for day in pd.date_range(last - pd.Timedelta(days=30), last):
        flagged = detect(daily, day, EmergingParams())
        if len(flagged) >= 2:
            break
    alerts = group_alerts(flagged, daily, day)
    assert alerts["n_junctions"].sum() == len(flagged)
    assert set().union(*alerts["segment_ids"]) == {s for ids in flagged["segment_ids"] for s in ids}
    assert (alerts["ratio"] >= 1).all() and alerts["headline"].str.len().gt(0).all()


def test_active_alerts_cover_recent_hazards(city_sim, daily):
    _, sim = city_sim
    alerts = active_alerts(daily, daily.days[-1], lookback_days=30)
    assert "first_flagged" in alerts
    if len(alerts):
        assert (alerts["first_flagged"] >= daily.days[-1] - pd.Timedelta(days=29)).all()
        flagged = {s for ids in alerts["segment_ids"] for s in ids}
        assert flagged & set(sim.emerging["segment_id"])
