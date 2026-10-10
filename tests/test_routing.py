from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from routesense.routing.engine import (
    Conditions, RouteEngine, congestion_profile, fleet_level_thresholds,
)


@pytest.fixture(scope="module")
def engine(city_segments, city_sim, city_model):
    _, sim = city_sim
    ex, _, table = city_model
    cong = congestion_profile(sim.trips, sim.trip_segments, city_segments)
    return RouteEngine(city_segments, table, cong, level_thresholds=fleet_level_thresholds(table, ex))


@pytest.fixture(scope="module")
def od(engine, city_segments):
    w, s, e, n = city_segments.total_bounds
    return engine.nearest_node(s + 0.1 * (n - s), w + 0.1 * (e - w)), engine.nearest_node(n - 0.1 * (n - s), e - 0.1 * (e - w))


COND = Conditions(pd.Timestamp("2026-08-20 18:15"), rain=True, vehicle_type="truck")


def test_congestion_profile_recovers_sim(city_segments, city_sim):
    params, sim = city_sim
    cong = congestion_profile(sim.trips, sim.trip_segments, city_segments)
    hours = sorted(sim.trips["depart_ts"].dt.hour.unique())
    np.testing.assert_allclose(cong[hours], params.congestion[hours], rtol=1e-6)


def test_level_thresholds_increase(engine):
    assert np.all(np.diff(engine.level_thresholds) > 0)


def test_shortest_path_is_connected(engine, city_segments, od):
    tt, _, _ = engine.segment_costs(COND)
    sids = engine.shortest(*od, tt)
    u, v = city_segments["u"].to_numpy(), city_segments["v"].to_numpy()
    assert (v[sids[:-1]] == u[sids[1:]]).all()
    assert u[sids[0]] == engine.node_ids[od[0]] and v[sids[-1]] == engine.node_ids[od[1]]


def test_options_are_pareto_and_recommend_one(engine, od):
    opts = engine.options(*od, COND)
    assert opts[0]["label"] == "Fastest"
    fastest = opts[0]["duration_s"]
    assert all(o["duration_s"] >= fastest - 1e-6 for o in opts)
    assert all(o["duration_s"] <= fastest * (1 + engine.policy.max_detour) + 1e-6 for o in opts)
    for a in opts:
        assert a["risk_lo"] <= a["risk"] <= a["risk_hi"]
        assert len(a["geometry"]) >= 2 and a["level"] in ("low", "moderate", "high", "very high")
        for b in opts:
            if a is not b:  # nobody is dominated
                assert not (b["duration_s"] < a["duration_s"] and b["planning_risk"] < a["planning_risk"])
    assert sum(o["recommended"] for o in opts) == 1
    if len(opts) > 1:
        safest = [o for o in opts if o["label"] == "Safest"]
        assert safest and safest[0]["risk"] == min(o["risk"] for o in opts)


def test_risk_aversion_plans_on_upper_bound(engine, od):
    opts = engine.options(*od, replace(COND, risk_aversion=2.0))
    assert all(o["planning_risk"] >= o["risk"] for o in opts)


def test_score_top_segments(engine, od):
    tt, _, _ = engine.segment_costs(COND)
    s = engine.score(engine.shortest(*od, tt), COND)
    shares = [t["share"] for t in s["top_segments"]]
    assert shares == sorted(shares, reverse=True) and sum(shares) <= 1 + 1e-9
    assert s["near_miss_per_1k_trips"] > 0 and 0 <= s["sparse_km_share"] <= 1


def test_departure_advice(engine, od):
    adv = engine.departure_advice(*od, COND)
    assert len(adv["slots"]) == 13
    base = next(s for s in adv["slots"] if s["offset_min"] == 0)
    assert adv["best"]["risk"] <= base["risk"]
    assert adv["advice"]
    deadline = COND.depart + pd.Timedelta(minutes=45)
    limited = engine.departure_advice(*od, COND, arrive_by=deadline)
    assert limited["best"] is base or limited["best"]["arrive"] <= deadline
