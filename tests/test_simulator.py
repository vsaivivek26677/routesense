import numpy as np
import pandas as pd
import pytest
import shapely

from routesense.sim.params import SimParams, hourly
from routesense.sim.simulator import SimResult, calibration_report, simulate


def small_params(**overrides) -> SimParams:
    p = SimParams(
        start_date="2026-03-01", end_date="2026-04-30", n_vehicles=30, n_depots=2,
        n_destinations=30, n_emerging=3, emerging_window_days=30,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


@pytest.fixture(scope="module")
def result(segments, weather):
    return simulate(segments, weather, small_params())


def test_hourly_profile_wraps_midnight():
    prof = hourly(1.0, h22_2=3.0)
    assert prof[22] == prof[23] == prof[0] == prof[1] == 3.0
    assert prof[2] == 1.0 and len(prof) == 24


def test_deterministic(segments, weather, result):
    again = simulate(segments, weather, small_params())
    pd.testing.assert_frame_equal(result.events, again.events)


def test_events_lie_on_their_segments(segments, result):
    ev = result.events
    assert len(ev) > 0
    geoms = segments.geometry.to_numpy()[ev["segment_id"].to_numpy()]
    pts = shapely.points(ev["lon"], ev["lat"])
    assert (shapely.distance(geoms, pts) < 1e-6).all()


def test_events_happen_during_their_trip(result):
    ev = result.events.merge(result.trips[["trip_id", "depart_ts", "arrive_ts"]], on="trip_id")
    slack = pd.Timedelta(seconds=1)
    assert ((ev["ts"] >= ev["depart_ts"] - slack) & (ev["ts"] <= ev["arrive_ts"] + slack)).all()


def test_trip_routes_are_connected(segments, result):
    ts = result.trip_segments.merge(segments[["segment_id", "u", "v"]], on="segment_id")
    ts = ts.sort_values(["trip_id", "seq"])
    same_trip = ts["trip_id"].to_numpy()[1:] == ts["trip_id"].to_numpy()[:-1]
    joined = ts["v"].to_numpy()[:-1] == ts["u"].to_numpy()[1:]
    assert joined[same_trip].all()


def test_trip_counts_match_events(result):
    assert result.trips["n_events"].sum() == len(result.events)
    assert result.trips["n_accident"].sum() == (result.events["type"] == "accident").sum()


def test_emerging_hazards_raise_event_rate(segments, weather):
    p = small_params(n_vehicles=60, n_emerging=4, emerging_mult={"harsh_brake": 6.0})
    res = simulate(segments, weather, p)
    start = res.emerging.groupby("segment_id")["start_date"].min()
    ts = res.trip_segments[res.trip_segments["segment_id"].isin(start.index)]
    ev = res.events[(res.events["type"] == "harsh_brake") & res.events["segment_id"].isin(start.index)]
    after_trav = (ts["enter_ts"] >= ts["segment_id"].map(start)).sum()
    after_ev = (ev["ts"] >= ev["segment_id"].map(start)).sum()
    rate_after = after_ev / after_trav
    rate_before = (len(ev) - after_ev) / (len(ts) - after_trav)
    assert rate_after > 3 * rate_before


def test_rejects_short_weather(segments, weather):
    with pytest.raises(ValueError, match="weather"):
        simulate(segments, weather[weather["ts"] < "2026-04-01"], small_params())


def test_save_load_roundtrip(result, tmp_path):
    result.save(tmp_path)
    loaded = SimResult.load(tmp_path)
    for name in ("events", "trips", "trip_segments", "truth", "emerging"):
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(result, name))


def test_calibration_report(result):
    rep = calibration_report(result)
    assert rep["events"] == len(result.events)
    assert abs(sum(rep["serious_event_share_by_slot_pct"].values()) - 100) < 0.5
    assert np.isfinite(rep["accidents_per_million_km"])
