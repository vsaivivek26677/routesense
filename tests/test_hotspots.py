import numpy as np

from routesense.ingest.osm import node_table
from routesense.models.hotspots import HOTSPOT_COLUMNS, detect_hotspots, zones


def test_node_arms(city_segments):
    nodes = node_table(city_segments)
    assert nodes["arms"].max() >= 4
    assert nodes["lat"].between(12.8, 13.1).all()


def test_hotspots_shape_and_order(city_segments, city_sim):
    _, sim = city_sim
    hot = detect_hotspots(sim.events, city_segments, sim.trip_segments)
    assert list(hot.columns) == HOTSPOT_COLUMNS
    assert len(hot) > 5
    assert hot["excess_events"].is_monotonic_decreasing
    assert (hot["excess_ratio"] >= 1.5).all()
    assert set(hot["kind"]) <= {"intersection", "road"}
    z = zones(hot)
    assert (z["n_hotspots"] >= 2).all()


def test_hotspots_enriched_in_black_spots(city_segments, city_sim):
    _, sim = city_sim
    hot = detect_hotspots(sim.events, city_segments, sim.trip_segments)
    black = set(np.flatnonzero(sim.truth["black_spot"].to_numpy()))
    road = hot[hot["kind"] == "road"]
    share = road["segment_ids"].map(lambda s: bool(set(s) & black)).mean()
    base = sim.truth["black_spot"].mean() * 2  # a link has two directed segments
    assert share > 3 * base


def test_as_of_excludes_future(city_segments, city_sim):
    _, sim = city_sim
    as_of = sim.events["ts"].quantile(0.5)
    hot = detect_hotspots(sim.events, city_segments, sim.trip_segments, as_of=as_of)
    assert (hot["last_event_ts"] <= as_of).all()
