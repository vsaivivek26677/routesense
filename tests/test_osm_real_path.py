"""The real-data path: OSM XML -> OSMnx -> our speeds -> GraphML round trip -> segments -> app."""
import math

import numpy as np
import osmnx as ox
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from osm_xml import write_osm_xml

from routesense.devcity import dev_weather
from routesense.ingest.osm import FREE_FLOW_KPH, build_segments, configure_osmnx, parse_maxspeed, set_urban_speeds


def test_parse_maxspeed():
    assert parse_maxspeed("40") == 40
    assert parse_maxspeed("30 mph") == pytest.approx(48.27)
    assert parse_maxspeed(["50", "60"]) == 50
    assert parse_maxspeed("40;60") == 40
    assert math.isnan(parse_maxspeed("IN:urban")) and math.isnan(parse_maxspeed(None))


@pytest.fixture(scope="module")
def osm_graph(tmp_path_factory):
    d = tmp_path_factory.mktemp("osm")
    configure_osmnx()
    G = ox.graph_from_xml(write_osm_xml(d / "test.osm"), simplify=True)
    set_urban_speeds(G)
    ox.save_graphml(G, d / "test.graphml")
    return G, ox.load_graphml(d / "test.graphml")


def test_graphml_round_trip_keeps_features(osm_graph):
    fresh, reloaded = (build_segments(g) for g in osm_graph)
    cols = ["road_class", "length_m", "speed_kph", "travel_time_s", "lanes", "lit", "junction_degree", "max_turn_deg", "is_bridge"]
    pd.testing.assert_frame_equal(fresh[cols], reloaded[cols], check_dtype=False)


def test_speeds_follow_class_not_imputed_limits(osm_graph):
    seg = build_segments(osm_graph[1])
    by_class = seg.groupby("road_class")["speed_kph"].agg(["min", "max"])
    assert by_class.loc["residential"].tolist() == [FREE_FLOW_KPH["residential"]] * 2
    assert by_class.loc["primary", "max"] <= FREE_FLOW_KPH["primary"]
    assert np.allclose(seg["travel_time_s"], seg["length_m"] / (seg["speed_kph"] / 3.6))


def test_real_tags_parsed(osm_graph):
    seg = build_segments(osm_graph[1])
    assert seg["is_bridge"].any() and (seg["max_turn_deg"] > 45).any()
    assert set(seg["lit"].dropna().unique()) <= {0.0, 0.5, 1.0} and seg["lit"].isna().any()
    assert seg.loc[seg["lanes"].notna(), "lanes"].eq(2).all()  # "2;3" -> 2
    assert seg["u"].max() > 2**32  # real-size OSM node ids


def test_end_to_end_on_osm_format(osm_graph, tmp_path):
    from routesense.api.app import create_app
    from routesense.pipeline import build
    from routesense.sim.params import SimParams
    from routesense.sim.simulator import simulate

    seg = build_segments(osm_graph[1])
    weather = dev_weather("2026-03-01", "2026-04-30")
    params = SimParams(start_date="2026-03-01", end_date="2026-04-30", n_vehicles=20, n_depots=2,
                       n_destinations=30, n_emerging=2, emerging_window_days=20)
    d = tmp_path / "osm_city"
    d.mkdir()
    seg.to_parquet(d / "segments.parquet", index=False)
    weather.to_parquet(d / "weather.parquet", index=False)
    simulate(seg, weather, params).save(d / "sim")
    build(d, log=lambda *_: None)
    client = TestClient(create_app(d))
    places = client.get("/api/meta").json()["places"]
    a, b = places[0], places[-1]
    r = client.post("/api/route/options", json={"origin": {"lat": a["lat"], "lon": a["lon"]},
                                                 "dest": {"lat": b["lat"], "lon": b["lon"]},
                                                 "depart": "2026-04-20T18:00:00", "rain": True})
    assert r.status_code == 200 and r.json()["options"][0]["label"] == "Fastest"
