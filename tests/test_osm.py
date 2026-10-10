import numpy as np
from shapely.geometry import LineString

from routesense.ingest.osm import build_segments, curvature, parse_lit, road_class, road_name, summarize_segments


def test_one_segment_per_directed_edge(graph, segments):
    assert len(segments) == graph.number_of_edges()
    assert segments["segment_id"].tolist() == list(range(len(segments)))
    assert segments.crs.to_epsg() == 4326


def test_interior_nodes_are_four_way_junctions(segments):
    # A grid edge between two interior nodes touches 4-arm junctions.
    assert segments["junction_degree"].max() >= 4


def test_curved_road_detected(segments):
    curved = segments[segments["road_class"] == "secondary"]
    assert len(curved) == 2
    assert (curved["max_turn_deg"] > 80).all()
    assert (curved["sinuosity"] > 1.3).all()
    straight = segments[segments["road_class"] == "residential"]
    assert (straight["max_turn_deg"] == 0).all()


def test_tag_parsing():
    assert parse_lit("yes") == 1.0 and parse_lit("no") == 0.0
    assert parse_lit(["limited", "yes"]) == 0.75 and parse_lit(["yes", "no"]) == 0.5
    assert np.isnan(parse_lit(None)) and np.isnan(parse_lit("unknown"))
    assert road_class(["primary_link", "secondary"]) == "primary"
    assert road_class("busway") == "other"
    # merged tags arrive in arbitrary order: results must not depend on it
    assert road_class(["service", "primary"]) == road_class(["primary", "service"]) == "primary"
    assert road_name(["B Road", "A Road"]) == road_name(["A Road", "B Road"]) == "A Road / B Road"
    assert road_name(float("nan")) is None


def test_curvature_of_right_angle():
    max_turn, total, sinuosity = curvature(LineString([(77.6, 12.9), (77.6, 12.91), (77.61, 12.91)]))
    assert 85 < max_turn < 95 and abs(total - max_turn) < 1e-9
    assert 1.3 < sinuosity < 1.5


def test_summary_counts(segments):
    s = summarize_segments(segments)
    assert s["segments"] == len(segments)
    assert 0 < s["lit_tag_coverage_pct"] < 100
    assert "primary" in s["km_by_class"]
