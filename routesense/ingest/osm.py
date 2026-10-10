"""Download the OSM drive network and turn its edges into a road-segment table."""
from __future__ import annotations

import math
import re
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString

from routesense.config import Region
from routesense.schema import validate

EXTRA_WAY_TAGS = ["lit", "lanes", "maxspeed", "surface", "bridge", "tunnel", "junction"]

ROAD_CLASSES = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service", "other",
)
LIT_YES = {"yes", "24/7", "automatic", "sunset-sunrise", "interval"}

# Typical off-peak (free-flow) speeds in Bengaluru by road class, km/h. ASSUMPTION:
# OSM maxspeed tags are sparse in India and give legal limits, not the speeds driven;
# OSMnx's default imputation would give side streets main-road speeds. The hourly
# congestion profile (learned from trip logs) is applied on top of these.
FREE_FLOW_KPH = {
    "motorway": 60, "trunk": 45, "primary": 35, "secondary": 30, "tertiary": 25,
    "unclassified": 20, "residential": 18, "living_street": 10, "service": 12, "other": 15,
}


def configure_osmnx() -> None:
    """Keep the extra way tags (lighting, lanes, speed limit ...) when downloading."""
    tags = list(ox.settings.useful_tags_way)
    ox.settings.useful_tags_way = tags + [t for t in EXTRA_WAY_TAGS if t not in tags]


def load_graph(region: Region, cache_dir: Path) -> nx.MultiDiGraph:
    """Drive network for ``region`` with speeds and travel times, cached as GraphML."""
    path = Path(cache_dir) / f"{region.name}.graphml"
    if path.exists():
        return ox.load_graphml(path)

    configure_osmnx()
    G = ox.graph_from_bbox(region.bbox, network_type="drive", simplify=True)
    set_urban_speeds(G)

    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)
    return G


def parse_maxspeed(value) -> float:
    """Lowest legal limit in km/h from OSM maxspeed tags ("40", "30 mph", "40;60", "IN:urban" ...)."""
    out = []
    for v in _values(value):
        text = str(v).lower()
        number = re.search(r"\d+(\.\d+)?", text)
        if number:
            kph = float(number.group())
            out.append(kph * 1.609 if "mph" in text else kph)
    return min(out) if out else np.nan


def set_urban_speeds(G: nx.MultiDiGraph) -> None:
    """Set ``speed_kph`` (class free-flow speed, capped by a lower tagged limit) and ``travel_time``."""
    for _, _, d in G.edges(data=True):
        speed = FREE_FLOW_KPH[road_class(d.get("highway"))]
        limit = parse_maxspeed(d.get("maxspeed"))
        if not np.isnan(limit) and limit > 0:
            speed = min(speed, limit)
        d["speed_kph"] = float(speed)
        d["travel_time"] = float(d["length"]) / (speed / 3.6)


def _first(value):
    return value[0] if isinstance(value, list) else value


def _values(value) -> list:
    """Tag values as a list without missing entries.

    OSMnx merges the tags of simplified road pieces into lists built from a set, so
    their order changes between runs: never rely on the first element.
    """
    items = value if isinstance(value, list) else [value]
    return [v for v in items if not _is_missing(v)]


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def road_class(highway) -> str:
    """Most important class among the (possibly merged) highway tags."""
    classes = [str(h).removesuffix("_link") for h in _values(highway)]
    classes = [c for c in classes if c in ROAD_CLASSES] or ["other"]
    return min(classes, key=ROAD_CLASSES.index)


def road_name(name) -> str | None:
    names = sorted({str(n) for n in _values(name)})
    return " / ".join(names) if names else None


def _lit_one(v) -> float:
    v = str(v).strip().lower()
    if v in LIT_YES:
        return 1.0
    if v == "no":
        return 0.0
    if v == "limited":
        return 0.5
    return np.nan


def parse_lit(value) -> float:
    """1.0 lit, 0.0 unlit, in between for partly lit pieces, NaN when OSM has no tag."""
    parsed = [x for x in map(_lit_one, _values(value)) if not np.isnan(x)]
    return float(np.mean(parsed)) if parsed else np.nan


def parse_lanes(value) -> float:
    """Fewest lanes among the tagged values ("2;3" counts as 2)."""
    out = []
    for v in _values(value):
        try:
            out.append(float(str(v).split(";")[0]))
        except ValueError:
            pass
    return min(out) if out else np.nan


def _is_yes(value) -> bool:
    return any(str(v).lower() not in ("no", "false", "0") for v in _values(value))


def curvature(line: LineString) -> tuple[float, float, float]:
    """(max_turn_deg, total_turn_deg, sinuosity) of a lon/lat polyline."""
    coords = np.asarray(line.coords)
    lat0 = math.radians(coords[:, 1].mean())
    xy = np.column_stack(
        (coords[:, 0] * 111_320 * math.cos(lat0), coords[:, 1] * 110_540)
    )
    steps = np.diff(xy, axis=0)
    steps = steps[np.hypot(steps[:, 0], steps[:, 1]) > 0.5]
    path_len = np.hypot(steps[:, 0], steps[:, 1]).sum() if len(steps) else 0.0
    chord = float(np.hypot(*(xy[-1] - xy[0])))
    sinuosity = path_len / chord if chord > 1.0 else 1.0
    if len(steps) < 2:
        return 0.0, 0.0, max(sinuosity, 1.0)
    heading = np.degrees(np.arctan2(steps[:, 1], steps[:, 0]))
    turn = np.abs((np.diff(heading) + 180) % 360 - 180)
    return float(turn.max()), float(turn.sum()), max(sinuosity, 1.0)


def build_segments(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """One row per directed edge, with the features the risk model and simulator use."""
    # Number of distinct roads meeting at a node (ignoring direction).
    arms = {n: len(set(G.predecessors(n)) | set(G.successors(n))) for n in G.nodes}

    rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry")
        if geom is None:
            geom = LineString(
                [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
            )
        max_turn, total_turn, sinuosity = curvature(geom)
        rows.append(
            {
                "u": u,
                "v": v,
                "key": k,
                "name": road_name(d.get("name")),
                "road_class": road_class(d.get("highway")),
                "length_m": float(d["length"]),
                "speed_kph": float(d.get("speed_kph", np.nan)),
                "travel_time_s": float(d.get("travel_time", np.nan)),
                "lanes": parse_lanes(d.get("lanes")),
                "lit": parse_lit(d.get("lit")),
                "junction_degree": max(arms[u], arms[v]),
                "max_turn_deg": max_turn,
                "total_turn_deg": total_turn,
                "sinuosity": sinuosity,
                "is_bridge": _is_yes(d.get("bridge")),
                "oneway": any(v is True or str(v).lower() in ("true", "yes", "1") for v in _values(d.get("oneway", False))),
                "geometry": geom,
            }
        )

    seg = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs="EPSG:4326")
    seg.insert(0, "segment_id", np.arange(len(seg), dtype=np.int64))
    return validate(seg, "segments")


def node_table(segments: pd.DataFrame) -> pd.DataFrame:
    """Node coordinates (from segment endpoints) and number of roads meeting there."""
    coords = segments.geometry
    start = pd.DataFrame({"node": segments["u"], "lon": coords.map(lambda g: g.coords[0][0]),
                          "lat": coords.map(lambda g: g.coords[0][1])})
    end = pd.DataFrame({"node": segments["v"], "lon": coords.map(lambda g: g.coords[-1][0]),
                        "lat": coords.map(lambda g: g.coords[-1][1])})
    nodes = pd.concat([start, end]).groupby("node")[["lat", "lon"]].first()
    pairs = pd.DataFrame({"a": np.minimum(segments["u"], segments["v"]),
                          "b": np.maximum(segments["u"], segments["v"])}).drop_duplicates()
    arms = pd.concat([pairs["a"], pairs["b"]]).value_counts()
    nodes["arms"] = arms.reindex(nodes.index, fill_value=0)
    return nodes


def summarize_segments(seg: pd.DataFrame) -> dict:
    """Headline numbers for the feasibility check and reports."""
    return {
        "segments": len(seg),
        "road_km": round(seg["length_m"].sum() / 1000, 1),
        "lit_tag_coverage_pct": round(100 * seg["lit"].notna().mean(), 1),
        "lit_share_where_tagged_pct": round(100 * seg["lit"].dropna().mean(), 1)
        if seg["lit"].notna().any()
        else None,
        "junctions_4plus_pct": round(100 * (seg["junction_degree"] >= 4).mean(), 1),
        "sharp_curve_pct": round(100 * (seg["max_turn_deg"] >= 45).mean(), 1),
        "km_by_class": (seg.groupby("road_class")["length_m"].sum() / 1000)
        .round(1)
        .sort_values(ascending=False)
        .to_dict(),
    }
