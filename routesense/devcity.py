"""A synthetic city for development and tests while real OSM/weather data is unavailable.

Produces a road graph in OSMnx's format (so ``build_segments`` treats it exactly
like downloaded data) and synthetic hourly weather with a monsoon season.
Nothing produced here should be presented as real-world data or results.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from routesense.config import Region

CENTER = (12.97, 77.62)  # (lat, lon)
SPEED_KPH = {"trunk": 60, "primary": 50, "secondary": 40, "tertiary": 35, "residential": 25}
LIT_TAG_PROBS = {  # P(tag = "yes"), P(tag = "no"); remainder untagged
    "trunk": (0.8, 0.0), "primary": (0.7, 0.05), "secondary": (0.4, 0.1),
    "tertiary": (0.15, 0.15), "residential": (0.05, 0.15),
}


def dev_region(n: int = 60, spacing_m: float = 150.0) -> Region:
    half = n * spacing_m / 2
    dlat = half / 110_540
    dlon = half / (111_320 * math.cos(math.radians(CENTER[0])))
    return Region("dev_city", (CENTER[1] - dlon, CENTER[0] - dlat, CENTER[1] + dlon, CENTER[0] + dlat))


def _line_class(k: int) -> str:
    if k % 10 == 0:
        return "primary"
    if k % 5 == 0:
        return "secondary"
    if k % 3 == 0:
        return "tertiary"
    return "residential"


def dev_city(n: int = 60, spacing_m: float = 150.0, seed: int = 0) -> nx.MultiDiGraph:
    """Jittered n x n street grid with arterials, a diagonal trunk road and bends."""
    rng = np.random.default_rng(seed)
    lat0, lon0 = CENTER
    m_per_dlon = 111_320 * math.cos(math.radians(lat0))
    G = nx.MultiDiGraph(crs="epsg:4326")

    def node_id(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            x = (j - n / 2) * spacing_m + rng.normal(0, 15)
            y = (i - n / 2) * spacing_m + rng.normal(0, 15)
            G.add_node(node_id(i, j), x=lon0 + x / m_per_dlon, y=lat0 + y / 110_540)

    def add(a, b, highway, name, bend=False):
        ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
        bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
        coords = [(ax, ay), (bx, by)]
        if bend:  # push the midpoint sideways to make a curve
            mx, my = (ax + bx) / 2, (ay + by) / 2
            dx, dy = bx - ax, by - ay
            off = rng.uniform(0.25, 0.45) * rng.choice([-1, 1])
            coords = [(ax, ay), (mx - dy * off, my + dx * off), (bx, by)]
        line = LineString(coords)
        length = sum(
            math.hypot((x2 - x1) * m_per_dlon, (y2 - y1) * 110_540)
            for (x1, y1), (x2, y2) in zip(line.coords[:-1], line.coords[1:])
        )
        yes, no = LIT_TAG_PROBS[highway]
        r = rng.random()
        lit = "yes" if r < yes else "no" if r < yes + no else None
        speed = SPEED_KPH[highway]
        for s, t, geom in ((a, b, line), (b, a, LineString(line.coords[::-1]))):
            attrs = dict(highway=highway, name=name, length=length, speed_kph=float(speed),
                         travel_time=length / (speed / 3.6), oneway=False, geometry=geom)
            if lit:
                attrs["lit"] = lit
            G.add_edge(s, t, **attrs)

    for i in range(n):
        for j in range(n):
            for (ni, nj, k, label) in ((i, j + 1, i, "Main Road"), (i + 1, j, j, "Cross Road")):
                if ni >= n or nj >= n:
                    continue
                cls = _line_class(k)
                if cls == "residential" and rng.random() < 0.12:
                    continue  # irregular local streets
                add(node_id(i, j), node_id(ni, nj), cls, f"{k} {label}",
                    bend=cls != "primary" and rng.random() < 0.08)
    for i in range(n - 1):  # diagonal arterial creates 5-6 arm junctions
        add(node_id(i, i), node_id(i + 1, i + 1), "trunk", "Diagonal Ring Road")
    return G


def dev_weather(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly weather: dry until June, afternoon/evening showers in the monsoon."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    monsoon = ts.month.isin([6, 7, 8, 9])
    evening = (ts.hour >= 14) & (ts.hour <= 22)
    p_rain = np.where(monsoon, np.where(evening, 0.22, 0.08), 0.02)
    precip = np.where(rng.random(len(ts)) < p_rain, rng.exponential(3.0, len(ts)), 0.0).round(1)
    return pd.DataFrame({
        "ts": ts, "precip_mm": precip, "rain_mm": precip,
        "weather_code": np.where(precip > 0, 61, 1), "temperature_c": 26.0, "cloud_cover": 50,
    })
