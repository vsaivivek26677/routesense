"""Risk trend reporting: event rates per 100 km by week and by dimension.

Dimensions: road class and city area (from where the driving happened), and vehicle
type, driver cohort, route and departure time slot (from the trip).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from routesense.ingest.osm import node_table
from routesense.models.risk import BAND_NAMES, band_of_hour

AREAS = (("South-West", "South", "South-East"), ("West", "Central", "East"), ("North-West", "North", "North-East"))
DIMENSIONS = ("road_class", "area", "vehicle_type", "driver_cohort", "route", "time_slot")
EVENT_COLS = {"harsh": ("harsh_brake", "harsh_turn", "harsh_accel"), "near_miss": ("near_miss",), "accident": ("accident",)}


def area_of(lat: np.ndarray, lon: np.ndarray, bbox) -> np.ndarray:
    """3x3 compass grid over the region's bounding box."""
    w, s, e, n = bbox
    row = np.clip(((np.asarray(lat) - s) / (n - s) * 3).astype(int), 0, 2)
    col = np.clip(((np.asarray(lon) - w) / (e - w) * 3).astype(int), 0, 2)
    return np.array([AREAS[r][c] for r, c in zip(row, col)], dtype=object)


def segment_areas(segments: pd.DataFrame) -> np.ndarray:
    mid = segments.geometry.map(lambda g: g.coords[len(g.coords) // 2])
    return area_of(mid.map(lambda c: c[1]).to_numpy(), mid.map(lambda c: c[0]).to_numpy(), segments.total_bounds)


def depot_nodes(trips: pd.DataFrame) -> pd.Series:
    """Depot id -> node: the most common trip origin for that depot (every delivery starts there)."""
    return trips.groupby("depot_id")["origin_node"].agg(lambda s: s.mode().iloc[0])


def _week(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("W-SUN").dt.start_time


def _event_counts(events: pd.DataFrame, keys: list[pd.Series]) -> pd.DataFrame:
    out = {}
    for col, types in EVENT_COLS.items():
        sel = events["type"].isin(types).to_numpy()
        out[col] = pd.Series(1, index=events.index)[sel].groupby([k[sel] for k in keys]).sum()
    return pd.DataFrame(out).fillna(0)


def weekly_table(segments: pd.DataFrame, trips: pd.DataFrame, trip_segments: pd.DataFrame,
                 events: pd.DataFrame) -> pd.DataFrame:
    """Long table: dimension, group, week, km, harsh, near_miss, accident."""
    seg_area = segment_areas(segments)
    nodes = node_table(segments)
    parts = []

    # Segment-based dimensions: where the km and events happened.
    sid = trip_segments["segment_id"].to_numpy()
    km = segments["length_m"].to_numpy()[sid] / 1000
    t_week = _week(trip_segments["enter_ts"])
    e_week = _week(events["ts"])
    e_sid = events["segment_id"].to_numpy()
    for dim, values in (("road_class", segments["road_class"].to_numpy()), ("area", seg_area)):
        expo = pd.Series(km).groupby([t_week.to_numpy(), values[sid]]).sum().rename("km")
        ev = _event_counts(events, [pd.Series(e_week.to_numpy(), index=events.index),
                                    pd.Series(values[e_sid], index=events.index)])
        parts.append(pd.concat([expo, ev], axis=1).fillna(0).rename_axis(["week", "group"]).reset_index().assign(dimension=dim))

    # Trip-based dimensions.
    t = trips.copy()
    t["week"] = _week(t["depart_ts"])
    is_out = t["origin_node"] == t["depot_id"].map(depot_nodes(t))
    area_node = np.where(is_out, t["dest_node"], t["origin_node"])
    t["route"] = ("Depot " + (t["depot_id"] + 1).astype(str) + " ↔ "
                  + area_of(nodes.loc[area_node, "lat"].to_numpy(), nodes.loc[area_node, "lon"].to_numpy(), segments.total_bounds))
    t["time_slot"] = np.array(BAND_NAMES)[band_of_hour(t["depart_ts"].dt.hour.to_numpy())]
    ev_trip = events[["trip_id", "type"]].merge(t[["trip_id", "week", *DIMENSIONS[2:]]], on="trip_id")
    for dim in DIMENSIONS[2:]:
        expo = (t["distance_m"] / 1000).groupby([t["week"], t[dim]]).sum().rename("km")
        ev = _event_counts(ev_trip, [ev_trip["week"], ev_trip[dim]])
        parts.append(pd.concat([expo, ev], axis=1).fillna(0).rename_axis(["week", "group"]).reset_index().assign(dimension=dim))

    out = pd.concat(parts, ignore_index=True)
    out["group"] = out["group"].astype(str)
    return out[["dimension", "group", "week", "km", "harsh", "near_miss", "accident"]]


def trends(weekly: pd.DataFrame, dimension: str, metric: str = "serious", min_km: float = 500.0) -> dict:
    """{group: [{week, value, km, events}]} with value = events per 100 km (serious = near-miss + accident).

    Weeks with less than ``min_km`` of driving in a group get no rate (a single event on
    a few km would read as a huge spike); their km and events still count in totals.
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    d = weekly[weekly["dimension"] == dimension].copy()
    events = d["near_miss"] + d["accident"] if metric == "serious" else d[metric]
    d["events"] = events
    d["value"] = 100 * events / d["km"].where(d["km"] >= min_km)
    # Drop partial first/last weeks with little driving.
    weeks = d.groupby("week")["km"].sum()
    full = weeks[weeks >= 0.5 * weeks.median()].index
    d = d[d["week"].isin(full)]
    return {
        g: [{"week": w.strftime("%Y-%m-%d"), "value": round(float(v), 4) if pd.notna(v) else None,
             "km": round(float(k)), "events": int(n)}
            for w, v, k, n in zip(x["week"], x["value"], x["km"], x["events"])]
        for g, x in d.sort_values("week").groupby("group")
    }


def summary(weekly: pd.DataFrame) -> dict:
    d = weekly[weekly["dimension"] == "vehicle_type"]
    by_week = d.groupby("week")[["km", "harsh", "near_miss", "accident"]].sum().sort_index()
    by_week = by_week[by_week["km"] >= 0.5 * by_week["km"].median()]
    last4, prev4 = by_week.iloc[-4:].sum(), by_week.iloc[-8:-4].sum()
    rate = lambda x: 100 * (x["near_miss"] + x["accident"]) / x["km"] if x["km"] else None  # noqa: E731
    return {
        "vehicle_km": round(float(d["km"].sum())),
        "harsh_events": int(d["harsh"].sum()),
        "near_misses": int(d["near_miss"].sum()),
        "accidents": int(d["accident"].sum()),
        "serious_per_100km_last_4_weeks": round(float(rate(last4)), 3) if rate(last4) is not None else None,
        "serious_per_100km_prev_4_weeks": round(float(rate(prev4)), 3) if len(by_week) >= 8 else None,
    }
