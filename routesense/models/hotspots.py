"""Hotspot detection: where do severe, recent events concentrate beyond what traffic explains?

1. Weight every event by type, severity and recency (exponential decay).
2. Attribute events near a junction (3+ roads) to that intersection, the rest to the
   road link they occurred on (both travel directions merged).
3. Normalise by exposure (recency-weighted traversals) and compare with the typical
   rate for similar places (intersections by size, links by road class), shrinking
   low-traffic places towards that typical rate.
4. Elements well above typical are hotspots; nearby hotspots are clustered (DBSCAN)
   into zones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from routesense.ingest.osm import node_table

TYPE_WEIGHT = {"harsh_brake": 1.0, "harsh_turn": 1.0, "harsh_accel": 0.5, "near_miss": 5.0, "accident": 20.0}


@dataclass
class HotspotParams:
    half_life_days: float = 60.0
    junction_radius_m: float = 30.0
    prior_traversals: float = 100.0  # shrinkage strength towards the typical rate
    min_excess: float = 1.5  # hotspot if shrunk rate >= 1.5x typical
    min_weight: float = 10.0
    min_events: int = 5
    zone_eps_m: float = 200.0


def _local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float) -> np.ndarray:
    return np.column_stack((lon * 111_320 * np.cos(np.radians(lat0)), lat * 110_540))


def event_weights(events: pd.DataFrame, as_of: pd.Timestamp, half_life_days: float) -> np.ndarray:
    type_w = events["type"].map(TYPE_WEIGHT).to_numpy(dtype=float)
    return type_w * events["severity"].to_numpy() / 3 * recency(events["ts"], as_of, half_life_days)


def recency(ts: pd.Series, as_of: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age_days = (as_of - ts).dt.total_seconds().to_numpy() / 86_400
    return 0.5 ** (np.clip(age_days, 0, None) / half_life_days)


def detect_hotspots(
    events: pd.DataFrame,
    segments: pd.DataFrame,
    trip_segments: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    params: HotspotParams | None = None,
) -> pd.DataFrame:
    """One row per hotspot (intersection or road link), most excess risk first."""
    p = params or HotspotParams()
    as_of = pd.Timestamp(as_of) if as_of is not None else events["ts"].max()
    events = events[events["ts"] <= as_of]
    trip_segments = trip_segments[trip_segments["enter_ts"] <= as_of]
    nodes = node_table(segments)
    lat0 = float(nodes["lat"].mean())

    # Exposure: recency-weighted traversals per directed segment.
    t_seg = np.bincount(
        trip_segments["segment_id"].to_numpy(),
        weights=recency(trip_segments["enter_ts"], as_of, p.half_life_days),
        minlength=len(segments),
    )

    # Attribute each event to the nearest end node if it is a junction within radius.
    sid = events["segment_id"].to_numpy()
    ev_xy = _local_xy(events["lat"].to_numpy(), events["lon"].to_numpy(), lat0)
    u, v = segments["u"].to_numpy()[sid], segments["v"].to_numpy()[sid]
    d = {}
    for name, node in (("u", u), ("v", v)):
        n_xy = _local_xy(nodes.loc[node, "lat"].to_numpy(), nodes.loc[node, "lon"].to_numpy(), lat0)
        d[name] = np.hypot(*(ev_xy - n_xy).T)
    near = np.where(d["u"] <= d["v"], u, v)
    near_d = np.minimum(d["u"], d["v"])
    at_junction = (near_d <= p.junction_radius_m) & (nodes.loc[near, "arms"].to_numpy() >= 3)

    link_a = np.minimum(segments["u"].to_numpy(), segments["v"].to_numpy())
    link_b = np.maximum(segments["u"].to_numpy(), segments["v"].to_numpy())
    ev = pd.DataFrame({
        "element": np.where(at_junction, "J" + near.astype(str),
                            "L" + link_a[sid].astype(str) + "-" + link_b[sid].astype(str)),
        "w": event_weights(events, as_of, p.half_life_days),
        "type": events["type"].to_numpy(),
        "ts": events["ts"].to_numpy(),
        "x": ev_xy[:, 0], "y": ev_xy[:, 1],
        "segment_id": sid,
    })

    # Elements: exposure, typical-rate group, names, member segments.
    seg_el = pd.DataFrame({
        "segment_id": segments["segment_id"].to_numpy(),
        "link": "L" + link_a.astype(str) + "-" + link_b.astype(str),
        "junction": "J" + segments["v"].astype(str).to_numpy(),
        "t": t_seg,
        "road_class": segments["road_class"].to_numpy(),
        "name": segments["name"].to_numpy(),
    })
    links = seg_el.groupby("link").agg(
        exposure=("t", "sum"), group=("road_class", "first"), name=("name", "first"),
        segment_ids=("segment_id", list),
    )
    # Junction exposure = traversals entering it; members = all segments touching it.
    jn = seg_el.groupby("junction").agg(exposure=("t", "sum"))
    touching = pd.concat([
        pd.DataFrame({"junction": "J" + segments["u"].astype(str), "segment_id": segments["segment_id"], "name": segments["name"]}),
        pd.DataFrame({"junction": "J" + segments["v"].astype(str), "segment_id": segments["segment_id"], "name": segments["name"]}),
    ])
    jn = jn.join(touching.groupby("junction").agg(
        segment_ids=("segment_id", lambda s: sorted(set(s))),
        name=("name", lambda s: " × ".join(sorted({str(x) for x in s if pd.notna(x)})[:3])),
    ))
    jn_arms = nodes["arms"].reindex(jn.index.str[1:].astype(np.int64)).to_numpy()
    jn["group"] = np.where(jn_arms >= 5, "junction_5+", np.where(jn_arms == 4, "junction_4", "junction_3"))
    elements = pd.concat([links, jn])

    agg = ev.groupby("element").agg(
        weighted_events=("w", "sum"), n_events=("w", "size"),
        x=("x", "mean"), y=("y", "mean"), last_event_ts=("ts", "max"),
    )
    counts = ev.pivot_table(index="element", columns="type", values="w", aggfunc="size", fill_value=0)
    el = elements.join(agg, how="left").fillna({"weighted_events": 0.0, "n_events": 0})
    el["n_harsh"] = counts.reindex(columns=["harsh_brake", "harsh_turn", "harsh_accel"], fill_value=0).sum(axis=1).reindex(el.index, fill_value=0)
    for t in ("near_miss", "accident"):
        el[f"n_{t}"] = counts[t].reindex(el.index, fill_value=0) if t in counts else 0

    # Typical rate per group and shrunk element rate.
    grp = el.groupby("group")[["weighted_events", "exposure"]].sum()
    base = (grp["weighted_events"] / grp["exposure"].clip(lower=1e-9)).rename("base_rate")
    el = el.join(base, on="group")
    el["rate"] = (el["weighted_events"] + p.prior_traversals * el["base_rate"]) / (el["exposure"] + p.prior_traversals)
    el["excess_ratio"] = el["rate"] / el["base_rate"]
    el["excess_events"] = el["weighted_events"] - el["base_rate"] * el["exposure"]

    hot = el[
        (el["excess_ratio"] >= p.min_excess)
        & (el["weighted_events"] >= p.min_weight)
        & (el["n_events"] >= p.min_events)
    ].copy()
    if hot.empty:
        return pd.DataFrame(columns=HOTSPOT_COLUMNS)

    # Zones: clusters of nearby hotspots.
    labels = DBSCAN(eps=p.zone_eps_m, min_samples=2).fit_predict(hot[["x", "y"]].to_numpy())
    hot["zone_id"] = labels

    # Recent trend: last 30 days vs the 30 before (raw counts).
    recent = ev[ev["ts"] > as_of - pd.Timedelta(days=30)].groupby("element").size()
    prev = ev[(ev["ts"] <= as_of - pd.Timedelta(days=30)) & (ev["ts"] > as_of - pd.Timedelta(days=60))].groupby("element").size()
    hot["trend_30d"] = (recent.reindex(hot.index, fill_value=0) + 1) / (prev.reindex(hot.index, fill_value=0) + 1)

    hot["kind"] = np.where(hot.index.str.startswith("J"), "intersection", "road")
    hot["node"] = np.where(hot["kind"] == "intersection", hot.index.str[1:], None)
    hot["lon"] = hot["x"] / (111_320 * np.cos(np.radians(lat0)))
    hot["lat"] = hot["y"] / 110_540
    hot["rate_per_1k"] = hot["rate"] * 1000
    hot = hot.sort_values("excess_events", ascending=False).reset_index(names="element")
    hot.insert(0, "hotspot_id", np.arange(len(hot)))
    return hot[HOTSPOT_COLUMNS]


HOTSPOT_COLUMNS = [
    "hotspot_id", "element", "kind", "node", "name", "lat", "lon", "zone_id", "segment_ids",
    "n_events", "n_harsh", "n_near_miss", "n_accident", "weighted_events", "exposure",
    "rate_per_1k", "base_rate", "excess_ratio", "excess_events", "trend_30d", "last_event_ts",
]


def zones(hotspots: pd.DataFrame) -> pd.DataFrame:
    """Summary per zone (clusters of 2+ nearby hotspots)."""
    z = hotspots[hotspots["zone_id"] >= 0]
    return z.groupby("zone_id").agg(
        n_hotspots=("hotspot_id", "size"), lat=("lat", "mean"), lon=("lon", "mean"),
        weighted_events=("weighted_events", "sum"), excess_events=("excess_events", "sum"),
        n_near_miss=("n_near_miss", "sum"), n_accident=("n_accident", "sum"),
        names=("name", lambda s: sorted({str(x) for x in s if pd.notna(x)})[:4]),
    ).sort_values("excess_events", ascending=False).reset_index()
