"""Emerging hotspot detection: junction areas whose event rate has suddenly risen.

Real road segments are short (tens of metres), so a single segment rarely carries
enough fleet traffic to show a change. Detection therefore works on *junction
areas*: all road within 75 m of a junction, pooled (a new hazard such as roadworks or
waterlogging affects all approaches). For each area, events in the recent window
are compared with what its own baseline rate (shrunk towards its road class)
predicts for the recent traffic, scaled by the network-wide trend so that city-wide
shifts (monsoon, festivals) do not trigger alerts. A one-sided Poisson test with
Benjamini-Hochberg control of the false-discovery rate decides which areas are
flagged; neighbouring flagged areas are grouped into one alert.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import poisson

from routesense.geo import junction_zones
from routesense.ingest.osm import ROAD_CLASSES, node_table

LEADING_TYPES = ("harsh_brake", "harsh_turn", "near_miss", "accident")


@dataclass
class EmergingParams:
    recent_days: int = 14
    baseline_days: int = 90
    min_events: int = 6
    min_ratio: float = 2.0
    fdr: float = 0.01
    prior_traversals: float = 200.0
    # Only monitor areas the fleet passes often enough to see a change: testing
    # rarely-driven areas adds multiple-testing burden without any chance of detection.
    min_recent_passes: int = 50


class DailyCounts:
    """Traversals and leading-indicator events per junction area per day (prefix sums)."""

    def __init__(self, segments: pd.DataFrame, trip_segments: pd.DataFrame, events: pd.DataFrame,
                 types=LEADING_TYPES, radius_m: float = 75.0):
        self.segments = segments
        self.events = events[events["type"].isin(types)]
        self.start = trip_segments["enter_ts"].min().normalize()
        end = max(trip_segments["enter_ts"].max(), events["ts"].max()).normalize()
        self.days = pd.date_range(self.start, end, freq="D")
        n_seg, n_days = len(segments), len(self.days)

        # Junction areas: every segment within ``radius_m`` of a node (or touching it).
        self.nodes, self.incidence = junction_zones(segments, radius_m)
        nodes = node_table(segments).loc[self.nodes]
        self.node_lat, self.node_lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
        self.node_segments = np.split(self.incidence.indices, self.incidence.indptr[1:-1])
        cls_rank = segments["road_class"].map({c: i for i, c in enumerate(ROAD_CLASSES)}).to_numpy()
        coo = self.incidence.tocoo()
        best = pd.Series(cls_rank[coo.col]).groupby(coo.row).min()
        self.area_class = np.array(ROAD_CLASSES)[best.reindex(np.arange(len(self.nodes))).fillna(len(ROAD_CLASSES) - 1).astype(int).to_numpy()]

        def daily(sid, ts):
            day = ((ts - self.start) // pd.Timedelta(days=1)).to_numpy()
            per_seg = np.bincount(sid * n_days + day, minlength=n_seg * n_days).reshape(n_seg, n_days)
            per_area = np.asarray(self.incidence @ per_seg.astype(np.float32))
            return np.concatenate([np.zeros((len(self.nodes), 1), np.float32), np.cumsum(per_area, axis=1)], axis=1)

        self._trav = daily(trip_segments["segment_id"].to_numpy(), trip_segments["enter_ts"])
        self._ev = daily(self.events["segment_id"].to_numpy(), self.events["ts"])

    def day_index(self, day) -> int:
        return int((pd.Timestamp(day).normalize() - self.start) // pd.Timedelta(days=1))

    def window(self, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
        """(traversals, events) per junction area over days first..last inclusive."""
        a = min(max(first, 0), self._trav.shape[1] - 1)
        b = min(max(last + 1, 0), self._trav.shape[1] - 1)
        return self._trav[:, b] - self._trav[:, a], self._ev[:, b] - self._ev[:, a]

    def area_index(self, node) -> int:
        return int(np.searchsorted(self.nodes, node))


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def detect(dc: DailyCounts, as_of, params: EmergingParams | None = None) -> pd.DataFrame:
    """Junction areas with a significant recent rise, as of the end of day ``as_of``."""
    p = params or EmergingParams()
    end = dc.day_index(as_of)
    t_r, o_r = dc.window(end - p.recent_days + 1, end)
    t_b, o_b = dc.window(end - p.recent_days - p.baseline_days + 1, end - p.recent_days)

    by_class = pd.DataFrame({"c": dc.area_class, "o": o_b, "t": t_b}).groupby("c")[["o", "t"]].sum()
    class_rate = (by_class["o"] / by_class["t"].clip(lower=1)).reindex(dc.area_class).to_numpy()
    rate_b = (o_b + p.prior_traversals * class_rate) / (t_b + p.prior_traversals)
    trend = (o_r.sum() / max(t_r.sum(), 1)) / max(o_b.sum() / max(t_b.sum(), 1), 1e-12)
    expected = rate_b * t_r * trend

    tested = np.flatnonzero((t_r >= max(p.min_recent_passes, 1)) & (expected > 0))
    pval = poisson.sf(o_r[tested] - 1, expected[tested])
    qval = _bh(pval)
    ratio = o_r[tested] / expected[tested]
    keep = (qval <= p.fdr) & (o_r[tested] >= p.min_events) & (ratio >= p.min_ratio)
    idx = tested[keep]
    return pd.DataFrame({
        "area": idx,
        "node": dc.nodes[idx],
        "segment_ids": [dc.node_segments[i].tolist() for i in idx],
        "observed": o_r[idx].astype(int),
        "expected": expected[idx],
        "ratio": ratio[keep],
        "q_value": qval[keep],
        "recent_traversals": t_r[idx].astype(int),
    }).sort_values("ratio", ascending=False, ignore_index=True)


ALERT_COLUMNS = ["alert_id", "segment_ids", "names", "lat", "lon", "n_junctions", "observed", "expected",
                 "ratio", "q_value", "top_event", "event_mix", "headline"]


def group_alerts(flagged: pd.DataFrame, dc: DailyCounts, as_of, recent_days: int = 14) -> pd.DataFrame:
    """Group neighbouring flagged junction areas into alerts, with what is driving them."""
    if flagged.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    sub = dc.incidence[flagged["area"].to_numpy()]
    _, label = connected_components(sub @ sub.T, directed=False)

    as_of = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
    recent = dc.events[(dc.events["ts"] >= as_of - pd.Timedelta(days=recent_days)) & (dc.events["ts"] < as_of)]
    out = []
    for alert_id, members in enumerate(pd.Series(np.arange(len(flagged))).groupby(label).groups.values()):
        f = flagged.iloc[list(members)]
        lead = f.loc[f["q_value"].idxmin()]  # strongest junction speaks for the alert
        sids = sorted({s for ids in f["segment_ids"] for s in ids})
        names = sorted({str(n) for n in dc.segments["name"].to_numpy()[sids] if pd.notna(n)})
        mix = recent[recent["segment_id"].isin(sids)]["type"].value_counts()
        top = mix.index[0] if len(mix) else None
        out.append({
            "alert_id": alert_id,
            "segment_ids": sids,
            "names": names,
            "lat": float(dc.node_lat[f["area"]].mean()),
            "lon": float(dc.node_lon[f["area"]].mean()),
            "n_junctions": int(len(f)),
            "observed": int(lead["observed"]),
            "expected": float(lead["expected"]),
            "ratio": float(lead["ratio"]),
            "q_value": float(lead["q_value"]),
            "top_event": top,
            "event_mix": mix.to_dict(),
            "headline": f"{(top or 'events').replace('_', ' ')} up {lead['ratio']:.1f}x in the last {recent_days} days"
                        + (f" near {' / '.join(names[:2])}" if names else ""),
        })
    return pd.DataFrame(out, columns=ALERT_COLUMNS).sort_values("ratio", ascending=False, ignore_index=True)


def replay(dc: DailyCounts, first_day, last_day, params: EmergingParams | None = None) -> pd.DataFrame:
    """Run ``detect`` for every day; return the first day each junction area was flagged."""
    first = {}
    for day in pd.date_range(first_day, last_day, freq="D"):
        for area in detect(dc, day, params)["area"]:
            first.setdefault(int(area), day)
    return pd.DataFrame({"area": list(first), "first_flagged": list(first.values())})


def score_detection(first_flags: pd.DataFrame, injected: pd.DataFrame, first_day, last_day,
                    dc: DailyCounts, recent_days: int = 14, busy_passes: int = 100) -> dict:
    """Detection delay per injected hazard, and false alarms away from any hazard.

    A hazard counts as detected when its junction area, or a neighbouring area sharing
    one of its segments, is flagged on or after onset. Each hazard's fleet traffic in
    the two weeks after onset is reported, since detection needs traffic to see.
    """
    flags = first_flags.set_index("area")["first_flagged"]
    hazard_segs = set(injected["segment_id"])
    near_hazard = {int(a) for a in np.unique(dc.incidence[:, sorted(hazard_segs)].nonzero()[0])}
    rows = []
    for hazard_id, h in injected.groupby("hazard_id"):
        start = pd.Timestamp(h["start_date"].iloc[0])
        if start > pd.Timestamp(last_day):
            continue
        areas = np.unique(dc.incidence[:, h["segment_id"].to_numpy()].nonzero()[0])
        hit = flags.reindex(areas).dropna()
        hit = hit[hit >= start]
        s = dc.day_index(start)
        passes = int(dc.window(s, s + recent_days - 1)[0][dc.area_index(h["node"].iloc[0])])
        rows.append({"hazard_id": int(hazard_id), "passes_2wk": passes, "detected": bool(len(hit)),
                     "delay_days": int((hit.min() - start).days) if len(hit) else None})
    per = pd.DataFrame(rows)
    false = flags[~flags.index.isin(near_hazard)]
    weeks = max((pd.Timestamp(last_day) - pd.Timestamp(first_day)).days / 7, 1)
    delays = per["delay_days"].dropna()
    busy = per[per["passes_2wk"] >= busy_passes]
    return {
        "hazards": int(len(per)),
        "detected": int(per["detected"].sum()),
        "median_delay_days": float(delays.median()) if len(delays) else None,
        "delays_days": [int(x) for x in delays],
        f"busy_hazards_{busy_passes}_passes_2wk": int(len(busy)),
        "busy_detected": int(busy["detected"].sum()),
        "per_hazard": rows,  # plain dicts: a DataFrame round trip would turn None delays into NaN
        "false_alarm_areas": int(len(false)),
        "false_alarms_per_week": round(len(false) / weeks, 2),
    }


def active_alerts(dc: DailyCounts, as_of, lookback_days: int = 30, params: EmergingParams | None = None) -> pd.DataFrame:
    """Alerts raised in the ``lookback_days`` up to ``as_of``, with when each first fired.

    Each junction area keeps its statistics from the most recent day it was flagged.
    """
    p = params or EmergingParams()
    as_of = pd.Timestamp(as_of).normalize()
    latest, first = {}, {}
    for day in pd.date_range(as_of - pd.Timedelta(days=lookback_days - 1), as_of, freq="D"):
        for row in detect(dc, day, p).to_dict("records"):
            latest[row["area"]] = row
            first.setdefault(row["area"], day)
    if not latest:
        return group_alerts(pd.DataFrame(), dc, as_of, p.recent_days).assign(first_flagged=pd.NaT)
    flagged = pd.DataFrame(list(latest.values()))
    alerts = group_alerts(flagged, dc, as_of, p.recent_days)
    area_of = {tuple(r["segment_ids"]): r["area"] for r in latest.values()}
    alerts["first_flagged"] = alerts["segment_ids"].map(
        lambda sids: min(first[a] for ids, a in area_of.items() if set(ids) <= set(sids)))
    return alerts
