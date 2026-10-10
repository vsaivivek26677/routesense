"""Fleet telematics simulator on a real road network.

Vehicles drive real shortest-ish routes between depots and destinations. On each
road segment traversed, events (harsh brake/turn/accel, near-miss, accident) are
drawn from a Poisson process whose rate depends on road geometry, lighting, time
of day, weather, vehicle type, driver cohort, a hidden per-segment hazard and
hazards that appear mid-period. Because the true rates are known, every model
downstream can be scored against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from routesense.geo import junction_zones
from routesense.schema import DRIVER_COHORTS, EVENT_TYPES, VEHICLE_TYPES, validate
from routesense.sim.params import SimParams

NO_HAZARD = np.iinfo(np.int64).max


@dataclass
class SimResult:
    events: pd.DataFrame
    trips: pd.DataFrame
    trip_segments: pd.DataFrame
    truth: pd.DataFrame  # per-segment ground truth (hidden from models)
    emerging: pd.DataFrame  # injected emerging hazards

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("events", "trips", "trip_segments", "truth", "emerging"):
            getattr(self, name).to_parquet(out_dir / f"{name}.parquet", index=False)

    @classmethod
    def load(cls, out_dir: Path) -> "SimResult":
        out_dir = Path(out_dir)
        return cls(**{
            name: pd.read_parquet(out_dir / f"{name}.parquet")
            for name in ("events", "trips", "trip_segments", "truth", "emerging")
        })


def _to_seconds(ts) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000_000)


def _seconds_to_ts(seconds) -> pd.Series | pd.DatetimeIndex:
    """Epoch seconds to timestamps at microsecond precision (what parquet stores)."""
    return pd.to_datetime(np.asarray(seconds, dtype=np.int64), unit="s").astype("datetime64[us]")


def _type_mult(table: dict, keys: tuple, event_type: str) -> np.ndarray:
    """Multiplier per category (in ``keys`` order) for one event type."""
    return np.array([table[k].get(event_type, 1.0) for k in keys])


class _Router:
    """Shortest paths from each depot, out and back, under several weightings."""

    def __init__(self, seg: pd.DataFrame, params: SimParams, rng: np.random.Generator):
        best = seg.sort_values("travel_time_s").drop_duplicates(["u", "v"])
        lo, hi = params.route_perturbation
        perturb = rng.uniform(lo, hi, size=(params.route_variants, len(best)))
        perturb[0] = 1.0

        dg = nx.DiGraph()
        for i, (u, v, sid, tt) in enumerate(
            zip(best["u"], best["v"], best["segment_id"], best["travel_time_s"])
        ):
            weights = {f"w{j}": tt * perturb[j, i] for j in range(params.route_variants)}
            dg.add_edge(u, v, sid=int(sid), **weights)

        core = max(nx.strongly_connected_components(dg), key=len)
        self.graph = dg.subgraph(core).copy()
        self.nodes = np.array(sorted(core))
        self.depots = rng.choice(self.nodes, params.n_depots, replace=False)
        n_dest = min(params.n_destinations, len(self.nodes))
        self.destinations = rng.choice(self.nodes, n_dest, replace=False)
        self._pred: dict = {}
        self._paths: dict = {}

    def path(self, depot_i: int, dest, variant: int, outbound: bool) -> np.ndarray:
        """Segment ids from depot to dest (outbound) or dest to depot (return)."""
        key = (depot_i, variant, outbound)
        cached = self._paths.get((key, dest))
        if cached is not None:
            return cached
        if key not in self._pred:
            g = self.graph if outbound else self.graph.reverse(copy=False)
            self._pred[key], _ = nx.dijkstra_predecessor_and_distance(
                g, self.depots[depot_i], weight=f"w{variant}"
            )
        pred = self._pred[key]
        chain = [dest]
        while chain[-1] != self.depots[depot_i]:
            chain.append(pred[chain[-1]][0])
        nodes = chain[::-1] if outbound else chain
        sids = np.array(
            [self.graph[a][b]["sid"] for a, b in zip(nodes[:-1], nodes[1:])], dtype=np.int64
        )
        self._paths[(key, dest)] = sids
        return sids


def _segment_truth(seg: pd.DataFrame, p: SimParams, rng: np.random.Generator):
    n = len(seg)
    hidden = rng.lognormal(-p.hidden_sigma**2 / 2, p.hidden_sigma, n)
    black = rng.random(n) < p.black_spot_frac
    lit = seg["lit"].to_numpy(dtype=float)
    p_lit = seg["road_class"].map(p.latent_lit_prob).fillna(0.5).to_numpy()
    latent_lit = np.where(np.isnan(lit), rng.random(n) < p_lit, lit >= 0.5)
    cls = seg["road_class"].map(p.road_class_mult).fillna(1.0).to_numpy()
    junction = seg["junction_degree"].to_numpy() >= 4
    curve = seg["max_turn_deg"].to_numpy() >= p.sharp_turn_deg

    static = {}
    for t in EVENT_TYPES:
        m = cls * hidden
        m = m * np.where(junction, p.junction_mult.get(t, 1.0), 1.0)
        m = m * np.where(curve, p.curve_mult.get(t, 1.0), 1.0)
        m = m * np.where(black, p.black_spot_mult.get(t, 1.0), 1.0)
        static[t] = m
    return hidden, black, latent_lit, static


class TruthArrays:
    """Per-segment ground-truth arrays used to compute true event rates."""

    def __init__(self, truth: pd.DataFrame, segments: pd.DataFrame, params: SimParams):
        self.p = params
        self.km = segments["length_m"].to_numpy(dtype=float) / 1000
        self.static = {t: truth[f"true_mult_{t}"].to_numpy() for t in EVENT_TYPES}
        self.latent_lit = truth["latent_lit"].to_numpy(dtype=bool)
        start = truth["emerging_start_day"].to_numpy()
        self.emerging_start = np.where(start < 0, NO_HAZARD, start)
        self.veh_m = {t: _type_mult(params.vehicle_mult, VEHICLE_TYPES, t) for t in EVENT_TYPES}
        self.drv_m = {t: _type_mult(params.driver_mult, DRIVER_COHORTS, t) for t in EVENT_TYPES}

    def rate(self, event_type, sids, hour, rain_frac, vtype_idx, cohort_idx, day_idx=None) -> np.ndarray:
        """Expected events of ``event_type`` per traversal of each segment in ``sids``.

        ``rain_frac`` is min(precip_mm, 10) / 10. ``day_idx`` (days since the simulation
        start) switches on emerging hazards; omit it to ignore them.
        """
        p, t = self.p, event_type
        night_from, night_to = p.night_hours
        night = (hour >= night_from) | (hour < night_to)
        lam = (
            p.base_rate_per_km[t] * self.km[sids] * self.static[t][sids]
            * p.tod_mult[t][hour]
            * (1 + p.rain_coef.get(t, 0.0) * rain_frac)
            * np.where(night & ~self.latent_lit[sids], p.unlit_night_mult.get(t, 1.0), 1.0)
            * self.veh_m[t][vtype_idx] * self.drv_m[t][cohort_idx]
        )
        if day_idx is not None:
            lam = lam * np.where(day_idx >= self.emerging_start[sids], p.emerging_mult.get(t, 1.0), 1.0)
        return lam


def _emerging_hazards(seg, router, n_days, p: SimParams, rng):
    """Pick junctions where a new hazard switches on; return per-segment start day.

    Hazards are placed at junctions on the fleet's usual (fastest) routes: hazards on
    roads the fleet never drives cannot be seen in, or matter to, its telematics. Each
    hazard covers the junction zone (all road within ``emerging_radius_m``), like
    roadworks or waterlogging would.
    """
    start_day = np.full(len(seg), NO_HAZARD, dtype=np.int64)
    rows = []
    if p.n_emerging == 0:
        return start_day, pd.DataFrame(columns=["hazard_id", "node", "segment_id", "start_day"])
    degree = dict(router.graph.to_undirected(as_view=True).degree())
    on_routes = pd.Series(np.concatenate([
        seg["v"].to_numpy()[router.path(d, dest, 0, True)]
        for d in range(len(router.depots)) for dest in router.destinations
        if dest != router.depots[d]
    ])).value_counts()
    busy = on_routes[on_routes >= on_routes.quantile(0.5)].index
    candidates = [n for n in busy if degree[n] >= 3] or [n for n in router.nodes if degree[n] >= 3]
    candidates = sorted(candidates) or list(router.nodes)
    nodes = rng.choice(candidates, min(p.n_emerging, len(candidates)), replace=False)
    first = max(0, n_days - p.emerging_window_days)
    last = max(first + 1, n_days - 7)
    zone_nodes, zones = junction_zones(seg, p.emerging_radius_m)
    for hazard_id, node in enumerate(nodes):
        day = int(rng.integers(first, last))
        zone = zones[int(np.searchsorted(zone_nodes, node))].indices
        for sid in zone:
            start_day[sid] = min(start_day[sid], day)
            rows.append((hazard_id, node, int(sid), day))
    return start_day, pd.DataFrame(rows, columns=["hazard_id", "node", "segment_id", "start_day"])


def simulate(
    segments: gpd.GeoDataFrame,
    weather: pd.DataFrame,
    params: SimParams | None = None,
    chunk_trips: int = 20_000,
) -> SimResult:
    """Simulate a fleet over ``segments`` (from ``build_segments``) and hourly ``weather``."""
    p = params or SimParams()
    rng = np.random.default_rng(p.seed)
    seg = segments.reset_index(drop=True)
    if not (seg["segment_id"].to_numpy() == np.arange(len(seg))).all():
        raise ValueError("segment_id must equal row position 0..n-1")

    days = pd.date_range(p.start_date, p.end_date, freq="D")
    t0 = _to_seconds(days[0])
    hidden, black, latent_lit, static = _segment_truth(seg, p, rng)
    router = _Router(seg, p, rng)
    emerging_start, emerging = _emerging_hazards(seg, router, len(days), p, rng)

    # Weather lookup by hour since t0.
    w = weather.sort_values("ts")
    if w["ts"].iloc[0] > days[0] or w["ts"].iloc[-1] < days[-1] + pd.Timedelta(hours=23):
        raise ValueError("weather does not cover the simulation period")
    w_start = _to_seconds(w["ts"].iloc[0])
    precip = w["precip_mm"].fillna(0.0).to_numpy()

    seg_tt = seg["travel_time_s"].to_numpy(dtype=float)
    seg_km = seg["length_m"].to_numpy(dtype=float) / 1000
    seg_speed = seg["speed_kph"].to_numpy(dtype=float)
    geoms = np.asarray(seg.geometry.values)

    # Fleet
    n_v = p.n_vehicles
    vtype = rng.choice(len(VEHICLE_TYPES), n_v, p=[p.vehicle_mix[k] for k in VEHICLE_TYPES])
    cohort = rng.choice(len(DRIVER_COHORTS), n_v, p=[p.driver_mix[k] for k in DRIVER_COHORTS])
    depot = rng.integers(0, len(router.depots), n_v)
    dep_prob = {k: v / v.sum() for k, v in p.departure_profile.items()}

    # Schedule trips (outbound + return per delivery).
    trips, paths = [], []
    for day in days:
        day_s = _to_seconds(day)
        n_deliv = rng.poisson(p.deliveries_per_day, n_v)
        for vid in np.flatnonzero(n_deliv):
            vt = VEHICLE_TYPES[vtype[vid]]
            for _ in range(n_deliv[vid]):
                dest = router.destinations[rng.integers(len(router.destinations))]
                if dest == router.depots[depot[vid]]:
                    continue
                depart = day_s + int(rng.choice(24, p=dep_prob[vt])) * 3600 + int(rng.integers(3600))
                for outbound in (True, False):
                    variant = 0 if rng.random() < p.p_fastest else int(rng.integers(1, p.route_variants))
                    sids = router.path(depot[vid], dest, variant, outbound)
                    hour = (depart // 3600) % 24
                    duration = float(seg_tt[sids].sum() * p.congestion[hour])
                    origin, end = (router.depots[depot[vid]], dest) if outbound else (dest, router.depots[depot[vid]])
                    trips.append((vid, depot[vid], origin, end, variant, depart, duration,
                                  float(seg_km[sids].sum() * 1000)))
                    paths.append(sids)
                    depart = int(depart + duration + 60 * rng.uniform(*p.dwell_minutes))

    trip_df = pd.DataFrame(trips, columns=[
        "vehicle_id", "depot_id", "origin_node", "dest_node", "route_variant",
        "depart_s", "duration_s", "distance_m",
    ])
    trip_df.insert(0, "trip_id", np.arange(len(trip_df), dtype=np.int64))

    truth = pd.DataFrame({
        "segment_id": seg["segment_id"],
        "hidden_factor": hidden,
        "black_spot": black,
        "latent_lit": latent_lit,
        "emerging_start_day": np.where(emerging_start == NO_HAZARD, -1, emerging_start),
        **{f"true_mult_{t}": static[t] for t in EVENT_TYPES},
    })
    true_rates = TruthArrays(truth, seg, p)

    event_parts, tripseg_parts = [], []
    for c0 in range(0, len(trip_df), chunk_trips):
        chunk = trip_df.iloc[c0:c0 + chunk_trips]
        cpaths = paths[c0:c0 + chunk_trips]
        lens = np.array([len(x) for x in cpaths])
        sids = np.concatenate(cpaths)
        trip_ids = np.repeat(chunk["trip_id"].to_numpy(), lens)
        vids = np.repeat(chunk["vehicle_id"].to_numpy(), lens)
        depart = np.repeat(chunk["depart_s"].to_numpy(), lens)
        cong = p.congestion[(depart // 3600) % 24]
        tt = seg_tt[sids] * cong
        cs = np.cumsum(tt)
        first = np.concatenate(([0], np.cumsum(lens)[:-1]))
        enter = depart + (cs - tt) - np.repeat(cs[first] - tt[first], lens)

        hour = ((enter // 3600) % 24).astype(int)
        day_idx = ((enter - t0) // 86_400).astype(np.int64)
        rain = np.minimum(precip[np.clip(((enter - w_start) // 3600).astype(int), 0, len(precip) - 1)], 10) / 10

        tripseg_parts.append(pd.DataFrame({
            "trip_id": trip_ids,
            "seq": np.arange(len(sids)) - np.repeat(first, lens),
            "segment_id": sids,
            "enter_ts": _seconds_to_ts(enter),
        }))

        for t in EVENT_TYPES:
            lam = true_rates.rate(t, sids, hour, rain, vtype[vids], cohort[vids], day_idx)
            counts = rng.poisson(lam)
            idx = np.repeat(np.arange(len(sids)), counts)
            if not len(idx):
                continue
            frac = rng.random(len(idx))
            pts = shapely.line_interpolate_point(geoms[sids[idx]], frac, normalized=True)
            levels, probs = p.severity[t]
            event_parts.append(pd.DataFrame({
                "trip_id": trip_ids[idx],
                "vehicle_id": vids[idx],
                "ts": _seconds_to_ts(enter[idx] + frac * tt[idx]),
                "lat": shapely.get_y(pts),
                "lon": shapely.get_x(pts),
                "type": t,
                "severity": rng.choice(levels, len(idx), p=probs),
                "speed_kmph": np.round(seg_speed[sids[idx]] / cong[idx] * rng.uniform(0.6, 1.1, len(idx)), 1),
                "segment_id": sids[idx],
            }))

    events = pd.concat(event_parts, ignore_index=True).sort_values("ts", ignore_index=True)
    events.insert(0, "event_id", np.arange(len(events), dtype=np.int64))
    events.insert(3, "driver_id", events["vehicle_id"])  # one driver per vehicle

    # Trip table in schema form.
    per_trip = events.groupby("trip_id")["type"]
    trip_df["n_events"] = per_trip.size().reindex(trip_df["trip_id"], fill_value=0).to_numpy()
    for col, t in (("n_near_miss", "near_miss"), ("n_accident", "accident")):
        trip_df[col] = (
            events[events["type"] == t].groupby("trip_id").size()
            .reindex(trip_df["trip_id"], fill_value=0).to_numpy()
        )
    trip_df["driver_id"] = trip_df["vehicle_id"]
    trip_df["vehicle_type"] = np.array(VEHICLE_TYPES)[vtype[trip_df["vehicle_id"]]]
    trip_df["driver_cohort"] = np.array(DRIVER_COHORTS)[cohort[trip_df["vehicle_id"]]]
    trip_df["depart_ts"] = _seconds_to_ts(trip_df["depart_s"])
    trip_df["arrive_ts"] = trip_df["depart_ts"] + pd.to_timedelta(trip_df["duration_s"], unit="s")
    trip_df = trip_df.drop(columns="depart_s")

    emerging["start_date"] = days[0] + pd.to_timedelta(emerging["start_day"].astype(int), unit="D")

    return SimResult(
        events=validate(events, "events"),
        trips=validate(trip_df, "trips"),
        trip_segments=validate(pd.concat(tripseg_parts, ignore_index=True), "trip_segments"),
        truth=truth,
        emerging=emerging,
    )


MORTH_SLOTS = ["00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"]


def calibration_report(res: SimResult) -> dict:
    """Aggregates to compare against published statistics when tuning ``SimParams``.

    Time slots follow the 3-hour bands used in MoRTH's "Road Accidents in India".
    """
    ev, km = res.events, res.trips["distance_m"].sum() / 1000
    serious = ev[ev["type"].isin(["accident", "near_miss"])]
    slots = pd.Categorical(
        np.array(MORTH_SLOTS)[serious["ts"].dt.hour // 3], categories=MORTH_SLOTS
    )
    return {
        "trips": len(res.trips),
        "vehicle_km": round(km),
        "events": len(ev),
        "events_per_100km": (ev["type"].value_counts() / km * 100).round(3).to_dict(),
        "accidents": int((ev["type"] == "accident").sum()),
        "accidents_per_million_km": round((ev["type"] == "accident").sum() / km * 1e6, 1),
        "serious_event_share_by_slot_pct": (
            pd.Series(slots).value_counts(normalize=True, sort=False) * 100
        ).round(1).to_dict(),
        "emerging_hazards": int(res.emerging["hazard_id"].nunique()),
    }
