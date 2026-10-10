"""Route risk scoring, safer alternatives and the departure-time advisor.

Costs are per directed segment for one set of conditions (departure time, rain,
vehicle type). Candidate routes come from shortest paths under
``time + lambda * kappa * risk`` for a sweep of lambdas (kappa puts risk on the same
scale as time network-wide); only routes no other candidate beats on both time and
risk are offered.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from routesense.ingest.osm import node_table
from routesense.models.risk import GROUPS, REFERENCE, RiskTable, band_of_hour
from routesense.schema import VEHICLE_TYPES

LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0)
LEVELS = ("low", "moderate", "high", "very high")


@dataclass(frozen=True)
class Conditions:
    depart: pd.Timestamp
    rain: bool = False
    vehicle_type: str = "lcv"
    risk_aversion: float = 0.0  # z: plan on mean + z * sd (0 = expected risk)

    @property
    def vehicle(self) -> int:
        return VEHICLE_TYPES.index(self.vehicle_type)


@dataclass
class Policy:
    """When is a slower route worth recommending?"""

    max_detour: float = 0.35  # never offer routes > 35% slower than the fastest
    max_extra_pct: float = 0.15  # recommend at most +15% travel time ...
    free_extra_min: float = 3.0  # ... or +3 minutes if that is more (short trips) ...
    max_extra_min: float = 10.0  # ... and never more than +10 minutes
    min_risk_cut: float = 0.10  # only worth it for >= 10% lower risk
    min_time_saving_min: float = 5.0  # ... or a smaller risk cut that also saves 5+ minutes


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, q=(0.5, 0.8, 0.95)) -> np.ndarray:
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights.sum()
    return values[order][np.minimum(np.searchsorted(cum, q), len(values) - 1)]


def fleet_level_thresholds(table: RiskTable, exposure: pd.DataFrame) -> np.ndarray:
    """Per-km risk at the 50th/80th/95th percentile of the vehicle-km the fleet drives.

    A route is "moderate" when riskier than half of all fleet driving, "high" above
    80% and "very high" above 95% (across all times, weather and vehicle types).
    """
    values = table.mean[
        exposure["segment_id"].to_numpy(), exposure["band"].to_numpy(),
        exposure["rain"].to_numpy(), exposure["vehicle"].to_numpy(),
    ]
    return _weighted_quantiles(values, exposure["exposure_km"].to_numpy())


def congestion_profile(trips: pd.DataFrame, trip_segments: pd.DataFrame, segments: pd.DataFrame) -> np.ndarray:
    """Median ratio of actual to free-flow trip time by departure hour (from trip logs)."""
    free = np.bincount(
        trip_segments["trip_id"].to_numpy(),
        weights=segments["travel_time_s"].to_numpy()[trip_segments["segment_id"].to_numpy()],
        minlength=int(trips["trip_id"].max()) + 1,
    )[trips["trip_id"].to_numpy()]
    ok = free > 0
    ratio = pd.Series(trips["duration_s"].to_numpy()[ok] / free[ok])
    by_hour = ratio.groupby(trips["depart_ts"].dt.hour.to_numpy()[ok]).median()
    return by_hour.reindex(range(24)).interpolate(limit_direction="both").fillna(1.0).to_numpy()


@dataclass
class RouteEngine:
    segments: pd.DataFrame
    table: RiskTable
    congestion: np.ndarray
    policy: Policy = field(default_factory=Policy)
    level_thresholds: np.ndarray | None = None  # see ``fleet_level_thresholds``

    def __post_init__(self):
        seg = self.segments
        self.km = seg["length_m"].to_numpy() / 1000
        self.tt_free = seg["travel_time_s"].to_numpy(dtype=float)
        self.names = seg["name"].astype(object).where(seg["name"].notna(), None).to_numpy()
        coords = seg.geometry.map(lambda g: list(g.coords))
        self.coords = coords.to_numpy()

        # One edge per (u, v): the fastest parallel segment.
        best = seg.assign(_i=np.arange(len(seg))).sort_values("travel_time_s").drop_duplicates(["u", "v"])
        nodes = node_table(seg)
        self.node_ids = nodes.index.to_numpy()
        index = pd.Series(np.arange(len(nodes)), index=nodes.index)
        self.edge_u = index[best["u"]].to_numpy()
        self.edge_v = index[best["v"]].to_numpy()
        self.edge_sid = best["_i"].to_numpy()
        n = len(nodes)
        self._edge_lookup = csr_matrix((self.edge_sid + 1, (self.edge_u, self.edge_v)), shape=(n, n))

        # Only snap to nodes in the largest strongly connected component (always routable).
        _, labels = connected_components(self._edge_lookup, directed=True, connection="strong")
        main = labels == np.bincount(labels).argmax()
        self.lat0 = float(nodes["lat"].mean())
        self._snap_idx = np.flatnonzero(main)
        self._tree = cKDTree(self._xy(nodes["lat"].to_numpy()[main], nodes["lon"].to_numpy()[main]))
        self.node_lat = nodes["lat"].to_numpy()
        self.node_lon = nodes["lon"].to_numpy()

        if self.level_thresholds is None:  # fallback: length-weighted, reference conditions
            b, r, v = REFERENCE
            self.level_thresholds = _weighted_quantiles(self.table.mean[:, b, r, v], self.km)

    # ---- helpers -------------------------------------------------------------
    def _xy(self, lat, lon):
        return np.column_stack((np.asarray(lon) * 111_320 * np.cos(np.radians(self.lat0)), np.asarray(lat) * 110_540))

    def nearest_node(self, lat: float, lon: float) -> int:
        """Index (into the node arrays) of the closest routable node."""
        _, i = self._tree.query(self._xy([lat], [lon])[0])
        return int(self._snap_idx[i])

    def level(self, risk_per_km: float) -> str:
        return LEVELS[int(np.searchsorted(self.level_thresholds, risk_per_km))]

    def segment_costs(self, cond: Conditions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(travel time s, expected risk, risk sd) per segment, at the departure hour."""
        hour = cond.depart.hour
        band = int(band_of_hour(hour))
        mean, var = self.table.at(band, int(cond.rain), cond.vehicle)
        return self.tt_free * self.congestion[hour], mean * self.km, np.sqrt(var) * self.km

    def shortest(self, origin: int, dest: int, seg_weight: np.ndarray) -> np.ndarray:
        """Segment indices of the min-weight path between node indices."""
        w = np.maximum(seg_weight[self.edge_sid], 1e-9)
        n = len(self.node_ids)
        g = csr_matrix((w, (self.edge_u, self.edge_v)), shape=(n, n))
        dist, pred = dijkstra(g, indices=origin, return_predecessors=True)
        if not np.isfinite(dist[dest]):
            raise ValueError("no route between these points")
        nodes = [dest]
        while nodes[-1] != origin:
            nodes.append(pred[nodes[-1]])
        nodes.reverse()
        return np.array([self._edge_lookup[a, b] - 1 for a, b in zip(nodes[:-1], nodes[1:])], dtype=np.int64)

    # ---- scoring ---------------------------------------------------------------
    def score(self, sids: np.ndarray, cond: Conditions, top: int = 5) -> dict:
        """Score a route, letting the time band change as the trip progresses."""
        tt = self.tt_free[sids] * self.congestion[cond.depart.hour]
        enter = cond.depart + pd.to_timedelta(np.concatenate(([0.0], np.cumsum(tt)[:-1])), unit="s")
        band = band_of_hour(enter.hour.to_numpy())
        rain, veh = int(cond.rain), cond.vehicle
        risk = self.table.mean[sids, band, rain, veh] * self.km[sids]
        sd = np.sqrt(self.table.var[sids, band, rain, veh]) * self.km[sids]
        total, total_sd = float(risk.sum()), float(np.sqrt((sd**2).sum()))
        distance = float(self.km[sids].sum())
        expected = {
            g: float((self.table.group_mean[g][sids, band, rain, veh] * self.km[sids]).sum()) for g in GROUPS
        }
        sparse_km = float(self.km[sids][self.table.data_sparse[sids]].sum())

        per_km = self.table.mean[sids, band, rain, veh]
        typical = float(self.level_thresholds[0])  # median fleet km
        worst = np.argsort(-risk)[:top]
        hotspots = [
            {
                "segment_id": int(sids[i]),
                "position": int(i),
                "name": self.names[sids[i]],
                "share": float(risk[i] / total) if total else 0.0,
                "times_typical": float(per_km[i] / typical) if typical else None,
                "km_from_start": float(self.km[sids[:i]].sum()),
                "lat": float(np.mean([c[1] for c in self.coords[sids[i]]])),
                "lon": float(np.mean([c[0] for c in self.coords[sids[i]]])),
                "data_sparse": bool(self.table.data_sparse[sids[i]]),
            }
            for i in worst
        ]
        return {
            "segment_ids": [int(s) for s in sids],
            "duration_s": float(tt.sum()),
            "distance_km": distance,
            "risk": total,
            "risk_lo": max(0.0, total - 1.96 * total_sd),
            "risk_hi": total + 1.96 * total_sd,
            "risk_per_km": total / distance if distance else 0.0,
            "level": self.level(total / distance) if distance else "low",
            "near_miss_per_1k_trips": 1000 * expected["near_miss"],
            "accidents_per_1k_trips": 1000 * expected["accident"],
            "harsh_events_per_trip": expected["harsh"],
            "sparse_km_share": sparse_km / distance if distance else 0.0,
            "top_segments": hotspots,
        }

    def geometry(self, sids) -> list[list[float]]:
        """[[lat, lon], ...] polyline for a route."""
        pts: list[list[float]] = []
        for s in sids:
            c = self.coords[s]
            pts.extend([[y, x] for x, y in (c if not pts else c[1:])])
        return pts

    # ---- alternatives ------------------------------------------------------------
    def options(self, origin: int, dest: int, cond: Conditions) -> list[dict]:
        """Pareto-optimal (time, risk) routes, fastest first, with a recommendation."""
        tt, risk, sd = self.segment_costs(cond)
        planning_risk = risk + cond.risk_aversion * sd
        kappa = tt.sum() / max(planning_risk.sum(), 1e-12)

        candidates = {"fastest": self.shortest(origin, dest, tt)}
        for lam in LAMBDAS:
            candidates[f"lambda_{lam}"] = self.shortest(origin, dest, tt + lam * kappa * planning_risk)

        scored, seen = [], set()
        for key, sids in candidates.items():
            sig = tuple(sids)
            if sig in seen:
                continue
            seen.add(sig)
            s = self.score(sids, cond)
            s["planning_risk"] = s["risk"] + cond.risk_aversion * (s["risk_hi"] - s["risk"]) / 1.96
            s["source"] = key
            scored.append(s)

        fastest = next(s for s in scored if s["source"] == "fastest")
        limit = fastest["duration_s"] * (1 + self.policy.max_detour)
        pool = [s for s in scored if s["duration_s"] <= limit]
        pareto = [
            s for s in pool
            if not any(
                o is not s and o["duration_s"] <= s["duration_s"] and o["planning_risk"] <= s["planning_risk"]
                and (o["duration_s"] < s["duration_s"] or o["planning_risk"] < s["planning_risk"])
                for o in pool
            )
        ]
        if fastest not in pareto:  # ties: keep the true fastest route as the reference
            pareto.append(fastest)
        pareto.sort(key=lambda s: s["duration_s"])
        # Ensure the reference is first even when another route ties on time.
        pareto.remove(fastest)
        pareto.insert(0, fastest)

        for i, s in enumerate(pareto):
            s["option_id"] = i
            s["extra_min"] = (s["duration_s"] - fastest["duration_s"]) / 60
            s["extra_time_pct"] = s["duration_s"] / fastest["duration_s"] - 1 if fastest["duration_s"] else 0.0
            s["risk_change_pct"] = s["risk"] / fastest["risk"] - 1 if fastest["risk"] else 0.0
            s["geometry"] = self.geometry(s["segment_ids"])
            s["recommended"] = False
        self._recommend(pareto)
        # Names: Fastest, Safest, Balanced (a recommended in-between route), Alternative n.
        safest = min(pareto, key=lambda s: s["planning_risk"])
        n_alt = 0
        for s in pareto:
            if s is fastest:
                s["label"] = "Fastest"
            elif s is safest:
                s["label"] = "Safest"
            elif s["recommended"]:
                s["label"] = "Balanced"
            else:
                n_alt += 1
                s["label"] = f"Alternative {n_alt}"
        return pareto

    def _recommend(self, opts: list[dict]) -> None:
        p = self.policy
        ok = [
            s for s in opts[1:]
            if s["extra_min"] <= min(p.max_extra_min, max(p.free_extra_min, p.max_extra_pct * opts[0]["duration_s"] / 60))
            and -s["risk_change_pct"] >= p.min_risk_cut
        ]
        best = min(ok, key=lambda s: s["planning_risk"]) if ok else opts[0]
        best["recommended"] = True

    # ---- departure-time advisor -----------------------------------------------------
    def departure_advice(
        self,
        origin: int,
        dest: int,
        cond: Conditions,
        before_min: int = 60,
        after_min: int = 120,
        step_min: int = 15,
        arrive_by: pd.Timestamp | None = None,
    ) -> dict:
        """Risk and travel time of the recommended route for each departure slot."""
        slots = []
        for offset in range(-before_min, after_min + 1, step_min):
            c = replace(cond, depart=cond.depart + pd.Timedelta(minutes=offset))
            rec = next(o for o in self.options(origin, dest, c) if o["recommended"])
            slots.append({
                "offset_min": offset,
                "depart": c.depart,
                "arrive": c.depart + pd.Timedelta(seconds=rec["duration_s"]),
                "duration_s": rec["duration_s"],
                "risk": rec["risk"],
                "risk_lo": rec["risk_lo"],
                "risk_hi": rec["risk_hi"],
                "label": rec["label"],
            })
        base = next(s for s in slots if s["offset_min"] == 0)
        feasible = [s for s in slots if arrive_by is None or s["arrive"] <= arrive_by]
        best = min(feasible, key=lambda s: (s["risk"], s["duration_s"])) if feasible else base
        cut = 1 - best["risk"] / base["risk"] if base["risk"] else 0.0
        saved_min = (base["duration_s"] - best["duration_s"]) / 60
        p = self.policy
        worth_it = best is not base and (
            cut >= p.min_risk_cut or (cut >= p.min_risk_cut / 3 and saved_min >= p.min_time_saving_min)
        )
        if worth_it:
            time_note = ("same travel time" if abs(saved_min) < 0.5
                         else f"{abs(saved_min):.0f} min {'faster' if saved_min > 0 else 'slower'}")
            advice = (
                f"Leave at {best['depart']:%H:%M} instead of {base['depart']:%H:%M}: "
                f"{cut:.0%} lower risk, {time_note}."
            )
        else:
            best, cut = base, 0.0
            advice = f"Departing at {base['depart']:%H:%M} is already close to the lowest-risk time in this window."
        return {"slots": slots, "best": best, "risk_cut": cut, "advice": advice}
