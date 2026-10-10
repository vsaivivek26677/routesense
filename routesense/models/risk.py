"""Segment risk model with uncertainty.

Two layers, per event group (harsh driving, near-miss, accident):

1. **Feature model** — a Poisson gradient-boosted model predicts the event rate per
   vehicle-km from road features (class, junctions, curvature, lighting, speed ...)
   and conditions (time band, rain, vehicle type). It generalises to roads with
   little or no history.
2. **Empirical-Bayes correction** — each segment gets a multiplier theta with a
   Gamma(alpha, alpha) prior: observed events O vs feature-expected events E give the
   posterior Gamma(alpha + O, alpha + E). Busy roads move towards their own history;
   quiet roads stay near the feature model and keep a wide confidence band.

The combined risk index weights groups by severity (``GROUP_WEIGHTS``) and is
expressed per km, for every segment x time band x rain x vehicle type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from routesense.ingest.osm import ROAD_CLASSES
from routesense.schema import VEHICLE_TYPES

# 3-hour slots, matching the time bands in MoRTH's "Road Accidents in India".
TIME_BANDS = tuple((f"{h:02d}-{h + 3:02d}", h, h + 3) for h in range(0, 24, 3))
BAND_NAMES = tuple(b[0] for b in TIME_BANDS)
HOUR_TO_BAND = np.array([next(i for i, (_, a, b) in enumerate(TIME_BANDS) if a <= h < b) for h in range(24)])
N_BANDS, N_RAIN, N_VEH = len(TIME_BANDS), 2, len(VEHICLE_TYPES)
REFERENCE = (BAND_NAMES.index("12-15"), 0, VEHICLE_TYPES.index("lcv"))  # for summaries

EVENT_GROUPS = {
    "harsh": ("harsh_brake", "harsh_turn", "harsh_accel"),
    "near_miss": ("near_miss",),
    "accident": ("accident",),
}
GROUPS = tuple(EVENT_GROUPS)
GROUP_WEIGHTS = {"harsh": 1.0, "near_miss": 10.0, "accident": 100.0}
WET_MM = 0.5  # an hour with >= 0.5 mm precipitation counts as wet

SEGMENT_FEATURES = [
    "road_class", "length_m", "speed_kph", "lanes", "lit", "junction_degree",
    "max_turn_deg", "total_turn_deg", "sinuosity", "is_bridge", "oneway",
]
CONDITION_FEATURES = ["band", "rain", "vehicle"]
FEATURES = SEGMENT_FEATURES + CONDITION_FEATURES
CATEGORICAL = ["road_class", "band", "vehicle"]


def band_of_hour(hour) -> np.ndarray:
    return HOUR_TO_BAND[np.asarray(hour) % 24]


def segment_features(segments: pd.DataFrame) -> pd.DataFrame:
    f = segments[SEGMENT_FEATURES].copy()
    f["road_class"] = f["road_class"].map({c: i for i, c in enumerate(ROAD_CLASSES)}).fillna(len(ROAD_CLASSES) - 1)
    return f.astype(float)


def _wet(ts: pd.Series, weather: pd.DataFrame) -> np.ndarray:
    w = weather.sort_values("ts")
    w0 = w["ts"].iloc[0]
    idx = ((ts - w0) // pd.Timedelta(hours=1)).to_numpy()
    precip = w["precip_mm"].fillna(0).to_numpy()
    return (precip[np.clip(idx, 0, len(precip) - 1)] >= WET_MM).astype(np.int64)


def exposure_table(
    segments: pd.DataFrame,
    trips: pd.DataFrame,
    trip_segments: pd.DataFrame,
    events: pd.DataFrame,
    weather: pd.DataFrame,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Vehicle-km and event counts per segment x band x rain x vehicle type in [start, end).

    Events take the conditions of the traversal they happened on (its entry time), so
    every event lands in a cell that has exposure.
    """
    ev = (
        events[["trip_id", "segment_id", "type"]].reset_index()
        .merge(trip_segments[["trip_id", "segment_id", "enter_ts"]], on=["trip_id", "segment_id"], how="inner")
        .drop_duplicates("index")
    )
    ts_mask = np.ones(len(trip_segments), bool)
    ev_mask = np.ones(len(ev), bool)
    if start is not None:
        ts_mask &= (trip_segments["enter_ts"] >= start).to_numpy()
        ev_mask &= (ev["enter_ts"] >= start).to_numpy()
    if end is not None:
        ts_mask &= (trip_segments["enter_ts"] < end).to_numpy()
        ev_mask &= (ev["enter_ts"] < end).to_numpy()
    ts, ev = trip_segments[ts_mask], ev[ev_mask]

    veh_code = pd.Series(
        trips["vehicle_type"].map({v: i for i, v in enumerate(VEHICLE_TYPES)}).to_numpy(),
        index=trips["trip_id"].to_numpy(),
    )
    km = segments["length_m"].to_numpy() / 1000
    n_keys = len(segments) * N_BANDS * N_RAIN * N_VEH

    def keys(sid, tstamp, trip_id):
        band = band_of_hour(tstamp.dt.hour.to_numpy())
        rain = _wet(tstamp, weather)
        veh = veh_code.reindex(trip_id).to_numpy()
        return ((sid * N_BANDS + band) * N_RAIN + rain) * N_VEH + veh

    sid = ts["segment_id"].to_numpy()
    exposure = np.bincount(keys(sid, ts["enter_ts"], ts["trip_id"]), weights=km[sid], minlength=n_keys)
    ev_keys = keys(ev["segment_id"].to_numpy(), ev["enter_ts"], ev["trip_id"])
    counts = {
        g: np.bincount(ev_keys[ev["type"].isin(types).to_numpy()], minlength=n_keys)
        for g, types in EVENT_GROUPS.items()
    }
    nz = np.flatnonzero(exposure > 0)
    veh = nz % N_VEH
    rain = (nz // N_VEH) % N_RAIN
    band = (nz // (N_VEH * N_RAIN)) % N_BANDS
    seg_id = nz // (N_VEH * N_RAIN * N_BANDS)
    out = pd.DataFrame({"segment_id": seg_id, "band": band, "rain": rain, "vehicle": veh,
                        "exposure_km": exposure[nz]})
    for g in GROUPS:
        out[f"n_{g}"] = counts[g][nz]
    return out


class ConstantRate:
    """Stand-in feature model for an event group with too few events to learn from."""

    def __init__(self, rate: float):
        self.rate = rate
        self.n_iter_ = 0

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.rate)


MIN_EVENTS_TO_MODEL = 20


@dataclass
class RiskTable:
    """Risk per km for every segment x band x rain x vehicle type (float32 arrays)."""

    mean: np.ndarray  # (n_seg, N_BANDS, N_RAIN, N_VEH) combined index per km
    var: np.ndarray
    group_mean: dict  # group -> (n_seg, N_BANDS, N_RAIN, N_VEH) events per km
    data_sparse: np.ndarray  # (n_seg,) bool: too few fleet passes to trust the road's own history

    def at(self, band: int, rain: int, vehicle: int) -> tuple[np.ndarray, np.ndarray]:
        return self.mean[:, band, rain, vehicle], self.var[:, band, rain, vehicle]


@dataclass
class RiskModel:
    models: dict = field(default_factory=dict)  # group -> fitted regressor
    alpha: dict = field(default_factory=dict)
    theta_mean: dict = field(default_factory=dict)  # group -> (n_seg,)
    theta_var: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    exposure_km: np.ndarray | None = None
    features: list = field(default_factory=lambda: list(FEATURES))
    # Cross-fitting: each segment's feature-model prior comes from a model trained without
    # its road link, so the empirical-Bayes step compares history with an honest prior.
    n_folds: int = 3
    seg_fold: np.ndarray | None = None
    fold_models: dict = field(default_factory=dict)  # group -> list of per-fold models (or None)
    min_passes: int = 30  # fewer fleet passes than this = data-sparse

    @classmethod
    def fit(cls, segments: pd.DataFrame, exposure: pd.DataFrame, seed: int = 0, **gbm) -> "RiskModel":
        self = cls()
        n = len(segments)
        seg_f = segment_features(segments)
        # Features with < 2 distinct values (e.g. an untagged attribute) carry no signal.
        self.features = [c for c in FEATURES if c in CONDITION_FEATURES or seg_f[c].nunique() >= 2]
        X = _design(seg_f, exposure)[self.features].to_numpy()
        w = exposure["exposure_km"].to_numpy()
        sid = exposure["segment_id"].to_numpy()
        self.exposure_km = np.bincount(sid, weights=w, minlength=n)
        link = pd.factorize(pd.Series(list(zip(np.minimum(segments["u"], segments["v"]),
                                               np.maximum(segments["u"], segments["v"])))))[0]
        perm = np.random.default_rng(seed).permutation(link.max() + 1)
        self.seg_fold = perm[link] % self.n_folds
        row_fold = self.seg_fold[sid]
        params = dict(loss="poisson", learning_rate=0.08, max_iter=300, max_leaf_nodes=31,
                      min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                      validation_fraction=0.1, random_state=seed,
                      categorical_features=[self.features.index(c) for c in CATEGORICAL if c in self.features])
        params.update(gbm)
        def fit_one(rows):
            yy, ww = y[rows], w[rows]
            if yy.sum() < MIN_EVENTS_TO_MODEL:  # e.g. no accidents yet: one smoothed rate for all roads
                return ConstantRate((yy.sum() + 0.5) / ww.sum())
            return HistGradientBoostingRegressor(**params).fit(X[rows], yy / ww, sample_weight=ww)

        everything = np.ones(len(sid), bool)
        for g in GROUPS:
            y = exposure[f"n_{g}"].to_numpy()
            self.models[g] = fit_one(everything)
            if isinstance(self.models[g], ConstantRate):
                self.fold_models[g] = None
                oof = self.models[g].predict(X)
            else:
                self.fold_models[g] = []
                oof = np.empty(len(sid))
                for k in range(self.n_folds):
                    m = fit_one(row_fold != k)
                    oof[row_fold == k] = m.predict(X[row_fold == k])
                    self.fold_models[g].append(m)
            O = np.bincount(sid, weights=y, minlength=n)
            E = np.bincount(sid, weights=oof * w, minlength=n)
            seen = E > 0
            inv_alpha = ((O[seen] - E[seen]) ** 2 - O[seen]).sum() / (E[seen] ** 2).sum()
            alpha = 1 / float(np.clip(inv_alpha, 1e-3, 1e3))
            self.alpha[g] = alpha
            self.observed[g], self.expected[g] = O, E
            self.theta_mean[g] = (alpha + O) / (alpha + E)
            self.theta_var[g] = (alpha + O) / (alpha + E) ** 2
        return self

    def predict_prior(self, group: str, rows: pd.DataFrame, segment_ids=None) -> np.ndarray:
        """Feature-model rate per km for design rows (segment features + conditions).

        With ``segment_ids``, each row uses the fold model that never saw that road (the
        honest prior for known roads); without, the model trained on everything (new roads).
        """
        X = rows[self.features].to_numpy()
        folds = self.fold_models.get(group)
        if segment_ids is None or not folds:
            return self.models[group].predict(X)
        which = self.seg_fold[np.asarray(segment_ids)]
        out = np.empty(len(X))
        for k, m in enumerate(folds):
            sel = which == k
            if sel.any():
                out[sel] = m.predict(X[sel])
        return out

    def prior_rate(self, group: str, seg_f: pd.DataFrame, band: int, rain: int, vehicle: int,
                   segment_ids=None) -> np.ndarray:
        """Feature-model rate per km for segments described by ``seg_f`` under one condition."""
        rows = seg_f.assign(band=float(band), rain=float(rain), vehicle=float(vehicle))
        return self.predict_prior(group, rows, segment_ids)

    def risk_table(self, segments: pd.DataFrame) -> RiskTable:
        n = len(segments)
        seg_f = segment_features(segments)
        conds = [(b, r, v) for b in range(N_BANDS) for r in range(N_RAIN) for v in range(N_VEH)]
        big = pd.concat([seg_f.assign(band=float(b), rain=float(r), vehicle=float(v)) for b, r, v in conds])
        shape = (n, N_BANDS, N_RAIN, N_VEH)
        mean = np.zeros(shape, np.float64)
        var = np.zeros(shape, np.float64)
        group_mean = {}
        seg_ids = np.tile(np.arange(n), len(conds))
        for g in GROUPS:
            prior = self.predict_prior(g, big, seg_ids).reshape(len(conds), n).T.reshape(shape)
            rate = prior * self.theta_mean[g][:, None, None, None]
            group_mean[g] = rate.astype(np.float32)
            mean += GROUP_WEIGHTS[g] * rate
            var += GROUP_WEIGHTS[g] ** 2 * prior**2 * self.theta_var[g][:, None, None, None]
        passes = self.exposure_km / np.maximum(segments["length_m"].to_numpy() / 1000, 1e-6)
        return RiskTable(mean.astype(np.float32), var.astype(np.float32), group_mean, passes < self.min_passes)

    def credibility(self, group: str) -> np.ndarray:
        """Weight of each segment's own history in its estimate (0 = prior only, 1 = data only)."""
        E = self.expected[group]
        return E / (self.alpha[group] + E)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "RiskModel":
        return joblib.load(path)


def _design(seg_f: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    X = seg_f.iloc[exposure["segment_id"].to_numpy()].reset_index(drop=True)
    X["band"] = exposure["band"].to_numpy(float)
    X["rain"] = exposure["rain"].to_numpy(float)
    X["vehicle"] = exposure["vehicle"].to_numpy(float)
    return X


def save_table(table: RiskTable, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=table.mean, var=table.var, data_sparse=table.data_sparse,
                        **{f"group_{g}": table.group_mean[g] for g in GROUPS})


def load_table(path: Path) -> RiskTable:
    z = np.load(path)
    return RiskTable(z["mean"], z["var"], {g: z[f"group_{g}"] for g in GROUPS}, z["data_sparse"])
