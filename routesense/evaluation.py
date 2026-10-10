"""Backtest: does the risk score predict future incidents, and what would using it change?

Everything is fitted on data before ``split`` and scored on data after it.

1. Segment level: predicted vs observed serious events (near-miss + accident) in the
   test period, for the full model and three baselines (distance only, raw history,
   features only): deviance explained and top-10% lift.
2. Ground truth: rank correlation of each method with the simulator's hidden risk.
3. Trip level: does a trip's predicted risk separate trips that had a serious event?
4. Emerging hotspots: detection delay and false alarms over the test period.
5. Impact: for a sample of test trips, the recommended route vs the fastest route,
   both scored with the simulator's TRUE rates (not our model).
6. Speed: fit and request timings.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from routesense.models.emerging import DailyCounts, replay, score_detection
from routesense.models.risk import (
    GROUPS, REFERENCE, RiskModel, _wet, band_of_hour, exposure_table, segment_features,
)
from routesense.routing.engine import Conditions, Policy, RouteEngine, congestion_profile, fleet_level_thresholds
from routesense.schema import DRIVER_COHORTS, VEHICLE_TYPES
from routesense.sim.params import SimParams
from routesense.sim.simulator import SimResult, TruthArrays

SERIOUS = ("near_miss", "accident")


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-12)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * (term - (y - mu)).sum())


def _lift_top(rate: np.ndarray, km: np.ndarray, y: np.ndarray, top: float = 0.1) -> float:
    """Observed serious-event rate on the riskiest ``top`` share of test km vs overall."""
    order = np.argsort(-rate)
    cum = np.cumsum(km[order]) / km.sum()
    sel = order[: max(1, np.searchsorted(cum, top) + 1)]
    overall = y.sum() / km.sum()
    return float((y[sel].sum() / km[sel].sum()) / overall) if overall > 0 else float("nan")


def segment_backtest(segments, model: RiskModel, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    seg_f = segment_features(segments)
    n = len(segments)
    y_rows = sum(test[f"n_{g}"].to_numpy() for g in SERIOUS)
    km_rows = test["exposure_km"].to_numpy()
    sid = test["segment_id"].to_numpy()

    # Predicted serious rate per km for every test row, per method.
    prior = np.zeros(len(test))
    full = np.zeros(len(test))
    for g in SERIOUS:
        rows = seg_f.iloc[sid].reset_index(drop=True).assign(
            band=test["band"].to_numpy(float), rain=test["rain"].to_numpy(float), vehicle=test["vehicle"].to_numpy(float)
        )
        p = model.predict_prior(g, rows, sid)
        prior += p
        full += p * model.theta_mean[g][sid]
    train_y = np.bincount(train["segment_id"], weights=sum(train[f"n_{g}"] for g in SERIOUS), minlength=n)
    train_km = np.bincount(train["segment_id"], weights=train["exposure_km"], minlength=n)
    global_rate = train_y.sum() / train_km.sum()
    raw_seg = (train_y + 0.5) / (train_km + 0.5 / global_rate)  # minimal smoothing so unseen roads != 0
    methods = {
        "distance_only": np.full(len(test), global_rate),
        "raw_history": raw_seg[sid],
        "features_only": prior,
        "full_model": full,
    }

    y = np.bincount(sid, weights=y_rows, minlength=n)
    km = np.bincount(sid, weights=km_rows, minlength=n)
    driven = km > 0
    null = _poisson_deviance(y[driven], (global_rate * km)[driven])
    out = {}
    for name, rate in methods.items():
        mu = np.bincount(sid, weights=rate * km_rows, minlength=n)
        seg_rate = np.where(driven, mu / np.maximum(km, 1e-12), 0)
        out[name] = {
            "deviance_explained": round(1 - _poisson_deviance(y[driven], mu[driven]) / null, 4),
            "top10pct_km_lift": round(_lift_top(seg_rate[driven], km[driven], y[driven]), 3),
            "predicted_total": round(float(mu.sum()), 1),
        }
    out["observed_total"] = int(y.sum())
    return out


def truth_recovery(segments, model: RiskModel, truth: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Spearman correlation with the simulator's hidden per-segment multipliers."""
    seg_f = segment_features(segments)
    b, r, v = REFERENCE
    km = np.bincount(train["segment_id"], weights=train["exposure_km"], minlength=len(segments))
    driven = km > 5
    out = {}
    for g, t in (("near_miss", "near_miss"), ("harsh", "harsh_brake")):
        p = model.prior_rate(g, seg_f, b, r, v, segment_ids=np.arange(len(segments)))
        raw = model.observed[g] / np.maximum(model.exposure_km, 1e-12)
        target = truth[f"true_mult_{t}"].to_numpy()[driven]
        out[g] = {
            "raw_history": round(float(spearmanr(raw[driven], target)[0]), 3),
            "features_only": round(float(spearmanr(p[driven], target)[0]), 3),
            "full_model": round(float(spearmanr((p * model.theta_mean[g])[driven], target)[0]), 3),
        }
    out["segments_compared"] = int(driven.sum())
    return out


def trip_backtest(segments, table, sim: SimResult, weather: pd.DataFrame, split) -> dict:
    trips = sim.trips[sim.trips["depart_ts"] >= split]
    ts = sim.trip_segments[sim.trip_segments["trip_id"].isin(trips["trip_id"])]
    veh = trips.set_index("trip_id")["vehicle_type"].map({v: i for i, v in enumerate(VEHICLE_TYPES)})
    sid = ts["segment_id"].to_numpy()
    band = band_of_hour(ts["enter_ts"].dt.hour.to_numpy())
    rain = _wet(ts["enter_ts"], weather)
    v = veh.reindex(ts["trip_id"]).to_numpy()
    km = segments["length_m"].to_numpy()[sid] / 1000
    serious = sum(table.group_mean[g][sid, band, rain, v] for g in SERIOUS) * km
    pred = pd.Series(serious).groupby(ts["trip_id"].to_numpy()).sum().reindex(trips["trip_id"]).to_numpy()
    y = ((trips["n_near_miss"] + trips["n_accident"]) > 0).to_numpy()
    deciles = pd.qcut(pred, 10, labels=False, duplicates="drop")
    calib = pd.DataFrame({"pred": pred, "obs": (trips["n_near_miss"] + trips["n_accident"]).to_numpy(), "d": deciles})
    calib = calib.groupby("d")[["pred", "obs"]].mean()
    return {
        "test_trips": int(len(trips)),
        "trips_with_serious_event_pct": round(100 * y.mean(), 2),
        "auc_full_model": round(float(roc_auc_score(y, pred)), 3),
        "auc_distance_only": round(float(roc_auc_score(y, trips["distance_m"])), 3),
        "auc_duration_only": round(float(roc_auc_score(y, trips["duration_s"])), 3),
        "calibration_by_decile": {
            "predicted": calib["pred"].round(4).tolist(),
            "observed": calib["obs"].round(4).tolist(),
        },
    }


def impact(engine: RouteEngine, truth: TruthArrays, sim: SimResult, weather: pd.DataFrame, params: SimParams,
           split, n_trips: int = 1500, seed: int = 0) -> dict:
    """Recommended vs fastest route on sampled test trips, both scored with TRUE rates."""
    trips = sim.trips[sim.trips["depart_ts"] >= split]
    trips = trips.sample(min(n_trips, len(trips)), random_state=seed)
    node_pos = pd.Series(np.arange(len(engine.node_ids)), index=engine.node_ids)
    w = weather.set_index("ts")["precip_mm"]
    start = pd.Timestamp(params.start_date)

    def true_serious(sids, depart, rain_mm, vt, coh):
        tt = engine.tt_free[sids] * engine.congestion[depart.hour]
        enter = depart + pd.to_timedelta(np.concatenate(([0.0], np.cumsum(tt)[:-1])), unit="s")
        hour = enter.hour.to_numpy()
        day = np.full(len(sids), (depart.normalize() - start).days)
        rain = np.full(len(sids), min(rain_mm, 10) / 10)
        return sum(truth.rate(t, sids, hour, rain, np.full(len(sids), vt), np.full(len(sids), coh), day).sum()
                   for t in SERIOUS)

    rows = []
    for t in trips.itertuples():
        rain_mm = float(w.get(t.depart_ts.floor("h"), 0.0))
        cond = Conditions(t.depart_ts, rain=rain_mm >= 0.5, vehicle_type=t.vehicle_type)
        o, d = node_pos[t.origin_node], node_pos[t.dest_node]
        opts = engine.options(o, d, cond)
        fastest, rec = opts[0], next(x for x in opts if x["recommended"])
        vt, coh = VEHICLE_TYPES.index(t.vehicle_type), DRIVER_COHORTS.index(t.driver_cohort)
        f_true = true_serious(np.array(fastest["segment_ids"]), t.depart_ts, rain_mm, vt, coh)
        r_true = true_serious(np.array(rec["segment_ids"]), t.depart_ts, rain_mm, vt, coh)
        rows.append((f_true, r_true, fastest["duration_s"], rec["duration_s"], rec is not fastest))
    df = pd.DataFrame(rows, columns=["fastest_true", "rec_true", "fastest_s", "rec_s", "rerouted"])
    cut = 1 - df["rec_true"].sum() / df["fastest_true"].sum()
    rr = df[df["rerouted"]]
    return {
        "trips_evaluated": int(len(df)),
        "rerouted_pct": round(100 * df["rerouted"].mean(), 1),
        "true_serious_events_cut_pct_all_trips": round(100 * cut, 2),
        "true_serious_events_cut_pct_rerouted_trips": round(100 * (1 - rr["rec_true"].sum() / rr["fastest_true"].sum()), 2) if len(rr) else 0.0,
        "extra_minutes_per_rerouted_trip": round(float((rr["rec_s"] - rr["fastest_s"]).mean() / 60), 2) if len(rr) else 0.0,
        "extra_time_pct_all_trips": round(100 * (df["rec_s"].sum() / df["fastest_s"].sum() - 1), 2),
        "serious_events_avoided_per_10k_trips": round(10_000 * (df["fastest_true"] - df["rec_true"]).mean(), 2),
        "note": "Both routes scored with the simulator's true event rates, not the model's estimates.",
    }


def run(segments, sim: SimResult, weather: pd.DataFrame, params: SimParams, split="2026-07-01",
        impact_trips: int = 1500, policy: Policy | None = None) -> dict:
    split = pd.Timestamp(split)
    timings = {}
    t = time.perf_counter()
    train = exposure_table(segments, sim.trips, sim.trip_segments, sim.events, weather, end=split)
    test = exposure_table(segments, sim.trips, sim.trip_segments, sim.events, weather, start=split)
    timings["exposure_tables_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    model = RiskModel.fit(segments, train)
    timings["model_fit_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    table = model.risk_table(segments)
    timings["risk_table_s"] = round(time.perf_counter() - t, 2)

    report = {
        "split": str(split.date()),
        "data": {"segments": len(segments), "trips": len(sim.trips), "events": len(sim.events),
                 "train_vehicle_km": round(float(train["exposure_km"].sum())),
                 "test_vehicle_km": round(float(test["exposure_km"].sum()))},
        "segment_backtest": segment_backtest(segments, model, train, test),
        "truth_recovery": truth_recovery(segments, model, sim.truth, train),
        "trip_backtest": trip_backtest(segments, table, sim, weather, split),
    }

    dc = DailyCounts(segments, sim.trip_segments, sim.events)
    t = time.perf_counter()
    flags = replay(dc, split, dc.days[-1])
    timings["emerging_replay_s"] = round(time.perf_counter() - t, 2)
    report["emerging_detection"] = score_detection(flags, sim.emerging, split, dc.days[-1], dc)

    t = time.perf_counter()
    cong = congestion_profile(sim.trips[sim.trips["depart_ts"] < split], sim.trip_segments, segments)
    engine = RouteEngine(segments, table, cong, policy=policy or Policy(), level_thresholds=fleet_level_thresholds(table, train))
    timings["engine_build_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    report["impact"] = impact(engine, TruthArrays(sim.truth, segments, params), sim, weather, params, split, impact_trips)
    report["impact"]["policy"] = asdict(engine.policy)
    timings["route_options_ms_per_trip"] = round(1000 * (time.perf_counter() - t) / max(report["impact"]["trips_evaluated"], 1), 1)
    report["timings"] = timings
    return report


def to_markdown(r: dict) -> str:
    sb, tr, tb, em, im = (r[k] for k in ("segment_backtest", "truth_recovery", "trip_backtest", "emerging_detection", "impact"))
    lines = [
        f"# Backtest (train < {r['split']} <= test)", "",
        f"Data: {r['data']['segments']:,} segments, {r['data']['trips']:,} trips, {r['data']['events']:,} events; "
        f"train {r['data']['train_vehicle_km']:,} vehicle-km, test {r['data']['test_vehicle_km']:,} vehicle-km.", "",
        "## Segment level: serious events (near-miss + accident) in the test period", "",
        "| Method | Deviance explained | Top-10% km lift |", "|---|---|---|",
    ]
    for m in ("distance_only", "raw_history", "features_only", "full_model"):
        lines.append(f"| {m.replace('_', ' ')} | {sb[m]['deviance_explained']:.3f} | {sb[m]['top10pct_km_lift']:.2f}x |")
    lines += ["", "## Agreement with the simulator's hidden risk (Spearman)", "",
              "| Event group | Raw history | Features only | Full model |", "|---|---|---|---|"]
    for g in ("near_miss", "harsh"):
        lines.append(f"| {g.replace('_', ' ')} | {tr[g]['raw_history']} | {tr[g]['features_only']} | {tr[g]['full_model']} |")
    lines += ["", "## Trip level", "",
              f"{tb['test_trips']:,} test trips, {tb['trips_with_serious_event_pct']}% had a serious event. "
              f"AUC: full model {tb['auc_full_model']}, distance only {tb['auc_distance_only']}, duration only {tb['auc_duration_only']}.",
              "", "## Emerging hotspots", "",
              f"Detected {em['detected']} of {em['hazards']} injected hazards"
              + (f" ({em['busy_detected']} of {em['busy_hazards_100_passes_2wk']} at junctions the fleet passed 100+ times in the two weeks after onset)"
                 if "busy_detected" in em else "")
              + f", median {em['median_delay_days']} days after onset; "
              f"{em['false_alarms_per_week']} false-alarm junction areas per week.",
              "", "## Impact (scored with the simulator's true rates)", "",
              f"On {im['trips_evaluated']:,} test trips, {im['rerouted_pct']}% were rerouted. True serious events fell "
              f"{im['true_serious_events_cut_pct_all_trips']}% across all trips "
              f"({im['true_serious_events_cut_pct_rerouted_trips']}% on rerouted trips) for "
              f"{im['extra_minutes_per_rerouted_trip']} extra minutes per rerouted trip "
              f"({im['extra_time_pct_all_trips']}% total travel time): "
              f"{im['serious_events_avoided_per_10k_trips']} serious events avoided per 10,000 trips.",
              "", "## Timings", ""]
    lines += [f"- {k}: {v}" for k, v in r["timings"].items()]
    return "\n".join(lines) + "\n"
