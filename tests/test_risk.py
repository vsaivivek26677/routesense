import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from routesense.models.risk import (
    GROUPS, REFERENCE, RiskModel, band_of_hour, exposure_table, load_table, save_table,
    segment_features,
)


@pytest.fixture(scope="module")
def fitted(city_model):
    ex, model, _ = city_model
    return ex, model


def test_band_of_hour():
    assert band_of_hour(2) == 0 and band_of_hour(3) == 1 and band_of_hour(23) == 7
    assert list(band_of_hour([0, 12, 18])) == [0, 4, 6]


def test_exposure_table_conserves_totals(city_segments, city_sim, fitted):
    _, sim = city_sim
    ex, _ = fitted
    km = city_segments["length_m"].to_numpy()[sim.trip_segments["segment_id"].to_numpy()].sum() / 1000
    assert ex["exposure_km"].sum() == pytest.approx(km)
    assert ex[[f"n_{g}" for g in GROUPS]].to_numpy().sum() == len(sim.events)


def test_exposure_table_time_window(city_segments, city_sim, city_weather):
    _, sim = city_sim
    cut = pd.Timestamp("2026-06-01")
    before = exposure_table(city_segments, sim.trips, sim.trip_segments, sim.events, city_weather, end=cut)
    after = exposure_table(city_segments, sim.trips, sim.trip_segments, sim.events, city_weather, start=cut)
    total = exposure_table(city_segments, sim.trips, sim.trip_segments, sim.events, city_weather)
    assert before["exposure_km"].sum() + after["exposure_km"].sum() == pytest.approx(total["exposure_km"].sum())


def test_model_recovers_true_risk_better_than_features_alone(city_segments, city_sim, fitted):
    _, sim = city_sim
    _, model = fitted
    table = model.risk_table(city_segments)
    used = model.exposure_km > 5
    b, r, v = REFERENCE
    truth = sim.truth["true_mult_harsh_brake"].to_numpy()[used]
    full = spearmanr(table.group_mean["harsh"][used, b, r, v], truth)[0]
    prior = spearmanr(model.prior_rate("harsh", segment_features(city_segments), b, r, v)[used], truth)[0]
    assert full > 0.5 and full > prior + 0.1


def test_rain_raises_risk(city_segments, fitted):
    _, model = fitted
    table = model.risk_table(city_segments)
    driven = model.exposure_km > 0
    assert table.mean[driven][:, :, 1, :].mean() > table.mean[driven][:, :, 0, :].mean()
    assert table.group_mean["harsh"][driven][:, :, 1, :].mean() > table.group_mean["harsh"][driven][:, :, 0, :].mean()


def test_never_driven_segments_rest_on_prior(city_segments, fitted):
    _, model = fitted
    table = model.risk_table(city_segments)
    never = model.exposure_km == 0
    assert never.any()
    assert table.data_sparse[never].all()
    for g in GROUPS:
        assert np.allclose(model.theta_mean[g][never], 1.0)
    assert (table.var > 0).all() and (table.mean > 0).all()


def test_save_load(city_segments, fitted, tmp_path):
    _, model = fitted
    table = model.risk_table(city_segments)
    model.save(tmp_path / "model.joblib")
    save_table(table, tmp_path / "table.npz")
    again = RiskModel.load(tmp_path / "model.joblib").risk_table(city_segments)
    loaded = load_table(tmp_path / "table.npz")
    np.testing.assert_allclose(again.mean, table.mean)
    np.testing.assert_array_equal(loaded.data_sparse, table.data_sparse)


def test_group_without_events_falls_back_to_constant(city_segments, fitted):
    from routesense.models.risk import ConstantRate

    ex, _ = fitted
    quiet = ex.assign(n_accident=0)
    model = RiskModel.fit(city_segments, quiet)
    assert isinstance(model.models["accident"], ConstantRate)
    table = model.risk_table(city_segments)
    assert np.isfinite(table.mean).all() and (table.group_mean["accident"] > 0).all()
