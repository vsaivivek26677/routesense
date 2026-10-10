import numpy as np
import pytest

from routesense.evaluation import _lift_top, _poisson_deviance, run, to_markdown


def test_poisson_deviance_zero_for_perfect_fit():
    y = np.array([0.0, 2.0, 5.0])
    assert _poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)
    assert _poisson_deviance(y, np.full(3, y.mean())) > 0


def test_lift_top():
    rate = np.array([10.0, 1.0, 1.0, 1.0])
    km = np.ones(4)
    y = np.array([4.0, 0.0, 0.0, 0.0])
    assert _lift_top(rate, km, y, top=0.25) == pytest.approx(4.0)


@pytest.fixture(scope="module")
def report(city_segments, city_sim, city_weather):
    params, sim = city_sim
    return run(city_segments, sim, city_weather, params, split="2026-07-01", impact_trips=80)


def test_full_model_beats_baselines(report):
    sb = report["segment_backtest"]
    assert sb["distance_only"]["deviance_explained"] == pytest.approx(0.0, abs=1e-9)
    assert sb["full_model"]["deviance_explained"] > sb["features_only"]["deviance_explained"]
    assert sb["full_model"]["deviance_explained"] > 0
    assert report["truth_recovery"]["harsh"]["full_model"] > report["truth_recovery"]["harsh"]["features_only"]


def test_trip_level_and_impact(report):
    tb, im = report["trip_backtest"], report["impact"]
    assert 0.5 < tb["auc_full_model"] <= 1
    assert len(tb["calibration_by_decile"]["predicted"]) == len(tb["calibration_by_decile"]["observed"])
    assert im["trips_evaluated"] == 80
    assert im["extra_time_pct_all_trips"] >= 0
    assert im["true_serious_events_cut_pct_all_trips"] > -5  # not materially worse than fastest


def test_markdown_report(report):
    md = to_markdown(report)
    assert "Backtest" in md and "Impact" in md and "full model" in md
