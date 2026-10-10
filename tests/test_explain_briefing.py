from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

import routesense.briefing as briefing_mod
from routesense.briefing import briefing
from routesense.explain import Explainer, _plural
from routesense.models.risk import BAND_NAMES
from routesense.routing.engine import Conditions, RouteEngine, congestion_profile, fleet_level_thresholds

COND = Conditions(pd.Timestamp("2026-08-20 20:15"), rain=True, vehicle_type="truck")


@pytest.fixture(scope="module")
def setup(city_segments, city_sim, city_model):
    _, sim = city_sim
    ex, model, table = city_model
    eng = RouteEngine(city_segments, table, congestion_profile(sim.trips, sim.trip_segments, city_segments),
                      level_thresholds=fleet_level_thresholds(table, ex))
    w, s, e, n = city_segments.total_bounds
    o, d = eng.nearest_node(s + 0.1 * (n - s), w + 0.1 * (e - w)), eng.nearest_node(n - 0.1 * (n - s), e - 0.1 * (e - w))
    return eng, Explainer(city_segments, model, table), eng.options(o, d, COND)[0]


def test_plural():
    assert _plural(1, "accident") == "1 accident" and _plural(2, "near-miss") == "2 near-misses"


def test_reasons_name_junction_rain_and_time(city_segments, setup):
    _, xp, _ = setup
    sid = int(np.flatnonzero(city_segments["junction_degree"].to_numpy() >= 4)[0])
    band = BAND_NAMES.index("18-21")
    codes = {r["code"] for r in xp.reasons(sid, band, rain=1, vehicle=0, limit=10)}
    assert {"junction", "rain", "time", "vehicle"} & codes
    dry_codes = {r["code"] for r in xp.reasons(sid, band, rain=0, vehicle=1, limit=10)}
    assert "rain" not in dry_codes and "vehicle" not in dry_codes


def test_reason_effects_are_sorted(setup):
    _, xp, option = setup
    sid = option["top_segments"][0]["segment_id"]
    effects = [r["effect_pct"] for r in xp.reasons(sid, 6, 1, 0, limit=10) if r["code"] not in ("emerging", "hotspot", "sparse")]
    assert effects == sorted(effects, reverse=True)


def test_template_briefing_without_key(setup, monkeypatch):
    _, xp, option = setup
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    b = briefing(option, COND, xp)
    assert b["source"] == "template" and b["hazards"]
    for h in b["hazards"]:
        assert f"km {h['km_from_start']:.1f}" in b["text"]
    assert "wet" in b["text"]
    hi = briefing(option, COND, xp, language="hi")
    assert hi["language"] == "en" and "Hindi" in hi["text"]
    with pytest.raises(ValueError):
        briefing(option, COND, xp, language="fr")


class _FakePost:
    """Stands in for requests.post against Groq's chat completions endpoint."""

    def __init__(self, status=200, content="<think>plan</think>ಸುರಕ್ಷಿತವಾಗಿ ಚಲಾಯಿಸಿ"):
        self.calls, self.status, self.content = [], status, content

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})

        def raise_for_status():
            if self.status >= 400:
                raise requests.HTTPError(f"{self.status} Too Many Requests")
        return SimpleNamespace(raise_for_status=raise_for_status,
                               json=lambda: {"choices": [{"message": {"content": self.content}}]})


@pytest.mark.parametrize("status,source", [(200, "llm"), (429, "template")])
def test_llm_briefing_request(setup, monkeypatch, status, source):
    _, xp, option = setup
    fake = _FakePost(status)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(briefing_mod.requests, "post", fake)
    briefing_mod._llm_text.cache_clear()
    b = briefing(option, COND, xp, language="kn")
    assert b["source"] == source
    if status == 200:
        assert b["text"] == "ಸುರಕ್ಷಿತವಾಗಿ ಚಲಾಯಿಸಿ" and b["language"] == "kn"
    call = fake.calls[0]
    assert call["url"] == briefing_mod.GROQ_URL and call["json"]["model"] == briefing_mod.MODEL
    assert call["headers"]["Authorization"] == "Bearer test"
    system, user = call["json"]["messages"]
    assert "Kannada" in system["content"] and "hazards" in user["content"]
    briefing(option, COND, xp, language="kn")
    assert len(fake.calls) == (1 if status == 200 else 2)  # successes are cached, failures retried
    briefing_mod._llm_text.cache_clear()


def test_daily_limit_falls_back_to_template(setup, monkeypatch):
    _, xp, option = setup
    fake = _FakePost()
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("ROUTESENSE_LLM_DAILY_LIMIT", "1")
    monkeypatch.setattr(briefing_mod, "_usage", {"day": None, "calls": 0})
    monkeypatch.setattr(briefing_mod.requests, "post", fake)
    briefing_mod._llm_text.cache_clear()
    assert briefing(option, COND, xp, language="kn")["source"] == "llm"
    assert briefing(option, COND, xp, language="hi")["source"] == "template"
    assert len(fake.calls) == 1
    briefing_mod._llm_text.cache_clear()
