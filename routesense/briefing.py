"""Pre-trip driver briefing: the few hazards to watch on the chosen route.

The hazard list is always computed by our own models. An open-weight LLM on Groq's free API
only turns it into a short briefing in the driver's language; without GROQ_API_KEY (or on any
API error, including the free tier's rate limits) a deterministic English template is used.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from functools import lru_cache

import requests

from routesense.models.risk import band_of_hour
from routesense.routing.engine import Conditions

log = logging.getLogger(__name__)

LANGUAGES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Any Groq chat model id works; gpt-oss-120b is on Groq's free tier (about 1,000 requests a day).
MODEL = os.environ.get("ROUTESENSE_LLM_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You write pre-trip safety briefings for commercial drivers in Bengaluru, India.

You receive a JSON description of one planned trip: its overall risk and a short list of
hazards our risk model found along the route, each with its road name, distance from the
start and the reasons it is risky.

Write the briefing in {language}:
- First line: one sentence on the trip (departure time, overall risk level).
- Then one short bullet per hazard, in route order: where it is (road name and roughly how
  many km into the trip) and one concrete thing the driver should do there.
- Last line: one short reminder that fits the conditions (rain, night, vehicle type).

Use only the hazards and facts in the JSON; do not add places, numbers or dangers that are
not there. Keep road names as given. Use simple words a driver understands. Under 120 words,
plain text, no headings."""


def route_hazards(option: dict, cond: Conditions, explainer, limit: int = 4) -> list[dict]:
    """Top hazards on a scored route option, in route order, with reasons."""
    band = int(band_of_hour(cond.depart.hour))
    top = sorted(option["top_segments"][:limit], key=lambda t: t["km_from_start"])
    hazards = []
    for t in top:
        reasons = explainer.reasons(t["segment_id"], band, int(cond.rain), cond.vehicle, limit=3)
        hazards.append({
            "road": t["name"] or "unnamed road",
            "km_from_start": round(t["km_from_start"], 1),
            "share_of_route_risk_pct": round(100 * t["share"]),
            "reasons": [r["text"] for r in reasons],
            "lat": t["lat"],
            "lon": t["lon"],
        })
    return hazards


def _trip_facts(option: dict, cond: Conditions, hazards: list[dict]) -> dict:
    return {
        "departure": cond.depart.strftime("%H:%M"),
        "vehicle": cond.vehicle_type.replace("_", " "),
        "raining": cond.rain,
        "distance_km": round(option["distance_km"], 1),
        "duration_min": round(option["duration_s"] / 60),
        "risk_level": option["level"],
        "hazards": [{k: h[k] for k in ("road", "km_from_start", "reasons")} for h in hazards],
    }


def template_briefing(option: dict, cond: Conditions, hazards: list[dict]) -> str:
    lines = [
        f"Trip at {cond.depart:%H:%M}: {option['distance_km']:.1f} km, about "
        f"{option['duration_s'] / 60:.0f} min, {option['level']} risk."
    ]
    for h in hazards:
        why = "; ".join(h["reasons"][:2]) or "higher incident rate"
        lines.append(f"- km {h['km_from_start']:.1f}, {h['road']}: {why}. Slow down and keep a safe gap.")
    tips = []
    if cond.rain:
        tips.append("roads are wet, brake early")
    if cond.depart.hour >= 19 or cond.depart.hour < 6:
        tips.append("it is dark, use headlights and watch for pedestrians")
    if cond.vehicle_type == "truck":
        tips.append("take wide turns slowly")
    lines.append(("Remember: " + ", ".join(tips) + ".") if tips else "Drive safe.")
    return "\n".join(lines)


def _llm_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


# LLM requests made today; ROUTESENSE_LLM_DAILY_LIMIT caps them so a public demo link stays
# inside the provider's free quota (unset = no cap). Over the cap, the template briefing is used.
_usage = {"day": None, "calls": 0}


def _within_daily_limit() -> bool:
    limit = os.environ.get("ROUTESENSE_LLM_DAILY_LIMIT")
    if not limit:
        return True
    if _usage["day"] != date.today():
        _usage.update(day=date.today(), calls=0)
    return _usage["calls"] < int(limit)


@lru_cache(maxsize=256)
def _llm_text(facts_json: str, language: str) -> str:
    """One chat completion on Groq. Raises on any failure, so failures are not cached."""
    _usage["calls"] += 1
    body = {
        "model": MODEL,
        "temperature": 0.3,
        "max_completion_tokens": 2000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(language=LANGUAGES[language])},
            {"role": "user", "content": facts_json},
        ],
    }
    if "gpt-oss" in MODEL:  # reasoning model (its reasoning comes back outside `content`): keep it short
        body["reasoning_effort"] = "low"
    response = requests.post(GROQ_URL, json=body, timeout=30,
                             headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"] or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # models that inline reasoning
    if not text:
        raise ValueError("empty completion")
    return text


def _llm(facts_json: str, language: str) -> str | None:
    try:
        return _llm_text(facts_json, language)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
        log.warning("briefing request failed: %s", e)
        return None


def briefing(option: dict, cond: Conditions, explainer, language: str = "en") -> dict:
    """Briefing text for a route option, written by the LLM when a Groq key is set."""
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
    hazards = route_hazards(option, cond, explainer)
    if _llm_available() and _within_daily_limit():
        facts = json.dumps(_trip_facts(option, cond, hazards), ensure_ascii=False, sort_keys=True)
        text = _llm(facts, language)
        if text:
            return {"text": text, "language": language, "source": "llm", "hazards": hazards}
    note = "" if language == "en" else f" ({LANGUAGES[language]} needs the AI briefing; showing English.)"
    return {"text": template_briefing(option, cond, hazards) + note, "language": "en",
            "source": "template", "hazards": hazards}
