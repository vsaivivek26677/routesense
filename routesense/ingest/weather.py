"""Hourly historical weather from the Open-Meteo archive API (free, no key)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from routesense.config import TIMEZONE
from routesense.schema import validate

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ["precipitation", "rain", "weather_code", "temperature_2m", "cloud_cover"]


def fetch_hourly_weather(
    lat: float, lon: float, start: str, end: str, cache_dir: Path
) -> pd.DataFrame:
    """Hourly weather for one point between ``start`` and ``end`` (YYYY-MM-DD, inclusive).

    Results are cached as parquet so the API is hit once per point and period.
    """
    path = Path(cache_dir) / f"weather_{lat:.3f}_{lon:.3f}_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": TIMEZONE,
        },
        timeout=60,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"]).rename(
        columns={
            "time": "ts",
            "precipitation": "precip_mm",
            "rain": "rain_mm",
            "temperature_2m": "temperature_c",
        }
    )
    df["ts"] = pd.to_datetime(df["ts"])
    df["precip_mm"] = df["precip_mm"].fillna(0.0)
    validate(df, "weather")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
