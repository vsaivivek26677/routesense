"""Column contracts for the tables every component reads and writes.

Keeping these in one place lets the data, model, routing and UI work proceed in
parallel: anything that produces a table validates it here.
"""
from __future__ import annotations

import pandas as pd

EVENT_TYPES = ("harsh_brake", "harsh_turn", "harsh_accel", "near_miss", "accident")
VEHICLE_TYPES = ("truck", "lcv", "two_wheeler")
DRIVER_COHORTS = ("novice", "regular", "experienced")

SEGMENTS = [
    "segment_id", "u", "v", "key", "name", "road_class", "length_m", "speed_kph",
    "travel_time_s", "lanes", "lit", "junction_degree", "max_turn_deg",
    "total_turn_deg", "sinuosity", "is_bridge", "oneway", "geometry",
]
EVENTS = [
    "event_id", "trip_id", "vehicle_id", "driver_id", "ts", "lat", "lon",
    "type", "severity", "speed_kmph", "segment_id",
]
TRIPS = [
    "trip_id", "vehicle_id", "driver_id", "vehicle_type", "driver_cohort",
    "depot_id", "origin_node", "dest_node", "route_variant", "depart_ts",
    "arrive_ts", "distance_m", "duration_s", "n_events", "n_near_miss", "n_accident",
]
TRIP_SEGMENTS = ["trip_id", "seq", "segment_id", "enter_ts"]
WEATHER = ["ts", "precip_mm", "rain_mm", "weather_code", "temperature_c", "cloud_cover"]

TABLES = {
    "segments": SEGMENTS,
    "events": EVENTS,
    "trips": TRIPS,
    "trip_segments": TRIP_SEGMENTS,
    "weather": WEATHER,
}


def validate(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Raise if ``df`` is missing any column of ``table``; return it unchanged."""
    missing = [c for c in TABLES[table] if c not in df.columns]
    if missing:
        raise ValueError(f"{table} table is missing columns: {missing}")
    if table == "events":
        bad = set(df["type"].unique()) - set(EVENT_TYPES)
        if bad:
            raise ValueError(f"unknown event types: {sorted(bad)}")
    return df
