"""Parameters of the fleet simulator.

Every number here is a modelling ASSUMPTION, not a measured fact. They are chosen
to be plausible and to give enough events for evaluation, and should be tuned so
that the aggregates printed by ``calibration_report`` match published figures
(MoRTH "Road Accidents in India": time-of-day and weather shares, vehicle mix).

Accident rates are deliberately inflated versus reality so that a few months of
simulated fleet data contains enough accidents to evaluate against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def hourly(default: float, **ranges: float) -> np.ndarray:
    """24-value profile; keyword ``h<start>_<end>`` sets hours start..end-1 (wraps midnight)."""
    out = np.full(24, default, dtype=float)
    for name, value in ranges.items():
        start, end = (int(x) for x in name[1:].split("_"))
        hours = range(start, end) if start < end else [*range(start, 24), *range(0, end)]
        out[list(hours)] = value
    return out


@dataclass
class SimParams:
    seed: int = 7
    start_date: str = "2026-03-01"  # covers pre-monsoon and the Jun-Aug monsoon
    end_date: str = "2026-08-31"

    # Fleet
    n_vehicles: int = 120
    n_depots: int = 6
    n_destinations: int = 400
    deliveries_per_day: float = 2.5  # each delivery = outbound + return trip
    vehicle_mix: dict = field(
        default_factory=lambda: {"truck": 0.3, "lcv": 0.5, "two_wheeler": 0.2}
    )
    driver_mix: dict = field(
        default_factory=lambda: {"novice": 0.25, "regular": 0.5, "experienced": 0.25}
    )
    # Departure-hour weights by vehicle type (trucks skew to night, as heavy-vehicle
    # city-entry restrictions push them there).
    departure_profile: dict = field(
        default_factory=lambda: {
            "truck": hourly(0.3, h22_6=2.0, h6_9=1.0),
            "lcv": hourly(0.1, h7_20=1.5, h20_23=0.6),
            "two_wheeler": hourly(0.2, h9_15=1.2, h18_23=1.8, h15_18=0.8),
        }
    )
    dwell_minutes: tuple = (15, 45)

    # Routing: variant 0 is the fastest route; others use perturbed travel times so
    # drivers do not all take identical paths.
    route_variants: int = 4
    p_fastest: float = 0.7
    route_perturbation: tuple = (0.8, 1.5)
    congestion: np.ndarray = field(
        default_factory=lambda: hourly(1.1, h8_11=1.6, h11_17=1.3, h17_21=1.8, h22_6=0.9)
    )

    # Base event rates per vehicle-km before multipliers.
    base_rate_per_km: dict = field(
        default_factory=lambda: {
            "harsh_brake": 0.08,
            "harsh_turn": 0.03,
            "harsh_accel": 0.03,
            "near_miss": 0.004,
            "accident": 0.00004,
        }
    )
    road_class_mult: dict = field(
        default_factory=lambda: {
            "motorway": 1.2, "trunk": 1.3, "primary": 1.2, "secondary": 1.1,
            "tertiary": 1.0, "unclassified": 1.0, "residential": 0.7,
            "living_street": 0.5, "service": 0.6, "other": 0.8,
        }
    )
    junction_mult: dict = field(  # segments touching a junction of 4+ arms
        default_factory=lambda: {"harsh_brake": 1.6, "near_miss": 1.5, "accident": 1.4}
    )
    sharp_turn_deg: float = 45.0
    curve_mult: dict = field(
        default_factory=lambda: {"harsh_turn": 2.0, "near_miss": 1.2, "accident": 1.3}
    )
    unlit_night_mult: dict = field(
        default_factory=lambda: {"harsh_brake": 1.2, "near_miss": 1.4, "accident": 1.6}
    )
    night_hours: tuple = (20, 6)  # 20:00-05:59
    # Probability an untagged road is actually lit, by class.
    latent_lit_prob: dict = field(
        default_factory=lambda: {
            "motorway": 0.9, "trunk": 0.85, "primary": 0.8, "secondary": 0.65,
            "tertiary": 0.5, "unclassified": 0.35, "residential": 0.4,
            "living_street": 0.4, "service": 0.25, "other": 0.3,
        }
    )
    # Unobserved per-segment hazard (sight lines, encroachments, potholes ...).
    hidden_sigma: float = 0.5
    black_spot_frac: float = 0.01
    black_spot_mult: dict = field(
        default_factory=lambda: {"harsh_brake": 2.5, "near_miss": 4.0, "accident": 4.0}
    )

    tod_mult: dict = field(
        default_factory=lambda: {
            "harsh_brake": hourly(1.0, h8_11=1.5, h17_21=1.5, h22_6=1.1),
            "harsh_turn": hourly(1.0, h22_6=1.1),
            "harsh_accel": hourly(1.0, h22_6=1.3),
            "near_miss": hourly(0.9, h17_21=1.2, h21_22=1.3, h22_6=1.6, h6_9=1.0),
            "accident": hourly(0.9, h17_21=1.2, h21_22=1.3, h22_6=1.6, h6_9=1.0),
        }
    )
    # Multiplier = 1 + coef * min(precip_mm, 10) / 10
    rain_coef: dict = field(
        default_factory=lambda: {
            "harsh_brake": 0.4, "harsh_turn": 0.2, "harsh_accel": 0.0,
            "near_miss": 0.6, "accident": 0.8,
        }
    )
    vehicle_mult: dict = field(
        default_factory=lambda: {
            "truck": {"harsh_brake": 1.2, "harsh_turn": 1.4, "harsh_accel": 0.6,
                      "near_miss": 1.2, "accident": 1.3},
            "lcv": {},
            "two_wheeler": {"harsh_brake": 1.3, "harsh_turn": 1.2, "harsh_accel": 1.5,
                            "near_miss": 1.5, "accident": 1.8},
        }
    )
    driver_mult: dict = field(
        default_factory=lambda: {
            "novice": {"harsh_brake": 1.5, "harsh_turn": 1.5, "harsh_accel": 1.5,
                       "near_miss": 1.5, "accident": 1.6},
            "regular": {},
            "experienced": {"harsh_brake": 0.7, "harsh_turn": 0.7, "harsh_accel": 0.7,
                            "near_miss": 0.7, "accident": 0.7},
        }
    )

    # Emerging hazards (construction, potholes, waterlogging) that switch on mid-period:
    # ground truth for testing new-hotspot detection.
    n_emerging: int = 24
    emerging_window_days: int = 60  # they start within the last N days
    emerging_radius_m: float = 75.0  # road within this distance of the junction is affected
    emerging_mult: dict = field(
        default_factory=lambda: {"harsh_brake": 4.0, "harsh_turn": 2.0,
                                 "near_miss": 4.0, "accident": 3.0}
    )

    severity: dict = field(  # (levels, probabilities) on a 1-5 scale
        default_factory=lambda: {
            "harsh_brake": ([1, 2, 3], [0.6, 0.3, 0.1]),
            "harsh_turn": ([1, 2, 3], [0.6, 0.3, 0.1]),
            "harsh_accel": ([1, 2, 3], [0.7, 0.25, 0.05]),
            "near_miss": ([3, 4], [0.7, 0.3]),
            "accident": ([4, 5], [0.75, 0.25]),
        }
    )
