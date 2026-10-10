"""Published reference figures used to sanity-check the simulator.

MoRTH, "Road Accidents in India 2022" (released Oct 2023), as reported by PIB and
Business Standard. These are shares of ALL reported road accidents in India, so they
mix traffic volume with risk; a delivery fleet's mix of hours differs from all traffic.
"""
MORTH_2022 = {
    "source": "MoRTH, Road Accidents in India 2022",
    "total_accidents": 461_312,
    "accident_share_by_slot_pct": {"18-21": 20.4, "15-18": 17.3},
    "lowest_slots": "00-06",
    "adverse_weather_share_pct": 16.6,  # rain, fog, hail etc. combined
}
