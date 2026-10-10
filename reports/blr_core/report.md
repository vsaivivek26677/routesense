# blr_core: real roads and weather, simulated fleet

Roads (OpenStreetMap): 129,261 directed segments, 7,474.8 km; lighting tagged on 3.6% of segments. 80.0% of segments were never driven by the simulated fleet; 85.3% have fewer than 30 fleet passes.

Weather (Open-Meteo), by month:

| month | precip_mm | wet_hours_pct |
|---|---|---|
| 2026-03 | 4.4 | 0.4 |
| 2026-04 | 2.3 | 0.3 |
| 2026-05 | 87.0 | 4.8 |
| 2026-06 | 87.8 | 6.9 |
| 2026-07 | 64.0 | 5.5 |
| 2026-08 | 147.2 | 11.7 |

## Simulator check against MoRTH 2022 (all-India, all traffic)

| Slot | Simulated fleet serious events | MoRTH accidents |
|---|---|---|
| 00-03 | 15.0% |  |
| 03-06 | 14.0% |  |
| 06-09 | 6.9% |  |
| 09-12 | 9.4% |  |
| 12-15 | 10.2% |  |
| 15-18 | 9.8% | 17.3% |
| 18-21 | 14.4% | 20.4% |
| 21-24 | 20.2% |  |

MoRTH shares mix traffic volume and risk for all vehicles; the fleet's hours differ (trucks drive at night).

## Backtest (train < 2026-07-01 <= test)

Data: 129,261 segments, 110,284 trips, 283,796 events; train 721,654 vehicle-km, test 367,086 vehicle-km.

## Segment level: serious events (near-miss + accident) in the test period

| Method | Deviance explained | Top-10% km lift |
|---|---|---|
| distance only | 0.000 | 0.96x |
| raw history | 0.058 | 2.17x |
| features only | -0.021 | 1.04x |
| full model | 0.115 | 2.16x |

## Agreement with the simulator's hidden risk (Spearman)

| Event group | Raw history | Features only | Full model |
|---|---|---|---|
| near miss | 0.25 | 0.264 | 0.352 |
| harsh | 0.678 | 0.349 | 0.735 |

## Trip level

37,114 test trips, 8.98% had a serious event. AUC: full model 0.675, distance only 0.631, duration only 0.574.

## Emerging hotspots

Detected 14 of 24 injected hazards (13 of 19 at junctions the fleet passed 100+ times in the two weeks after onset), median 10.5 days after onset; 1.35 false-alarm junction areas per week.

## Impact (scored with the simulator's true rates)

On 1,500 test trips, 67.1% were rerouted. True serious events fell 13.78% across all trips (20.13% on rerouted trips) for 2.56 extra minutes per rerouted trip (7.15% total travel time): 128.58 serious events avoided per 10,000 trips.

## Timings

- exposure_tables_s: 10.71
- model_fit_s: 32.85
- risk_table_s: 55.79
- emerging_replay_s: 1.79
- engine_build_s: 2.48
- route_options_ms_per_trip: 103.8
