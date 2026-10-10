# Netradyne Campus Innovation Challenge — Master Plan

**Theme:** Fleet Route Risk Intelligence
**Working name:** *RouteSense* (placeholder — pick a final name before the deck)
**Pitch:** *Know the risk before the trip.* Dispatchers see how risky each route is, **why**, **when** it is safest to leave, and what a safer option costs in minutes.

| Phase | Dates (IST) | Deliverable (assumed) |
|---|---|---|
| Pre-work | Fri Oct 9 – Sun Oct 11 | Data feasibility check, concept locked |
| Idea Phase | Mon Oct 12 – Sun Oct 18, 11:59 PM | PDF slide deck |
| Gap | Mon Oct 19 – Tue Oct 20 | Data pipeline + simulator done; risk engine started |
| Prototype Phase | Wed Oct 21 – Sat Oct 31, 11:59 PM | Working prototype + data we sourced/built ourselves |

---

## 1. How we win: building for the judging criteria

Every feature and slide should clearly earn points on at least one criterion.

| Criterion | What earns it |
|---|---|
| Theme Relevance / Business & Industry Relevance | All 8 tasks from the brief are covered; framed around the dispatcher's real workflow; a clear path to running on Netradyne's own event data |
| AI Usage | Clustering, a risk model that reports its uncertainty, detection of new hotspots, and an LLM driver briefing. Each is used where it adds value. |
| Technical Solution & Innovation / Technical Depth | Uncertainty-aware scoring, a departure-time advisor, backtesting against held-out data, synthetic data with a known ground truth |
| Implementation Feasibility & Impact | Runs on public data plus a documented simulator; impact estimated as incidents avoided per 10k trips |
| Demo & Presentation / Functionality & UX | One polished dispatcher flow, a map-first UI, and a 3-minute demo story |

**Strategy:** do the 8 required tasks *solidly*, add **5 differentiators** that most teams won't build, and be **rigorously honest** about the data.

---

## 2. Product concept

### Users
- **Dispatcher:** assigns trips and wants a safe route without missing delivery windows.
- **Driver:** wants to know what to watch out for on today's route.
- **Fleet safety manager:** wants trends, recurring hotspots and proof that the system reduces incidents.

### Core (the brief's 8 tasks)
1. Data preprocessing: telematics events, road data and weather, all mapped onto road segments
2. Hotspot detection: clustering weighted by severity and how recent each event is
3. Route risk score with a confidence band
4. Alternate routes with the risk-vs-time tradeoff
5. Map heatmap and highlighted high-risk segments
6. Dispatch screen that logs the chosen route and its risk score
7. Analytics: trends by route, region, driver cohort and time period; searchable hotspots
8. Evaluation: score accuracy against later incidents, and scalability

### Differentiators (what makes us unique)
| # | Feature | Why it matters |
|---|---|---|
| D1 | **Departure-time advisor:** "Leave at 5:30 instead of 6:15 → −30% risk, +4 min" | The brief only asks *which* route. We also answer *when*. Easy for dispatchers to act on, and a memorable demo moment. |
| D2 | **Explainable risk + AI pre-trip briefing:** each risky segment lists its reasons (unlit junction, 14 harsh brakes at night in the last 30 days, monsoon waterlogging). An LLM turns these into a short driver briefing in **English / Hindi / Kannada**. | Scores are trusted only when they are explained. Strong on "AI Usage". Indian context. |
| D3 | **Emerging hotspot alerts:** detect segments whose event rate is *suddenly* rising (new construction, potholes, waterlogging) days before they become accident sites. | Moves from looking back at past incidents to warning ahead of new ones. Can be shown against hazards we deliberately inject into the simulator. |
| D4 | **Uncertainty-aware routing:** quiet roads aren't called "safe" just because data is thin. Sparse segments are pulled toward a baseline estimate, flagged, and the router can avoid uncertainty if the dispatcher chooses. | Covers task 3 properly. Most teams will skip it. |
| D5 | **Impact calculator:** expected incidents avoided per 10k trips, converted into cost and lives using published Indian road-accident statistics. | Turns the "Impact" criterion into a number. |

**Stretch goals (only if the core and D1–D5 are done by Oct 29):**
- S1: Different risk profiles per vehicle type (truck / light commercial vehicle / two-wheeler delivery)
- S2: Satellite night-lights (VIIRS) data to estimate street lighting where OpenStreetMap has no tag, plus SRTM elevation for steep gradients
- S3: Matching drivers to routes (send experienced drivers on high-risk routes)

---

## 3. Data strategy (Phase 2: we source everything)

### Real public data
| Data | Source | Used for |
|---|---|---|
| Road network: road type, junctions, curves, speed limits, `lit` tags | OpenStreetMap via **OSMnx** (Bengaluru, possibly also a highway corridor such as Bengaluru–Mysuru) | Segment features, routing graph |
| Historical and current weather (rain, visibility) | **Open-Meteo** (free, no key) | Weather adjustment to risk, monsoon effect |
| Aggregate accident statistics: time-of-day share, weather share, vehicle type | MoRTH *Road Accidents in India* annual report, data.gov.in | **Calibrating** the simulator; impact numbers |
| Known accident black spots | MoRTH / state lists (check what is available) | Seeding and sanity-checking hotspots |
| *(stretch)* Night lights, elevation | NASA VIIRS Black Marble, SRTM | Lighting and gradient features |

### Simulated fleet telematics (no public dataset exists)
Build a **fleet simulator** instead of random noise:
1. Depots and destinations: real warehouses and industrial areas from OSM.
2. Simulate N vehicles × 6 months of trips over real road routes with realistic departure times.
3. **Hidden ground-truth risk** for each segment = f(road geometry, lighting, time, weather, vehicle, driver cohort) × a hidden per-segment factor the model can't see directly. Events (harsh brake/turn/accel, near-miss, accident) are drawn from it, with severity.
4. Inject **emerging hazards** (e.g., a construction zone from Aug 10) to test D3.
5. **Calibrate** totals to the published MoRTH figures (night share, monsoon increase, and so on).

**Why this is a strength, not a weakness:** because we know the ground truth, we can *measure* how well each component recovers it. Real data can't do that. We document this openly in a "Data provenance" section, and the same pipeline accepts real telematics in the same schema.

### Data schema (fixed early so everyone can work in parallel)
- `events(event_id, vehicle_id, driver_id, ts, lat, lon, type, severity, speed_kmph, segment_id)`
- `trips(trip_id, vehicle_id, driver_id, depart_ts, route_segment_ids, duration_s, incident_flag)`
- `segments(segment_id, geometry, road_class, length_m, curvature, junction_degree, lit, speed_limit, ...)`
- `weather(ts, cell_id, precip_mm, visibility_m, ...)`

---

## 4. Technical architecture

```
 OSM ─┐                                 ┌─► Hotspot clusters ─────────┐
 Weather ─┼─► Ingest + snap events to ──┼─► Segment risk model ───────┼─► Route scoring + alternatives ─► API ─► Dispatcher UI
 Simulator ┘   road segments (task 1)   └─► New-hotspot detector (D3) ┘   + departure advisor (D1)            ├─ Map heatmap (task 5)
                                                                           + explanations / LLM brief (D2)     ├─ Analytics (task 7)
                                                                                                               └─ Trip log (task 6)
```

| Component | Method |
|---|---|
| Hotspots | HDBSCAN on event locations; each event weighted by severity × a recency decay (older events count less) |
| Segment risk | Poisson model / LightGBM-Poisson on segment features + time-of-day + weather, giving expected events per vehicle-km. **Gamma-Poisson shrinkage** pulls sparse segments toward the model's prediction and gives the confidence band. |
| Route score | Sum of segment risk weighted by distance along the route; uncertainty combined across segments; reported both total and per km |
| Alternatives | k-shortest paths plus routes searched with a combined cost of time + λ·risk over several λ values. Keep only routes that no other option beats on both time and risk. |
| Departure advisor | Re-score the chosen routes across a ±2 h window in 15-min steps and plot risk against departure time |
| New-hotspot detection | Per-segment rolling event rate compared with its long-run baseline (Poisson CUSUM / space-time scan) |
| Explanations | Per-segment breakdown of which features drive the score (SHAP or the model's coefficients) → templated reasons → LLM briefing (falls back to the template if the API is unavailable) |
| Backend | Python, FastAPI, GeoPandas, NetworkX/OSMnx, SQLite (Postgres/PostGIS if time allows) |
| Frontend | React + Leaflet/deck.gl (or Streamlit + pydeck if short on time) |
| Hosting | Render / Railway / Streamlit Community Cloud for a live demo link |

---

## 5. Evaluation plan (task 8 + credibility)

- **Backtest over time:** train on months 1–4, score routes and segments for months 5–6.
- **Metrics:**
  - How well risky segments are ranked: AUC / PR-AUC
  - **Share of incidents captured in the top 10% riskiest segments**
  - Whether predicted rates match observed ones (calibration plot)
  - Correlation between route scores and the incidents that actually happened on those routes
- **Baselines to beat:** (a) distance only, (b) raw event counts with no model, (c) no recency weighting.
- **Ground-truth recovery:** correlation between our segment risk and the simulator's hidden true risk.
- **D3:** how many days after an injected hazard starts we flag it, and how many false alarms.
- **Scalability:** scoring time per route and pipeline time as events grow from 100k → 1M → 10M; results presented as a chart.

---

## 6. Phase 1: Idea-phase PDF deck (Oct 12–18)

### Slide outline (~14 slides)
1. **Title**: name, tagline, team
2. **The problem**: India's road-safety scale (real MoRTH numbers); dispatch by distance/time ignores risk
3. **Who feels it**: dispatcher, driver and safety manager personas, with a "today vs with us" contrast
4. **Our solution in one picture**: dispatcher screen mockup (route options + heatmap)
5. **How it works**: architecture diagram
6. **Data & provenance**: real public data + calibrated simulator with ground truth; schema
7. **Risk engine**: hotspots → segment model → shrinkage → route score with a confidence band
8. **Safer alternatives + departure-time advisor**: route tradeoff chart and risk-by-departure-time chart mockups
9. **What makes us different**: D1–D5
10. **Where AI is used**: each model and why it is the right tool
11. **Evaluation plan**: metrics, baselines, backtest design
12. **Impact hypothesis**: e.g., "X% fewer high-severity events for ≤Y% extra travel time". Give the method, not invented numbers.
13. **Feasibility**: tech stack, 11-day build plan, team roles, risks and mitigations
14. **Deployment plan & roadmap**: pilot → plug in Netradyne's camera-based event data → live rerouting

### Schedule
| Date | Work |
|---|---|
| Oct 9–11 (pre-work) | Lock concept and name. **Data feasibility check:** load the Bengaluru graph with OSMnx, pull Open-Meteo data, find MoRTH statistics. Agree on the schema. |
| Oct 12–13 | Write the content for every slide (text first). Collect real statistics with citations. |
| Oct 14–15 | Architecture diagram, UI mockups (Figma), chart mockups. Optional: a screenshot of a real Bengaluru risk heatmap from the feasibility check. That's a strong signal of feasibility. |
| Oct 16 | Design pass: consistent template, one message per slide |
| Oct 17 | Full review against the Idea-phase criteria; cut text; check every number has a source |
| Oct 18 | Export the PDF, final check, **submit before evening** (deadline 11:59 PM) |

---

## 7. Phase 2: Prototype build plan (Oct 19–31)

| Date | Milestone |
|---|---|
| Oct 19–20 | Hotspots + baseline segment risk on the real Bengaluru data (ingestion and simulator are built early, see below) |
| Oct 21 (D1) | Simulator calibrated against MoRTH figures; segment risk model started |
| Oct 22 (D2) | Hotspot clustering + baseline segment risk; first heatmap on the map |
| Oct 23 (D3) | Segment risk model + shrinkage + confidence bands |
| Oct 24 (D4) | Route scoring + alternatives + tradeoff; API endpoints |
| Oct 25 (D5) | Dispatcher UI: origin/destination → route options → choose → logged. **End-to-end demo works.** |
| Oct 26 (D6) | D1 departure advisor + D2 explanations + LLM briefing |
| Oct 27 (D7) | D3 new-hotspot detection + alerts panel; analytics dashboard (task 7) |
| Oct 28 (D8) | Evaluation: backtest, baselines, scalability benchmark; D5 impact calculator |
| Oct 29 (D9) | **Feature freeze.** Bug fixes, UX polish, deploy the live demo, README |
| Oct 30 (D10) | Record the demo video, final deck with real results, rehearse the pitch |
| Oct 31 (D11) | Buffer → **submit by afternoon** |

### Demo script (≈3 min)
1. A dispatcher plans a Bengaluru trip at 6:15 PM in the rain. The fastest route scores **high risk**.
2. The heatmap shows why: an unlit junction cluster and a waterlogging-prone underpass.
3. An alternative is **+5 min, −40% risk**. The advisor says leaving at 5:45 also cuts risk.
4. One click → the driver gets a Kannada briefing listing the 3 hazards to watch.
5. The safety manager's view shows an **emerging hotspot** alert at a new construction zone.
6. Close with the evaluation results and the impact estimate.

---

## 8. Team roles (4 people)

| Role | Owns |
|---|---|
| **Data engineer** | OSM/weather ingestion, simulator, calibration, schema |
| **ML lead** | Hotspots, segment model, shrinkage, D3 new-hotspot detection, evaluation |
| **Backend / routing** | Graph routing, alternatives, D1 advisor, API, trip logging, deployment |
| **Frontend + story** | Map UI, dispatcher flow, analytics, D2 briefing UI, deck, demo video |

Everyone reviews the deck. Hold a short daily check-in during Phase 2.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| "Your data is fake" | A calibrated simulator with ground truth, a provenance slide, and a pipeline built to accept real data |
| Full Bengaluru graph is too slow | Limit to a bounding box or corridor; cache graph and precomputed risk |
| LLM API key, cost or availability | Fall back to templated text; cache briefings |
| Scope creep | D1–D5 only; stretch goals after the Oct 29 freeze, otherwise drop them |
| Demo breaks live | Record the video by Oct 30; seed the demo with saved scenarios |
| Team-size rule (eligibility says teams of 4) | Confirm with the organizers now |

---

## 10. Open questions

- [ ] Confirm the team-size rule (1–4 vs "teams of 4 only")
- [ ] Confirm the Idea-phase format (assumed: PDF deck) and any slide or page limit
- [x] Is building before Oct 21 allowed? **Assumed yes.** If not, move the code to a fresh repo when the Prototype Phase starts.
- [ ] Phase 2 deliverables: repo link? video? hosted demo? deck?
- [ ] Final product name

## 11. Progress log

| Date | Done |
|---|---|
| Oct 9 | Plan. Project skeleton, table contracts, OSM segment features, Open-Meteo loader, fleet simulator (ground truth + emerging hazards), 16 tests. Sim at Bengaluru-core scale: ~110k trips / 1.4M vehicle-km / 335k events in ~30 s. |
| Oct 9 | All 8 brief tasks + D1–D5 built and tested on a synthetic dev city (network blocked here): hotspots, risk model with confidence, routing + alternatives + departure advisor, emerging alerts, explanations + LLM briefing, backtest/impact evaluation, analytics, FastAPI service, dispatcher web UI. 62 tests. |
| Oct 10 | First run on real Bengaluru data (OSM roads + Open-Meteo weather, downloaded on a laptop). Fixed what real roads exposed (junction zones, urban speeds, cross-fitting, sparse flag). Results: 13.8% fewer serious events for +2.6 min per rerouted trip; 14 of 24 new hazards detected. |

## 12. Proposed repo layout (Phase 2)

```
routesense/
  PLAN.md
  data/            # raw + processed (gitignored if large)
  sim/             # fleet simulator
  pipeline/        # ingest, snapping, features
  models/          # hotspots, segment risk, emerging hotspots
  routing/         # scoring, alternatives, departure advisor
  api/             # FastAPI app
  web/             # frontend
  eval/            # backtests, benchmarks, notebooks
  deck/            # idea + final decks
```
