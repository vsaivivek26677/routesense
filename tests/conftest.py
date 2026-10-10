"""Synthetic test fixtures. These stand in for OSM/Open-Meteo only inside tests."""
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from routesense.ingest.osm import build_segments


def grid_graph(n: int = 8, spacing_deg: float = 0.003) -> nx.MultiDiGraph:
    """n x n two-way street grid near Bengaluru in OSMnx's graph format."""
    G = nx.MultiDiGraph(crs="epsg:4326")
    lon0, lat0 = 77.60, 12.95
    for i in range(n):
        for j in range(n):
            G.add_node(i * n + j, x=lon0 + j * spacing_deg, y=lat0 + i * spacing_deg)

    def add(a, b, highway, lit, geometry=None):
        ax, ay, bx, by = G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[b]["x"], G.nodes[b]["y"]
        line = geometry or LineString([(ax, ay), (bx, by)])
        length = line.length * 111_000
        speed = 50.0 if highway == "primary" else 25.0
        for s, t, geom in ((a, b, line), (b, a, LineString(line.coords[::-1]))):
            attrs = dict(highway=highway, length=length, speed_kph=speed,
                         travel_time=length / (speed / 3.6), oneway=False)
            if lit is not None:
                attrs["lit"] = lit
            if geometry is not None:
                attrs["geometry"] = geom
            G.add_edge(s, t, **attrs)

    for i in range(n):
        for j in range(n):
            node = i * n + j
            if j + 1 < n:  # east-west: every 3rd row is a lit primary road
                add(node, node + 1, "primary" if i % 3 == 0 else "residential",
                    "yes" if i % 3 == 0 else None)
            if i + 1 < n:
                add(node, node + n, "residential", "no" if j == 0 else None)

    # One curved road across a block diagonal.
    a, b = 0, n + 1
    ax, ay, bx, by = G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[b]["x"], G.nodes[b]["y"]
    add(a, b, "secondary", None, LineString([(ax, ay), (ax, by), (bx, by)]))
    return G


@pytest.fixture(scope="session")
def graph():
    return grid_graph()


@pytest.fixture(scope="session")
def segments(graph):
    return build_segments(graph)


@pytest.fixture(scope="session")
def weather():
    ts = pd.date_range("2026-03-01", "2026-04-30 23:00", freq="h")
    rng = np.random.default_rng(0)
    precip = np.where(rng.random(len(ts)) < 0.1, rng.exponential(4, len(ts)), 0.0)
    return pd.DataFrame({"ts": ts, "precip_mm": precip, "rain_mm": precip,
                         "weather_code": 0, "temperature_c": 25.0, "cloud_cover": 50})


def small_sim_params(**overrides):
    from routesense.sim.params import SimParams

    p = SimParams(
        start_date="2026-03-01", end_date="2026-04-30", n_vehicles=30, n_depots=2,
        n_destinations=30, n_emerging=3, emerging_window_days=30,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


@pytest.fixture(scope="session")
def city_segments():
    """Larger synthetic city for model tests."""
    from routesense.devcity import dev_city

    return build_segments(dev_city(n=24, seed=1))


@pytest.fixture(scope="session")
def city_weather():
    from routesense.devcity import dev_weather

    return dev_weather("2026-03-01", "2026-08-31", seed=1)


@pytest.fixture(scope="session")
def city_sim(city_segments, city_weather):
    """Six months, 60 vehicles, plenty of black spots so they are learnable."""
    from routesense.sim.simulator import simulate

    p = small_sim_params(
        start_date="2026-03-01", end_date="2026-08-31", n_vehicles=60, n_depots=3,
        n_destinations=120, n_emerging=4, emerging_window_days=45, black_spot_frac=0.05,
    )
    return p, simulate(city_segments, city_weather, p)


@pytest.fixture(scope="session")
def city_model(city_segments, city_sim, city_weather):
    """(exposure table, fitted RiskModel, RiskTable) on the city simulation."""
    from routesense.models.risk import RiskModel, exposure_table

    _, sim = city_sim
    ex = exposure_table(city_segments, sim.trips, sim.trip_segments, sim.events, city_weather)
    model = RiskModel.fit(city_segments, ex)
    return ex, model, model.risk_table(city_segments)


@pytest.fixture(scope="session")
def region_dir(tmp_path_factory, city_segments, city_weather, city_sim):
    """A built region (artifacts + eval report) for pipeline and API tests."""
    import json

    from routesense.evaluation import run
    from routesense.pipeline import build

    params, sim = city_sim
    d = tmp_path_factory.mktemp("regions") / "test_city"
    d.mkdir()
    city_segments.to_parquet(d / "segments.parquet", index=False)
    city_weather.to_parquet(d / "weather.parquet", index=False)
    sim.save(d / "sim")
    build(d, log=lambda *_: None)
    (d / "eval").mkdir()
    report = run(city_segments, sim, city_weather, params, impact_trips=30)
    (d / "eval" / "report.json").write_text(json.dumps(report, default=str))
    return d
