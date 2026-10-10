"""Small geometry helpers shared by the simulator and the models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from routesense.ingest.osm import node_table


def local_xy(lat, lon, lat0: float) -> np.ndarray:
    """Equirectangular metres around latitude ``lat0`` (accurate at city scale)."""
    return np.column_stack((np.asarray(lon) * 111_320 * np.cos(np.radians(lat0)), np.asarray(lat) * 110_540))


def junction_zones(segments: pd.DataFrame, radius_m: float = 75.0) -> tuple[np.ndarray, csr_matrix]:
    """(node ids, incidence) where row i marks the segments in node i's junction zone.

    A zone holds every segment touching the node plus every segment whose midpoint lies
    within ``radius_m``. OSM splits junctions into very short pieces, so the touching
    segments alone can be only a few metres of road.
    """
    nodes = node_table(segments)
    lat0 = float(nodes["lat"].mean())
    mids = shapely.line_interpolate_point(segments.geometry.to_numpy(), 0.5, normalized=True)
    seg_xy = local_xy(shapely.get_y(mids), shapely.get_x(mids), lat0)
    node_xy = local_xy(nodes["lat"].to_numpy(), nodes["lon"].to_numpy(), lat0)

    near = cKDTree(seg_xy).query_ball_point(node_xy, r=radius_m)
    rows = np.repeat(np.arange(len(nodes)), [len(x) for x in near])
    cols = np.concatenate([np.asarray(x, dtype=np.int64) for x in near]) if len(rows) else np.array([], np.int64)
    index = pd.Series(np.arange(len(nodes)), index=nodes.index)
    sids = np.arange(len(segments))
    rows = np.concatenate([rows, index[segments["u"]].to_numpy(), index[segments["v"]].to_numpy()])
    cols = np.concatenate([cols, sids, sids])
    m = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(nodes), len(segments)))
    m.data[:] = 1.0  # duplicates (touching and within radius) count once
    m.sum_duplicates()
    m.data[:] = 1.0
    return nodes.index.to_numpy(), m
