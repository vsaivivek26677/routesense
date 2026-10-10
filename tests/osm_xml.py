"""Writes a small OSM XML file with the tag variety real OpenStreetMap data has."""
import math
from pathlib import Path

import numpy as np


def write_osm_xml(path: Path, n: int = 12, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    lat0, lon0, step = 12.97, 77.62, 0.0015
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="routesense-test">']

    def nid(i, j):  # OSM node ids are large integers
        return 9_000_000_000 + i * n + j

    for i in range(n):
        for j in range(n):
            lines.append(f'<node id="{nid(i, j)}" lat="{lat0 + i * step + rng.normal(0, 1e-4):.7f}" '
                         f'lon="{lon0 + j * step + rng.normal(0, 1e-4):.7f}"/>')
    ways = []
    for i in range(n):  # rows: each split into two ways so simplification merges tag lists
        hw = "primary" if i % 4 == 0 else "residential"
        tags = {"highway": hw, "name": f"{i} Main Road"}
        if hw == "primary":
            tags.update({"maxspeed": "30 mph" if i == 4 else "40", "lanes": "2;3" if i == 8 else "2", "lit": "yes"})
        ways.append(([nid(i, j) for j in range(n // 2 + 1)], tags))
        tags2 = dict(tags, **({"lit": "no", "name": f"{i} Main Road Extension"} if hw == "residential" else {}))
        ways.append(([nid(i, j) for j in range(n // 2, n)], tags2))
    for j in range(n):  # columns
        hw = "secondary_link" if j == 5 else "secondary" if j % 3 == 0 else "service" if j == n - 1 else "unclassified"
        tags = {"highway": hw}
        if j % 2:
            tags["name"] = f"{j} Cross"
        if j == 7:
            tags["oneway"] = "yes"
        ways.append(([nid(i, j) for i in range(n)], tags))
    shape = []  # a curvy bridge with intermediate shape nodes
    for k in range(5):
        sid = 9_100_000_000 + k
        lines.append(f'<node id="{sid}" lat="{lat0 + 0.0003 * (k + 1):.7f}" lon="{lon0 + 0.0004 * math.sin(k):.7f}"/>')
        shape.append(sid)
    ways.append(([nid(0, 0), *shape, nid(2, 1)], {"highway": "tertiary", "name": "Bend Road", "maxspeed": "IN:urban", "bridge": "yes"}))
    for k, (nodes, tags) in enumerate(ways):
        refs = "".join(f'<nd ref="{x}"/>' for x in nodes)
        tag_xml = "".join(f'<tag k="{a}" v="{b}"/>' for a, b in tags.items())
        lines.append(f'<way id="{k + 1}">{refs}{tag_xml}</way>')
    lines.append("</osm>")
    path = Path(path)
    path.write_text("\n".join(lines))
    return path
