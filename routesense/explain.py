"""Why is this road risky? Plain-language reasons with their size.

Feature reasons come from counterfactuals on the feature model: the segment's risk
index is recomputed with one attribute set to a benign value (e.g. a simple 2-way
junction, a straight road, a lit road, dry weather), and the difference is that
attribute's effect. History reasons come from the empirical-Bayes multiplier, and
live context from hotspots and emerging alerts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from routesense.models.risk import (
    BAND_NAMES, GROUP_WEIGHTS, GROUPS, REFERENCE, RiskModel, RiskTable, segment_features,
)
from routesense.schema import VEHICLE_TYPES

def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "es" if word.endswith("s") else "s")


NIGHT_BANDS = {BAND_NAMES.index(b) for b in ("00-03", "03-06", "18-21", "21-24")}


@dataclass
class Explainer:
    segments: pd.DataFrame
    model: RiskModel
    table: RiskTable
    hotspots: pd.DataFrame | None = None
    alerts: pd.DataFrame | None = None

    def __post_init__(self):
        self.seg_f = segment_features(self.segments)
        self._hotspot_of: dict[int, dict] = {}
        if self.hotspots is not None:
            for rank, h in enumerate(self.hotspots.itertuples()):
                for sid in h.segment_ids:
                    self._hotspot_of.setdefault(int(sid), {"rank": rank + 1, "kind": h.kind, "name": h.name})
        self._alert_of: dict[int, dict] = {}
        if self.alerts is not None:
            for a in self.alerts.itertuples():
                for sid in a.segment_ids:
                    self._alert_of[int(sid)] = {"ratio": a.ratio, "top_event": a.top_event, "headline": a.headline}

    def _index(self, rows: pd.DataFrame, band: int, rain: int, vehicle: int, sid: int) -> np.ndarray:
        """Combined feature-model risk index for each row."""
        ids = np.full(len(rows), sid)
        return sum(GROUP_WEIGHTS[g] * self.model.prior_rate(g, rows, band, rain, vehicle, ids) for g in GROUPS)

    def reasons(self, sid: int, band: int, rain: int, vehicle: int, limit: int = 4) -> list[dict]:
        """Largest contributors to segment ``sid``'s risk under the given conditions."""
        s = self.segments.iloc[sid]
        base = self.seg_f.iloc[[sid]]
        variants = {}  # code -> (feature row, band, rain, vehicle, text)
        if s["junction_degree"] >= 4:
            variants["junction"] = (base.assign(junction_degree=2.0), band, rain, vehicle,
                                    f"Busy junction ({int(s['junction_degree'])} roads meet)")
        if s["max_turn_deg"] >= 30:
            variants["curve"] = (base.assign(max_turn_deg=0.0, total_turn_deg=0.0, sinuosity=1.0), band, rain, vehicle,
                                 f"Sharp bend ({s['max_turn_deg']:.0f}°)")
        if band in NIGHT_BANDS and not (s["lit"] == 1):
            text = "Unlit at night" if s["lit"] == 0 else "Street lighting unknown at night"
            variants["lighting"] = (base.assign(lit=1.0), band, rain, vehicle, text)
        if rain:
            variants["rain"] = (base, band, 0, vehicle, "Wet road")
        ref_band = REFERENCE[0]
        if band != ref_band:
            variants["time"] = (base, ref_band, rain, vehicle, f"Time of day ({BAND_NAMES[band]} h)")
        lcv = VEHICLE_TYPES.index("lcv")
        if vehicle != lcv:
            variants["vehicle"] = (base, band, rain, lcv, f"Vehicle type ({VEHICLE_TYPES[vehicle].replace('_', ' ')})")

        actual = float(self._index(base, band, rain, vehicle, sid)[0])
        out = []
        for code, (row, b, r, v, text) in variants.items():
            cf = float(self._index(row, b, r, v, sid)[0])
            effect = actual / cf - 1 if cf > 0 else 0.0
            if effect >= 0.05:
                out.append({"code": code, "text": text, "effect_pct": round(100 * effect)})

        # Segment-specific history beyond what its features explain.
        prior = {g: GROUP_WEIGHTS[g] * self.model.prior_rate(g, base, band, rain, vehicle, [sid])[0] for g in GROUPS}
        theta = sum(prior[g] * self.model.theta_mean[g][sid] for g in GROUPS) / sum(prior.values())
        cred = sum(prior[g] * self.model.credibility(g)[sid] for g in GROUPS) / sum(prior.values())
        if theta >= 1.2 and cred >= 0.3:
            nm = int(self.model.observed["near_miss"][sid])
            acc = int(self.model.observed["accident"][sid])
            harsh = int(self.model.observed["harsh"][sid])
            counts = ", ".join(_plural(n, word) for n, word in (
                (harsh, "harsh event"), (nm, "near-miss"), (acc, "accident")) if n)
            out.append({"code": "history", "effect_pct": round(100 * (theta - 1)),
                        "text": f"Incident history: {theta:.1f}x similar roads ({counts})"})
        out.sort(key=lambda r: -r["effect_pct"])
        out = out[:limit]

        if sid in self._alert_of:
            a = self._alert_of[sid]
            out.insert(0, {"code": "emerging", "effect_pct": round(100 * (a["ratio"] - 1)),
                           "text": f"New: {a['headline']}"})
        if sid in self._hotspot_of:
            h = self._hotspot_of[sid]
            out.append({"code": "hotspot", "effect_pct": None,
                        "text": f"Known {h['kind']} hotspot (#{h['rank']} in the city)"})
        if self.table.data_sparse[sid]:
            out.append({"code": "sparse", "effect_pct": None,
                        "text": f"Fewer than {self.model.min_passes} fleet passes here; estimate leans on similar roads"})
        return out
