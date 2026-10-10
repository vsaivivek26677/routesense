"""Project-wide settings: study regions and data locations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# All timestamps in this project are naive local time (Asia/Kolkata).
TIMEZONE = "Asia/Kolkata"


@dataclass(frozen=True)
class Region:
    name: str
    bbox: tuple[float, float, float, float]  # (west, south, east, north), EPSG:4326

    @property
    def center(self) -> tuple[float, float]:
        """(lat, lon) of the bbox centre."""
        w, s, e, n = self.bbox
        return (s + n) / 2, (w + e) / 2


REGIONS = {
    # ~16 km x 13 km around central/east Bengaluru (MG Road, Indiranagar, Koramangala, ORR).
    "blr_core": Region("blr_core", (77.55, 12.90, 77.70, 13.02)),
    # Small area for quick experiments.
    "blr_small": Region("blr_small", (77.60, 12.95, 77.65, 12.99)),
}
