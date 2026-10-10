"""Build the model, hotspots, alerts and analytics the app serves for a region.

    python scripts/build.py --region dev_city
"""
from __future__ import annotations

import argparse

from routesense.config import PROCESSED_DIR
from routesense.pipeline import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="dev_city")
    build(PROCESSED_DIR / ap.parse_args().region)


if __name__ == "__main__":
    main()
