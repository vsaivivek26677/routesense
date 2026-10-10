"""Serve the API and dispatcher UI for a built region.

    python scripts/serve.py --region dev_city --port 8000
"""
from __future__ import annotations

import argparse
import os

import uvicorn

from routesense.config import PROCESSED_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default="dev_city")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    os.environ["ROUTESENSE_REGION_DIR"] = str(PROCESSED_DIR / args.region)
    uvicorn.run("routesense.api.app:app_from_env", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
