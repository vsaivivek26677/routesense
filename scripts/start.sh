#!/bin/sh
# Container entry point: build the model if the image build could not, then serve on $PORT.
set -e
if [ ! -f "$ROUTESENSE_REGION_DIR/artifacts/meta.json" ]; then
    echo "no model in the image; building it now (about 10 min)"
    python scripts/run_region.py --region "$ROUTESENSE_REGION"
    rm -rf "$ROUTESENSE_REGION_DIR/sim"
fi
exec uvicorn routesense.api.app:app_from_env --factory --host 0.0.0.0 --port "${PORT:-7860}"
