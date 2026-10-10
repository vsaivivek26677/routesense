# RouteSense web app as one container: FastAPI API + dispatcher UI on port 7860.
#
# Works on any Docker host (e.g. Azure Container Apps); runs as an unprivileged user (uid 1000):
#   docker build -t routesense . && docker run -p 7860:7860 routesense
#
# The risk model is built from the committed road and weather snapshot (snapshots/blr_core/)
# during the image build, so the container starts in about 15 s. If prebuilt artifacts are
# present in data/processed/blr_core/ (built locally with scripts/run_region.py), that step is skipped.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-deploy.txt /tmp/requirements-deploy.txt
RUN pip install -r /tmp/requirements-deploy.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PYTHONPATH=/home/user/app \
    ROUTESENSE_REGION=blr_core \
    ROUTESENSE_REGION_DIR=/home/user/app/data/processed/blr_core \
    ROUTESENSE_LLM_DAILY_LIMIT=300 \
    PORT=7860
WORKDIR /home/user/app
COPY --chown=user . .

# Simulate the fleet, fit the model and evaluate (~10 min). Simulation output is not needed at runtime.
# If this step fails on the build machine, scripts/start.sh builds the model when the container starts.
RUN if [ ! -f "$ROUTESENSE_REGION_DIR/artifacts/meta.json" ]; then \
        python scripts/run_region.py --region "$ROUTESENSE_REGION" \
        || { echo "model build failed during the image build; it will run at startup"; rm -rf data/processed; }; \
    fi; \
    rm -rf "$ROUTESENSE_REGION_DIR/sim" data/raw

EXPOSE 7860
CMD ["sh", "scripts/start.sh"]
