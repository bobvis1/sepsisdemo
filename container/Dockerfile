# Processing / evaluation image. Build from the repo root:
#   docker build --platform linux/amd64 --provenance=false --sbom=false -f container/Dockerfile -t sepsis:dev .
# (SageMaker instances are x86_64; build for amd64 even on Apple Silicon.)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# OpenMP runtime for XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY container/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Project code (src/sepsis) and, from Phase 6, the vendored scoring script.
COPY pyproject.toml /opt/sepsis/
COPY src /opt/sepsis/src
RUN pip install --no-deps /opt/sepsis

# Run as non-root. SageMaker sets the entrypoint/command per job.
RUN useradd --create-home --uid 1000 sepsis
USER sepsis
WORKDIR /home/sepsis
