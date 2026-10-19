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

# Project code (src/sepsis) and the vendored official scoring script (third_party/).
COPY pyproject.toml /opt/sepsis/
COPY src /opt/sepsis/src
COPY third_party /opt/sepsis/third_party
ENV SEPSIS_THIRD_PARTY=/opt/sepsis/third_party
RUN pip install --no-deps /opt/sepsis

# Training entry point for tuning jobs, which cannot override the entrypoint: SageMaker runs
# `docker run <image> train`, and the script reads /opt/ml/input/config/hyperparameters.json.
COPY container/train /usr/local/bin/train

# Inference entry point: SageMaker runs `docker run <image> serve` and expects port 8080.
COPY container/serve /usr/local/bin/serve
EXPOSE 8080

# Runs as root on purpose: SageMaker mounts /opt/ml/processing/* and /opt/ml/model root-owned,
# so a non-root user cannot write job output (PermissionError). Jobs run with network isolation
# and carry no credentials. SageMaker sets the entrypoint/command per job.
WORKDIR /opt/ml
