"""Log finished SageMaker runs to the managed MLflow server (after the fact, from the client).

    uv run python scripts/log_to_mlflow.py --pipeline-execution c4hbrf2kv8hz
    uv run python scripts/log_to_mlflow.py --tuning-job sepsis-xgb-tune-261019-0914

Needs `uv sync --extra tracking` and the MLflow server from Terraform (enable_mlflow: true).
"""

from __future__ import annotations

import argparse

from sepsis.config import load_infra
from sepsis.tracking import log_pipeline_execution, log_tuning_job


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-execution", action="append", default=[], metavar="ID")
    ap.add_argument("--tuning-job", action="append", default=[], metavar="NAME")
    args = ap.parse_args()
    infra = load_infra()
    for ex in args.pipeline_execution:
        print("pipeline", ex, "->", log_pipeline_execution(infra, ex))
    for job in args.tuning_job:
        print("tuning", job, "->", log_tuning_job(infra, job))


if __name__ == "__main__":
    main()
