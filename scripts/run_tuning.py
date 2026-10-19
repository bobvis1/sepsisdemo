"""Bayesian hyperparameter tuning of XGBoost on SageMaker (Phase 5).

Objective: validation AUPRC on the hospital A validation split. Hospital B is never read.
Jobs run on managed Spot, in the private VPC, with network isolation and KMS-encrypted volumes.

    uv run python scripts/run_tuning.py --image-tag <sha>
    uv run python scripts/run_tuning.py --image-tag <sha> --max-jobs 2 --max-parallel 1  # smoke

The tuning API cannot override the container entrypoint, so the image's `train` script reads
SageMaker's hyperparameters.json (see src/sepsis/train.py). Writes docs/tuning_result.json.
Cost: up to max-jobs ml.m5.large Spot runs of a few minutes each (well under $1 for 20 jobs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.parameter import ContinuousParameter, IntegerParameter
from sagemaker.train.configs import InputData
from sagemaker.train.tuner import HyperparameterTuner

from sepsis.config import boto_session, load_infra, load_settings
from sepsis.sm_common import latest_features_prefix, make_trainer

OBJECTIVE = "validation:auprc"
RESULT = Path("docs/tuning_result.json")


class SecureTuner(HyperparameterTuner):
    """The SDK tuner drops network isolation and the volume KMS key; set them here."""

    kms_key_id: str | None = None

    def _build_training_job_definition(self, inputs):
        definition = super()._build_training_job_definition(inputs)
        definition.enable_network_isolation = True
        definition.resource_config.volume_kms_key_id = self.kms_key_id
        return definition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-tag", help="ECR tag of the image to use (default: latest)")
    ap.add_argument("--features-prefix", help="default: latest under features/")
    ap.add_argument("--max-jobs", type=int, default=20)
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--no-spot", action="store_true", help="on-demand if Spot capacity is short")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    s, infra, boto = load_settings(), load_infra(), boto_session()
    prefix = args.features_prefix or latest_features_prefix(boto.client("s3"), infra.bucket)
    base = f"s3://{infra.bucket}/{prefix}"
    image = f"{infra.ecr_repository_url}:{args.image_tag}" if args.image_tag else None

    session = Session(boto_session=boto, default_bucket=infra.bucket)
    trainer = make_trainer(
        session, s, infra, model="xgboost", cli_args=[], instance_type="ml.m5.large",
        image_uri=image, spot=not args.no_spot,
    )  # fmt: skip
    # Static hyperparameters; the tuner adds the ranges below. `model` selects the algorithm.
    # SDK 3.25.0 expects boto-style tag dicts on the tuner but appends the trainer's Tag objects
    # to them and then fails; give the tuner the tags and clear the trainer's.
    tuning_tags = [{"Key": t.key, "Value": t.value} for t in trainer.tags]
    trainer.tags = []
    trainer.hyperparameters = {"model": "xgboost", "seed": 42, "num_round": 500}

    tuner = SecureTuner(
        model_trainer=trainer,
        objective_metric_name=OBJECTIVE,
        objective_type="Maximize",
        metric_definitions=[{"Name": OBJECTIVE, "Regex": r"validation:auprc=([0-9\.]+)"}],
        hyperparameter_ranges={
            "eta": ContinuousParameter(0.02, 0.3, scaling_type="Logarithmic"),
            "max_depth": IntegerParameter(3, 9),
            "min_child_weight": ContinuousParameter(1, 20, scaling_type="Logarithmic"),
            "subsample": ContinuousParameter(0.5, 1.0),
            "colsample_bytree": ContinuousParameter(0.4, 1.0),
        },
        strategy="Bayesian",
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel,
        early_stopping_type="Off",
        base_tuning_job_name="sepsis-xgb-tune",
        tags=tuning_tags,
    )
    tuner.kms_key_id = infra.kms_key_arn
    tuner.tune(
        inputs=[
            InputData(channel_name="train", data_source=f"{base}/train"),
            InputData(channel_name="validation", data_source=f"{base}/validation"),
        ],
        wait=not args.no_wait,
    )
    name = tuner.latest_tuning_job.hyper_parameter_tuning_job_name
    print(f"tuning job: {name}")
    if args.no_wait:
        return

    best = tuner.latest_tuning_job.refresh().best_training_job
    result = {
        "tuning_job": name,
        "objective": OBJECTIVE,
        "best_training_job": best.training_job_name,
        "best_value": best.final_hyper_parameter_tuning_job_objective_metric.value,
        "best_hyperparameters": dict(best.tuned_hyper_parameters),
        "max_jobs": args.max_jobs,
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
