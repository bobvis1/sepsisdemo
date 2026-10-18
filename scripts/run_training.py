"""Launch a training job (src/sepsis/train.py) on SageMaker with the custom image.

Managed Spot, private VPC, network isolation, KMS-encrypted volume and output. Waits for the
job, then prints validation metrics read from the job's metrics.json.

    uv run python scripts/run_training.py --model xgboost
    uv run python scripts/run_training.py --model logreg --hp C=0.5

Features default to the most recent Processing output under s3://<bucket>/features/.
Cost: one ml.m5.large on Spot for a few minutes (a few cents). Needs ml.m5.large Spot quota.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    Compute,
    InputData,
    Networking,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
    Tag,
)

from sepsis.config import boto_session, load_infra, load_settings


def latest_features_prefix(s3, bucket: str) -> str:
    """Most recent features/<job>/ prefix (Processing job names embed a timestamp)."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix="features/", Delimiter="/")
    prefixes = sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))
    if not prefixes:
        raise SystemExit("No features/ output found; run scripts/run_processing.py first")
    return prefixes[-1].rstrip("/")


def read_job_output(s3, model_artifacts_uri: str) -> dict[str, dict]:
    """Download the job's model.tar.gz and return its params.json and metrics.json."""
    bucket, key = model_artifacts_uri.removeprefix("s3://").split("/", 1)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "model.tar.gz"
        s3.download_file(bucket, key, str(tar_path))
        with tarfile.open(tar_path) as tf:
            return {n: json.load(tf.extractfile(n)) for n in ("params.json", "metrics.json")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", "xgboost"], required=True)
    ap.add_argument("--features-prefix", help="default: latest under features/")
    ap.add_argument("--image-uri", help="default: <ecr repo>:latest")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--no-spot", action="store_true", help="on-demand instead of managed Spot")
    ap.add_argument("--hp", action="append", default=[], metavar="K=V", help="e.g. --hp eta=0.05")
    args = ap.parse_args()

    s, infra, boto = load_settings(), load_infra(), boto_session()
    s3 = boto.client("s3")
    prefix = args.features_prefix or latest_features_prefix(s3, infra.bucket)
    base = f"s3://{infra.bucket}/{prefix}"
    cli_args: list[str] = ["--model", args.model]
    for kv in args.hp:
        k, v = kv.split("=", 1)
        cli_args += [f"--{k.replace('_', '-')}", v]

    session = Session(boto_session=boto, default_bucket=infra.bucket)
    trainer = ModelTrainer(
        sagemaker_session=session,
        role=infra.role_arn,
        base_job_name=f"sepsis-{args.model}",
        training_image=args.image_uri or f"{infra.ecr_repository_url}:latest",
        source_code=SourceCode(command="python -m sepsis.train", args=cli_args),
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=1,
            volume_size_in_gb=10,
            volume_kms_key_id=infra.kms_key_arn,
            enable_managed_spot_training=not args.no_spot,
        ),
        networking=Networking(
            subnets=infra.subnet_ids,
            security_group_ids=[infra.security_group_id],
            enable_network_isolation=True,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=1800,
            max_wait_time_in_seconds=None if args.no_spot else 3600,
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{infra.bucket}/models/{args.model}",
            kms_key_id=infra.kms_key_arn,
        ),
        tags=[
            Tag(key="project", value="sepsis-sagemaker"),
            Tag(key="owner", value=s.owner),
            Tag(key="env", value="demo"),
        ],
    )
    trainer.train(
        input_data_config=[
            InputData(channel_name="train", data_source=f"{base}/train"),
            InputData(channel_name="validation", data_source=f"{base}/validation"),
        ],
        wait=True,
    )
    job = trainer._latest_training_job
    job.refresh()
    print(f"status={job.training_job_status} job={job.training_job_name}")
    out = read_job_output(s3, job.model_artifacts.s3_model_artifacts)
    for k, v in out["metrics.json"].items():
        print(f"{k}={v:.4f}")
    if not args.no_spot and job.billable_time_in_seconds and job.training_time_in_seconds:
        saved = 100 * (1 - job.billable_time_in_seconds / job.training_time_in_seconds)
        print(f"spot savings: {saved:.0f}%")


if __name__ == "__main__":
    main()
