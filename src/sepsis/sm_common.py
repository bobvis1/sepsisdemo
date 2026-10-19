"""Shared SageMaker job settings (secure defaults) used by the scripts and the Pipeline.

Every job: private VPC subnets, network isolation, KMS-encrypted volume and output, project tags,
and the project's custom image. Pass a `PipelineSession` to get pipeline step arguments instead
of starting the job immediately.
"""

from __future__ import annotations

from sagemaker.core.network import NetworkConfig
from sagemaker.core.processing import Processor
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    Compute,
    Networking,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
    Tag,
)

from sepsis.config import Infra, Settings


def tags(s: Settings) -> list[Tag]:
    return [
        Tag(key="project", value="sepsis-sagemaker"),
        Tag(key="owner", value=s.owner),
        Tag(key="env", value="demo"),
    ]


def default_image(infra: Infra) -> str:
    return f"{infra.ecr_repository_url}:latest"


def make_processor(
    session,
    s: Settings,
    infra: Infra,
    *,
    module: str,
    name: str,
    instance_type,
    image_uri: str | None = None,
    max_runtime: int = 1800,
):
    """Processor that runs `python -m <module>` from the custom image."""
    return Processor(
        role=infra.role_arn,
        image_uri=image_uri or default_image(infra),
        instance_count=1,
        instance_type=instance_type,
        entrypoint=["python", "-m", module],
        volume_size_in_gb=10,
        volume_kms_key=infra.kms_key_arn,
        output_kms_key=infra.kms_key_arn,
        max_runtime_in_seconds=max_runtime,
        base_job_name=name,
        sagemaker_session=session,
        tags=[{"Key": t.key, "Value": t.value} for t in tags(s)],
        network_config=NetworkConfig(
            enable_network_isolation=True,
            security_group_ids=[infra.security_group_id],
            subnets=infra.subnet_ids,
        ),
    )


def make_trainer(
    session,
    s: Settings,
    infra: Infra,
    *,
    model: str,
    cli_args: list[str],
    instance_type,
    image_uri: str | None = None,
    spot: bool = True,
):
    """ModelTrainer that runs `python -m sepsis.train --model <model> ...` from the custom image."""
    return ModelTrainer(
        sagemaker_session=session,
        role=infra.role_arn,
        base_job_name=f"sepsis-{model}",
        training_image=image_uri or default_image(infra),
        source_code=SourceCode(
            command="python -m sepsis.train", args=["--model", model, *cli_args]
        ),
        compute=Compute(
            instance_type=instance_type,
            instance_count=1,
            volume_size_in_gb=10,
            volume_kms_key_id=infra.kms_key_arn,
            enable_managed_spot_training=spot,
        ),
        networking=Networking(
            subnets=infra.subnet_ids,
            security_group_ids=[infra.security_group_id],
            enable_network_isolation=True,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=1800, max_wait_time_in_seconds=3600 if spot else None
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{infra.bucket}/models/{model}", kms_key_id=infra.kms_key_arn
        ),
        tags=tags(s),
    )


def hp_to_args(hp: list[str]) -> list[str]:
    """['eta=0.05', 'max_depth=4'] -> ['--eta', '0.05', '--max-depth', '4']"""
    out: list[str] = []
    for kv in hp:
        k, v = kv.split("=", 1)
        out += [f"--{k.replace('_', '-')}", v]
    return out


def latest_features_prefix(s3, bucket: str) -> str:
    """Most recent features/<job>/ prefix (Processing job names embed a timestamp)."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix="features/", Delimiter="/")
    prefixes = sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))
    if not prefixes:
        raise SystemExit("No features/ output found; run scripts/run_processing.py first")
    return prefixes[-1].rstrip("/")
