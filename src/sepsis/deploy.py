"""Shared deployment helpers: build the deployable bundle from a registered model package and
create the SageMaker Model (VPC, network isolation, project tags).

The training artifact holds model.joblib only. The serving code also needs the calibration and the
alert threshold chosen on hospital A validation, which the pipeline's Evaluate step wrote to
evaluation.json and attached to the model package as its model-quality statistics. The bundle is
the registered artifact plus that file, so a deployed model is traceable to one package version.
"""

from __future__ import annotations

import tarfile
import tempfile
import time
from pathlib import Path

from sagemaker.core.resources import Model
from sagemaker.core.shapes import ContainerDefinition, Tag, VpcConfig

from sepsis.config import Infra, Settings


def latest_approved_package(sm, group: str) -> dict:
    resp = sm.list_model_packages(
        ModelPackageGroupName=group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    if not resp:
        raise SystemExit(f"No Approved model package in group {group}")
    return sm.describe_model_package(ModelPackageName=resp[0]["ModelPackageArn"])


def _split(uri: str) -> tuple[str, str]:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return bucket, key


def build_bundle(s3, infra: Infra, package: dict) -> tuple[str, str]:
    """Upload the registered artifact + evaluation.json as one model.tar.gz; return (uri, image)."""
    container = package["InferenceSpecification"]["Containers"][0]
    stats = package["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]
    version = package["ModelPackageVersion"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s3.download_file(*_split(container["ModelDataUrl"]), str(tmp / "in.tar.gz"))
        s3.download_file(*_split(stats), str(tmp / "evaluation.json"))
        with tarfile.open(tmp / "in.tar.gz") as tf:
            tf.extractall(tmp / "model", filter="data")
        with tarfile.open(tmp / "model.tar.gz", "w:gz") as out:
            for name in ("model.joblib", "params.json"):
                out.add(tmp / "model" / name, arcname=name)
            out.add(tmp / "evaluation.json", arcname="evaluation.json")
        key = f"deploy/{package['ModelPackageGroupName']}-v{version}/model.tar.gz"
        s3.upload_file(
            str(tmp / "model.tar.gz"), infra.bucket, key,
            ExtraArgs={"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": infra.kms_key_arn},
        )  # fmt: skip
    return f"s3://{infra.bucket}/{key}", container["Image"]


def project_tags(s: Settings) -> list[Tag]:
    return [
        Tag(key="project", value="sepsis-sagemaker"),
        Tag(key="owner", value=s.owner),
        Tag(key="env", value="demo"),
    ]


def create_model(
    boto,
    s: Settings,
    infra: Infra,
    *,
    name: str,
    image: str,
    model_data: str,
    isolate: bool = True,
    vpc: bool = True,
):
    """Model with the project image in the private VPC; `isolate` removes network access.

    Serverless endpoints do not support network isolation (AWS restriction), so they pass False.
    """
    return Model.create(
        model_name=name,
        primary_container=ContainerDefinition(image=image, model_data_url=model_data),
        execution_role_arn=infra.role_arn,
        vpc_config=VpcConfig(security_group_ids=[infra.security_group_id], subnets=infra.subnet_ids)
        if vpc
        else None,
        enable_network_isolation=isolate,
        tags=project_tags(s),
        session=boto,
        region=s.region,
    )


def stamp() -> str:
    return time.strftime("%y%m%d-%H%M%S")


__all__ = ["build_bundle", "create_model", "latest_approved_package", "project_tags", "stamp"]
