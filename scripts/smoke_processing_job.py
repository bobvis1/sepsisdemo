"""Smoke test: a trivial Processing job in the private VPC with network isolation,
KMS-encrypted volume and output. Verifies the output object is SSE-KMS with our key.

The job also tries to reach the internet; with isolation on, it must fail ("BLOCKED").
Run: make smoke   (costs a few cents: one ml.t3.medium (override: SMOKE_INSTANCE) for ~2-3 minutes)
"""

from __future__ import annotations

import os
import time

from sagemaker.core import image_uris
from sagemaker.core.resources import ProcessingJob
from sagemaker.core.shapes import (
    AppSpecification,
    NetworkConfig,
    ProcessingClusterConfig,
    ProcessingOutput,
    ProcessingOutputConfig,
    ProcessingResources,
    ProcessingS3Output,
    ProcessingStoppingCondition,
    Tag,
    VpcConfig,
)

from sepsis.config import boto_session, load_infra, load_settings

INSTANCE = os.environ.get("SMOKE_INSTANCE", "ml.t3.medium")
PROBE = (
    "python3 -c \"import urllib.request as u; u.urlopen('https://example.com', timeout=5)\" "
    "&& echo REACHED > /opt/ml/processing/output/smoke.txt "
    "|| echo BLOCKED > /opt/ml/processing/output/smoke.txt"
)


def main() -> None:
    s, infra, session = load_settings(), load_infra(), boto_session()
    name = f"smoke-{int(time.time())}"
    out_uri = f"s3://{infra.bucket}/smoke/{name}"
    image = image_uris.retrieve(
        framework="sklearn", region=s.region, version="1.2-1", py_version="py3",
        instance_type=INSTANCE,
    )  # fmt: skip

    job = ProcessingJob.create(
        processing_job_name=name,
        role_arn=infra.role_arn,
        app_specification=AppSpecification(
            image_uri=image, container_entrypoint=["bash", "-c", PROBE]
        ),
        processing_resources=ProcessingResources(
            cluster_config=ProcessingClusterConfig(
                instance_count=1,
                instance_type=INSTANCE,
                volume_size_in_gb=5,
                volume_kms_key_id=infra.kms_key_arn,
            )
        ),
        processing_output_config=ProcessingOutputConfig(
            kms_key_id=infra.kms_key_arn,
            outputs=[
                ProcessingOutput(
                    output_name="out",
                    s3_output=ProcessingS3Output(
                        s3_uri=out_uri,
                        local_path="/opt/ml/processing/output",
                        s3_upload_mode="EndOfJob",
                    ),
                )
            ],
        ),
        network_config=NetworkConfig(
            enable_network_isolation=True,
            vpc_config=VpcConfig(
                security_group_ids=[infra.security_group_id], subnets=infra.subnet_ids
            ),
        ),
        stopping_condition=ProcessingStoppingCondition(max_runtime_in_seconds=600),
        tags=[
            Tag(key="project", value="sepsis-sagemaker"),
            Tag(key="owner", value=s.owner),
            Tag(key="env", value="demo"),
        ],
        session=session,
    )
    print(f"started {name}; waiting...")
    job.wait()
    job.refresh()
    status = job.processing_job_status
    if status != "Completed":
        raise SystemExit(f"job {status}: {getattr(job, 'failure_reason', '')}")

    s3 = session.client("s3")
    key = f"smoke/{name}/smoke.txt"
    head = s3.head_object(Bucket=infra.bucket, Key=key)
    body = s3.get_object(Bucket=infra.bucket, Key=key)["Body"].read().decode().strip()
    print(f"status={status} sse={head['ServerSideEncryption']} kms={head['SSEKMSKeyId']}")
    print(f"internet probe: {body}")
    assert head["ServerSideEncryption"] == "aws:kms" and head["SSEKMSKeyId"] == infra.kms_key_arn
    assert body == "BLOCKED", "network isolation is not working"
    print("OK: isolated, in-VPC job wrote a KMS-encrypted object")


if __name__ == "__main__":
    main()
