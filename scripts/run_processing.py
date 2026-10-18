"""Run the feature-engineering Processing job (src/sepsis/process.py) on SageMaker.

Custom image, private VPC, network isolation, KMS-encrypted volume and output. Then, with
--compare-local, download the output and check it equals a local run (Phase 3 acceptance).

    uv run python scripts/run_processing.py --compare-local data/features
Cost: one ml.t3.large for a few minutes (about $0.01); ml.m5 quotas default to 0.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from sagemaker.core.resources import ProcessingJob
from sagemaker.core.shapes import (
    AppSpecification,
    NetworkConfig,
    ProcessingClusterConfig,
    ProcessingInput,
    ProcessingOutput,
    ProcessingOutputConfig,
    ProcessingResources,
    ProcessingS3Input,
    ProcessingS3Output,
    ProcessingStoppingCondition,
    Tag,
    VpcConfig,
)

from sepsis.config import boto_session, load_infra, load_settings

IN, OUT = "/opt/ml/processing/input", "/opt/ml/processing/output"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-uri", help="default: <ecr repo>:latest")
    ap.add_argument("--instance-type", default="ml.t3.large")
    ap.add_argument("--input-prefix", default="data/parquet")
    ap.add_argument("--compare-local", type=Path, help="local features dir to compare against")
    args = ap.parse_args()

    s, infra, session = load_settings(), load_infra(), boto_session()
    name = f"sepsis-features-{int(time.time())}"
    out_uri = f"s3://{infra.bucket}/features/{name}"
    job = ProcessingJob.create(
        processing_job_name=name,
        role_arn=infra.role_arn,
        app_specification=AppSpecification(
            image_uri=args.image_uri or f"{infra.ecr_repository_url}:latest",
            container_entrypoint=["python", "-m", "sepsis.process"],
        ),
        processing_inputs=[
            ProcessingInput(
                input_name="parquet",
                s3_input=ProcessingS3Input(
                    s3_uri=f"s3://{infra.bucket}/{args.input_prefix}",
                    local_path=IN,
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                ),
            )
        ],
        processing_resources=ProcessingResources(
            cluster_config=ProcessingClusterConfig(
                instance_count=1,
                instance_type=args.instance_type,
                volume_size_in_gb=10,
                volume_kms_key_id=infra.kms_key_arn,
            )
        ),
        processing_output_config=ProcessingOutputConfig(
            kms_key_id=infra.kms_key_arn,
            outputs=[
                ProcessingOutput(
                    output_name="features",
                    s3_output=ProcessingS3Output(
                        s3_uri=out_uri, local_path=OUT, s3_upload_mode="EndOfJob"
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
        stopping_condition=ProcessingStoppingCondition(max_runtime_in_seconds=1800),
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
    if job.processing_job_status != "Completed":
        raise SystemExit(f"job {job.processing_job_status}: {getattr(job, 'failure_reason', '')}")
    print(f"Completed -> s3://<bucket>/features/{name}/")

    if args.compare_local:
        compare(session, infra.bucket, f"features/{name}", args.compare_local)


def compare(session, bucket: str, prefix: str, local: Path) -> None:
    import pandas as pd

    s3 = session.client("s3")
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("train", "validation", "test"):
            dest = Path(tmp) / f"{split}.parquet"
            s3.download_file(bucket, f"{prefix}/{split}/features.parquet", str(dest))
            head = s3.head_object(Bucket=bucket, Key=f"{prefix}/{split}/features.parquet")
            pd.testing.assert_frame_equal(
                pd.read_parquet(dest), pd.read_parquet(local / split / "features.parquet")
            )
            print(f"{split}: identical to local run (sse={head['ServerSideEncryption']})")
    print("OK: SageMaker Processing output matches the local run")


if __name__ == "__main__":
    main()
