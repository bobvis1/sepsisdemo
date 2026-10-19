"""Deploy the latest Approved model as a SERVERLESS endpoint, invoke it, and delete it (Phase 8).

    uv run python scripts/deploy_endpoint.py                 # up, invoke the sample, down
    uv run python scripts/deploy_endpoint.py --keep          # leave it running (delete it yourself)
    uv run python scripts/deploy_endpoint.py --delete NAME   # delete a leftover endpoint

Serverless inference bills per request and per second of compute (no idle cost), so this demo costs
cents. Serverless endpoints reject both network isolation and (in this setup) the VPC config at
CreateEndpoint, so unlike the training and Batch Transform jobs the model runs outside the VPC.
The container only serves requests; the image holds no secrets. Everything is
deleted at the end, also when something fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sagemaker.core.resources import Endpoint, EndpointConfig, Model

from sepsis.config import boto_session, load_infra, load_settings
from sepsis.deploy import (
    build_bundle,
    create_model,
    latest_approved_package,
    project_tags,
    stamp,
)

SAMPLE = Path("docs/sample_request.json")


def delete(boto, region: str, name: str) -> None:
    """Delete endpoint, config and model that share `name` (ignore the ones already gone)."""
    for cls, kw in (
        (Endpoint, "endpoint_name"),
        (EndpointConfig, "endpoint_config_name"),
        (Model, "model_name"),
    ):
        try:
            res = cls.get(**{kw: name}, session=boto, region=region)
            res.delete()
            print(f"deleted {cls.__name__} {name}")
        except Exception as e:  # noqa: BLE001
            print(f"{cls.__name__} {name}: {type(e).__name__} (already gone?)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--delete", metavar="NAME")
    ap.add_argument("--image-tag", help="override the package's image (default: as registered)")
    ap.add_argument("--vpc", action="store_true", help="attach the VPC (CreateEndpoint rejects it)")
    ap.add_argument("--memory-mb", type=int, default=3072)
    ap.add_argument("--request", type=Path, default=SAMPLE)
    args = ap.parse_args()

    s, infra, boto = load_settings(), load_infra(), boto_session()
    if args.delete:
        return delete(boto, s.region, args.delete)

    sm, s3 = boto.client("sagemaker"), boto.client("s3")
    package = latest_approved_package(sm, infra.model_package_group_name)
    version = package["ModelPackageVersion"]
    model_data, image = build_bundle(s3, infra, package)
    image = f"{infra.ecr_repository_url}:{args.image_tag}" if args.image_tag else image
    name = f"sepsis-v{version}-{stamp()}"
    print(f"model package v{version} -> endpoint {name}")

    try:
        create_model(
            boto,
            s,
            infra,
            name=name,
            image=image,
            model_data=model_data,
            isolate=False,
            vpc=args.vpc,
        )  # serverless endpoints cannot use network isolation
        # boto3 for this one call: SDK 3.25.0 serialises serverless_config.memory_size_in_mb as
        # "MemorySizeInMb" and the API rejects it (it expects "MemorySizeInMB").
        sm.create_endpoint_config(
            EndpointConfigName=name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": name,
                    "ServerlessConfig": {"MemorySizeInMB": args.memory_mb, "MaxConcurrency": 2},
                }
            ],
            Tags=[{"Key": t.key, "Value": t.value} for t in project_tags(s)],
        )
        endpoint = Endpoint.create(
            endpoint_name=name, endpoint_config_name=name, tags=project_tags(s),
            session=boto, region=s.region,
        )  # fmt: skip
        endpoint.wait_for_status("InService")
        print("InService")

        body = args.request.read_text()
        runtime = boto.client("sagemaker-runtime")
        for label in ("cold start", "warm"):
            r = runtime.invoke_endpoint(
                EndpointName=name, ContentType="application/json", Body=body
            )
            out = json.loads(r["Body"].read())
            print(f"[{label}] {json.dumps(out)}")
    finally:
        if args.keep:
            print(f"left running; delete with: --delete {name}")
        else:
            delete(boto, s.region, name)


if __name__ == "__main__":
    main()
