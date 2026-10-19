"""Batch Transform over a sample of hospital B patients, checked against the evaluate.py scores.

    uv run python scripts/run_batch_transform.py [--n-patients 1000] [--image-tag TAG]

Scores every hour of each sampled hospital B patient with the Approved model (same container as the
endpoint, network-isolated, in the private VPC) and compares them with the scores that
sepsis.evaluate computes from the batch features (Platt calibration from the same evaluation.json).
Cost: one ml.m5.large for a few minutes (cents). The model and job are deleted/finished afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sagemaker.core.resources import Model, TransformJob
from sagemaker.core.shapes import (
    TransformDataSource,
    TransformInput,
    TransformOutput,
    TransformResources,
    TransformS3DataSource,
)

from sepsis import io
from sepsis.config import boto_session, load_infra, load_settings
from sepsis.deploy import (
    build_bundle,
    create_model,
    latest_approved_package,
    project_tags,
    stamp,
)
from sepsis.evaluate import load_model, platt_apply, predict

DROP = {"patient_id", "hospital", "SepsisLabel"}
ROOT = Path(__file__).resolve().parents[1]


def sample_lines(n: int, seed: int) -> tuple[list[str], list[str]]:
    df = io.read_hospital(ROOT / "data" / "parquet", "B")
    ids = sorted(df["patient_id"].unique())
    pick = sorted(np.random.default_rng(seed).choice(ids, size=min(n, len(ids)), replace=False))
    lines = []
    for pid, g in df[df["patient_id"].isin(pick)].groupby("patient_id", sort=True):
        g = g.sort_values("ICULOS")
        records = [
            {k: (int(v) if k == "ICULOS" else float(v)) for k, v in r.items()
             if k not in DROP and v == v}
            for r in g.to_dict("records")
        ]  # fmt: skip
        lines.append(json.dumps({"patient_id": pid, "records": records, "all_hours": True}))
    return pick, lines


def reference_scores(
    model_dir: Path, evaluation: dict, patients: list[str]
) -> dict[str, np.ndarray]:
    """evaluate.py's path: batch features -> model -> Platt calibration (hospital B rows)."""
    model, cols = load_model(model_dir)
    test = pd.read_parquet(ROOT / "data" / "features" / "test" / "features.parquet")
    test = test[test["patient_id"].isin(patients)].sort_values(["patient_id", "ICULOS"])
    p = platt_apply(predict(model, cols, test), **evaluation["calibration"])
    return {pid: p[idx] for pid, idx in test.groupby("patient_id").indices.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-tag")
    ap.add_argument("--reference-model-dir", type=Path, default=ROOT / "outputs/pipe2/model")
    args = ap.parse_args()

    s, infra, boto = load_settings(), load_infra(), boto_session()
    sm, s3 = boto.client("sagemaker"), boto.client("s3")
    package = latest_approved_package(sm, infra.model_package_group_name)
    model_data, image = build_bundle(s3, infra, package)
    image = f"{infra.ecr_repository_url}:{args.image_tag}" if args.image_tag else image
    name = f"sepsis-v{package['ModelPackageVersion']}-bt-{stamp()}"

    patients, lines = sample_lines(args.n_patients, args.seed)
    key = f"transform/{name}/input/hospital_B.jsonl"
    body = ("\n".join(lines) + "\n").encode()
    s3.put_object(
        Bucket=infra.bucket, Key=key, Body=body,
        ServerSideEncryption="aws:kms", SSEKMSKeyId=infra.kms_key_arn,
    )  # fmt: skip
    print(f"{len(patients)} hospital B patients, {len(body) / 1e6:.1f} MB -> {name}")

    create_model(boto, s, infra, name=name, image=image, model_data=model_data)  # isolated, VPC
    try:
        job = TransformJob.create(
            transform_job_name=name,
            model_name=name,
            transform_input=TransformInput(
                data_source=TransformDataSource(
                    s3_data_source=TransformS3DataSource(
                        s3_data_type="S3Prefix", s3_uri=f"s3://{infra.bucket}/{key}"
                    )
                ),
                content_type="application/jsonlines",
                split_type="Line",
            ),
            transform_output=TransformOutput(
                s3_output_path=f"s3://{infra.bucket}/transform/{name}/output",
                accept="application/jsonlines",
                assemble_with="Line",
                kms_key_id=infra.kms_key_arn,
            ),
            transform_resources=TransformResources(
                instance_type="ml.m5.large", instance_count=1, volume_kms_key_id=infra.kms_key_arn
            ),
            batch_strategy="MultiRecord",
            max_payload_in_mb=3,  # the handler rejects bodies over 4 MB
            max_concurrent_transforms=1,
            # dicts: TransformJob.create validates a different Tag class than Model/EndpointConfig
            tags=[{"key": t.key, "value": t.value} for t in project_tags(s)],
            session=boto,
            region=s.region,
        )
        job.wait()
        job.refresh()
        print("transform job:", job.transform_job_status)
    finally:
        Model.get(model_name=name, session=boto, region=s.region).delete()
        print("deleted model", name)

    out = (
        s3.get_object(Bucket=infra.bucket, Key=f"transform/{name}/output/hospital_B.jsonl.out")[
            "Body"
        ]
        .read()
        .decode()
    )
    got = {r["patient_id"]: np.array(r["scores"]) for r in map(json.loads, out.splitlines())}

    evaluation = json.loads((args.reference_model_dir.parent / "eval_xgboost.json").read_text())
    ref = reference_scores(args.reference_model_dir, evaluation, patients)
    missing = set(patients) - set(got)
    diffs = np.concatenate([np.abs(got[p] - ref[p]) for p in patients if p in got])
    n_rows = int(sum(len(v) for v in got.values()))
    thr = evaluation["threshold"]
    flips = int(sum(((got[p] >= thr) != (ref[p] >= thr)).sum() for p in got))
    result = {
        "transform_job": name, "model_package_version": package["ModelPackageVersion"],
        "patients": len(got), "missing_patients": len(missing), "rows": n_rows,
        "max_abs_diff": float(diffs.max()), "mean_abs_diff": float(diffs.mean()),
        "alert_flips": flips,
    }  # fmt: skip
    print(json.dumps(result, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/batch_transform_parity.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
