"""Build, register (upsert) and optionally start the sepsis SageMaker Pipeline.

    uv run python scripts/run_pipeline.py                 # dry run: build + validate the definition
    uv run python scripts/run_pipeline.py --upsert        # create/update the pipeline (free)
    uv run python scripts/run_pipeline.py --upsert --start   # also run it (billable: see below)
    uv run python scripts/run_pipeline.py --tuned --upsert --start   # XGBoost uses the Phase 5 best

A full execution runs 1 Processing + 2 Training (Spot) + 2 Processing jobs: roughly $0.05-0.10.
Pin the image with --image-tag <git sha> (scripts/push_image.sh prints it); default is :latest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sepsis.config import load_infra, load_settings
from sepsis.pipeline import build_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-tag", default="latest")
    ap.add_argument("--xgb-hp", action="append", default=[], metavar="K=V")
    ap.add_argument("--logreg-hp", action="append", default=[], metavar="K=V")
    ap.add_argument(
        "--tuned", action="store_true", help="XGBoost hyperparameters from docs/tuning_result.json"
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--upsert", action="store_true", help="create or update the pipeline")
    ap.add_argument("--start", action="store_true", help="start an execution and wait")
    args = ap.parse_args()

    xgb_hp = list(args.xgb_hp)
    if args.tuned:
        best = json.loads(Path("docs/tuning_result.json").read_text())["best_hyperparameters"]
        xgb_hp = [f"{k}={v}" for k, v in best.items()] + xgb_hp
        print("tuned XGBoost hyperparameters:", xgb_hp)

    s, infra = load_settings(), load_infra()
    pipe = build_pipeline(
        s, infra, image_uri=f"{infra.ecr_repository_url}:{args.image_tag}",
        xgb_hp=xgb_hp, logreg_hp=args.logreg_hp, n_boot=args.n_boot,
    )  # fmt: skip
    definition = json.loads(pipe.definition())
    out = Path("outputs/pipeline_definition.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(definition, indent=2))
    print("steps:", [st["Name"] for st in definition["Steps"]])
    print(f"definition written to {out}")

    if args.upsert or args.start:
        pipe.upsert(role_arn=infra.role_arn)
        print(f"pipeline {pipe.name} upserted")
    if args.start:
        execution = pipe.start()
        print("started", execution.arn.rsplit("/", 1)[-1])
        execution.wait()
        for st in execution.list_steps():
            print(f"  {st['StepName']}: {st['StepStatus']}")


if __name__ == "__main__":
    main()
