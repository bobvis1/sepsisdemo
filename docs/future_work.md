# Future work

Ideas deliberately left out to keep the project finishable. Add to this list as they come up.

- **SageMaker Studio in VPC-only mode.** This project drives SageMaker from the local machine
  with the Python SDK. A production lab would run notebooks in Studio with VPC-only networking.
- **CI/CD** for the image build and `terraform plan` on pull requests.
- **Terraform remote state** (S3 + lock) once more than one person applies the stack.
- **Docker/ECR build in the cloud** (CodeBuild) for people without a local Docker.
- **Per-prediction uncertainty** (conformal prediction, ensembles).
- **Deep sequence models** (GRU/transformer) as comparison to the gradient boosted model.
- **Permissions boundary** on `sepsis-*` roles to close the inline-admin escalation path (see infra/iam/README.md).

## Operating point (from Phase 6)
The challenge utility penalises a false alarm at only 0.05 per hour, so the utility-maximising
threshold alerts on 11-25% of patient-hours (about 2-4 false alarms per non-septic patient-day).
That is not a clinically usable alert burden. Future work: choose the threshold under a
false-alarm budget (for example at most 0.5 alerts per patient-day) and report that operating
point next to the utility-optimal one; consider alert de-duplication (refractory period).

## SageMaker Clarify (Phase 9)
Explainability and subgroup results were produced with XGBoost's exact TreeSHAP and the project's
own evaluator (`sepsis.explain`), not SageMaker Clarify: Clarify is first on the plan's cut list,
needs extra processing-instance quota and VPC wiring, and the fallback is the plan's stated
alternative. Future work: run Clarify SHAP and bias reports as a pipeline step and compare.

## From Phase 8 (deployment)

- Make the pipeline's Evaluate step emit the serving bundle (model + `evaluation.json`) and register
  that, so deployment needs no extra bundling step and the registered image is the deployed image.
  Registered package v2 references an OCI-manifest image that CreateModel rejects; rebuild with the
  fixed `push_image.sh` and re-register to remove the `--image-tag` override.
- Serverless endpoints with VPC and network isolation: revisit if AWS lifts the restriction; until
  then use an instance-based endpoint if the VPC boundary is required.
- Batch Transform parity was checked on a 1,000-patient sample for speed; the full 20,000-patient
  hospital B run is a larger version of the same script (`--n-patients 20000`).
