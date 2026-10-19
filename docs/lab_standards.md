# Lab standards for ML work on AWS (one page)

These are the conventions this project follows. Copy them for the next project.

**Naming.** Everything starts with `sepsis-` (prefix from `config/settings.yaml`): IAM roles
(`sepsis-sagemaker-execution`, `sepsis-mlflow-server`), the bucket (`sepsis-<account>-<region>`), ECR
repo `sepsis/main`, pipeline `sepsis-pipeline`, model group `sepsis-models`. The deployer IAM policy
is scoped by this prefix, so a mistyped name fails closed rather than touching other projects.

**Tagging.** Every resource and every job carries `project=sepsis-sagemaker`, `owner=<from config>`,
`env=demo`. Terraform sets them via provider `default_tags`; SDK jobs set them in
`sepsis.sm_common.tags`. `scripts/teardown.sh` finds what to delete by the `project` tag.

**Encryption.** One customer-managed KMS key encrypts the bucket (SSE-KMS, versioned, public access
blocked, TLS required), job volumes, job outputs, model artifacts and the ECR repository. Uploads
name the key explicitly.

**Network isolation.** Jobs run in private subnets with no NAT gateway and no internet route; only
an S3 gateway endpoint exists. Every job sets `enable_network_isolation=True`, so containers cannot
make network calls and **cannot `pip install` at runtime: bake every dependency into the image.**
Add an interface VPC endpoint only after a job proves it needs one, and record its hourly cost
(`interface_endpoints` in settings).

**Identity.** Humans and agents use SSO or short-lived credentials via `aws login`, never root and
never committed keys. The deployer policy (`infra/iam/deployer-policy.json`) is least-privilege and
region-pinned. Jobs use a separate execution role limited to the project bucket, key, logs and job
APIs. No account IDs, ARNs or credentials are committed; they come from `config/` (git-ignored).

**Cost.** Notify when creating anything billable. Prefer Spot training, serverless inference and
Batch Transform over always-on endpoints. Optional hourly services (managed MLflow) are behind a
flag that defaults to off. A monthly AWS Budgets alarm is part of the base stack. Request instance
quotas early: new accounts often have zero training and transform quota, and approval can take a day.

**Reproducibility.** Python and AWS SDK versions are pinned (`pyproject.toml`, `uv.lock`,
`container/requirements.txt`); the SageMaker SDK is v3 and **not** compatible with v2 examples. Tag
images with the git SHA (`scripts/push_image.sh`) and pin the SHA when running a pipeline. Every
script runs locally on a sample before it runs on SageMaker.

**Adding a new model to the pipeline.**
1. Add a branch to `src/sepsis/train.py` (`--model` choice) that saves `model.joblib`
   (`{"model", "columns"}`), `params.json` and `metrics.json`, and a unit test.
2. Add it to the model loop in `src/sepsis/pipeline.py` (a Train and an Evaluate step are created
   per model) and, if it should be a registration candidate, to the condition.
3. Re-evaluate with `sepsis.evaluate`: calibration and the threshold are fit on hospital A
   validation only; hospital B is never used for any choice.
4. Push the image, upsert the pipeline, review `evaluation.json` and the model package, then approve
   it manually in the Model Registry.

**Teardown.** `scripts/teardown.sh` deletes SDK-created endpoints, endpoint configs, models, the
pipeline and model packages, then runs `terraform destroy`, then lists any remaining tagged
resources. Finished job records cannot be deleted and may still be listed.
