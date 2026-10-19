# CLAUDE.md

Instructions for Claude Code working in this repository. Read PLAN.md before starting any phase.

## Project in one paragraph

A time-boxed (about 20 hours of human time) demonstration project: an hourly sepsis early-warning model built on the PhysioNet/Computing in Cardiology Challenge 2019 dataset, trained, evaluated, registered, and deployed through Amazon SageMaker in a locked-down AWS environment. The purpose is to demonstrate hands-on SageMaker, AWS, Docker, and secure-environment practice for a job application to a clinical ML research lab. The pipeline and its engineering practices matter more than squeezing out model performance.

## Dataset: fixed decision

The PhysioNet 2019 challenge data is the only dataset for this project.

- Source: https://physionet.org/content/challenge-2019/1.0.0/ (open access, CC BY 4.0; no credentialing needed)
- Two training sets: `training_setA` (about 20,336 patients) and `training_setB` (about 20,000 patients), from different hospital systems. One pipe-delimited `.psv` file per patient, one row per ICU hour.
- 40 variables plus `SepsisLabel`: 8 vitals (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2), 26 labs, and 6 demographic/administrative fields (Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS).
- `SepsisLabel` is already shifted: for septic patients it is 1 from 6 hours before clinical onset onward. Do not shift it again. Clinical onset time for a septic patient = first hour with label 1, plus 6.
- The challenge's hidden test set was never released. **Hospital B is the external test set.**
- Official scoring code: https://github.com/physionetchallenges/evaluation-2019 (`evaluate_sepsis_score.py`, utility score). Vendor it with attribution rather than reimplementing the utility function.

## Non-negotiable methodological rules

1. **Split by patient, never by row.** Hospital A is split into train and validation by patient, stratified by whether the patient ever has a positive label. Hospital B is used only for final evaluation.
2. **Nothing is tuned on hospital B.** Hyperparameters, alert threshold, feature choices, and calibration are all fit on hospital A only.
3. **Features must be causal.** A feature at hour t may use only rows at hours ≤ t for that patient. Forward-fill is allowed; back-fill, interpolation across future points, and patient-level aggregates over the full stay are not.
4. **Any fitted preprocessing** (scalers, imputation constants, encoders) is fit on the hospital A training fold only.
5. **Confidence intervals** on evaluation metrics come from a bootstrap that resamples patients, not rows.

## AWS and cost rules

- **Tell the user about charges; explicit approval is not required.** The overall AWS budget limit (see `config/settings.yaml`) is the guardrail. When you run a command that creates billable AWS resources (Terraform apply, endpoint creation, tuning jobs, MLflow tracking servers, interface VPC endpoints), give a one-line cost estimate as you do it, and say when it is stopped or deleted. Still ask before destructive actions (Terraform destroy, `scripts/teardown.sh`, deleting data), and stay within the budget.
- Create resources through IaC, not ad-hoc command line. Someone else should be able to grab this repo, point it to their own AWS account, and spin everything up.
- Prefer serverless inference and Batch Transform over real-time endpoints.
- Use managed Spot training where the job type supports it.
- Every resource gets tags: `project=sepsis-sagemaker`, `owner=<from config>`, `env=demo`.
- Region and AWS profile come from config (`config/settings.yaml` or env vars).
- Never commit credentials, account IDs, or ARNs to the repo.
- Maintain a `scripts/teardown.sh`.

## SageMaker SDK

The SageMaker Python SDK has changed significantly across major versions. Check the installed version (`pip show sagemaker`) and consult current AWS documentation before writing SDK code. Do not rely on remembered API signatures. Pin the version in `pyproject.toml` once chosen and note it in the README. Save documentation that will be referenced repeatedly in a local folder that is in the .gitignore.

Training and processing jobs run with `enable_network_isolation=True` where the job type supports it. That means runtime `pip install` will fail inside the job: all dependencies must be baked into the container image.

## Engineering conventions

- Python 3.11, package code under `src/sepsis/`, managed with `pyproject.toml`.
- Format and lint with ruff; type hints on public functions.
- Tests with pytest under `tests/`. Required tests are listed in PLAN.md.
- **Local first:** every processing and training script must run locally on a small sample (for example 500 patients) before it is run on SageMaker.
- Data files, model artifacts, and Terraform state are git-ignored.
- Prefer small, readable modules. This repo is meant to be read by reviewers.

## Working style

- Work phase by phase from PLAN.md. At the end of each phase, report what was done, how it was verified, and hours-relevant surprises, then wait for the user before starting the next phase.
- If a phase is running long, say so and propose what to cut, using the cut order in PLAN.md.
- When a choice is between something more impressive and something more likely to finish, choose the one more likely to finish and note the alternative in `docs/future_work.md`.

<!-- BEGIN AWS Agent Toolkit rules -->
# AWS Guidance

- Where these AWS rules conflict with the project's own instructions, the
  project's instructions take precedence.
- Prefer the AWS MCP Server for AWS interactions — it provides sandboxed
  execution, observability, and audit logging. If unavailable, use the
  AWS CLI directly.
- Before starting a task, check whether a relevant AWS skill is available.
  Load the skill with `retrieve_skill` and prefer its guidance over
  general knowledge.
- When uncertain about specific AWS details (API parameters, permissions,
  limits, error codes), verify against documentation rather than guessing.
  State uncertainty explicitly if you cannot confirm.
- When creating infrastructure, prefer infrastructure-as-code (AWS CDK or
  CloudFormation) over direct CLI commands.
- When working with infrastructure, follow AWS Well-Architected Framework
  principles.
- Do not use em dashes in AWS resource names or descriptions. Use
  hyphens instead.

## Secret Safety

- MUST load the `aws-secrets-manager` skill first for any secret,
  credential, API key, token, or password task. MUST NOT call
  `secretsmanager get-secret-value` or `batch-get-secret-value`, and MUST
  NOT hit the Secrets Manager Agent daemon directly. MUST use
  `{{resolve:secretsmanager:secret-id:SecretString:json-key}}` with
  `asm-exec` so the secret resolves at runtime without entering context.
<!-- END AWS Agent Toolkit rules -->
