# PLAN.md: Sepsis early warning on SageMaker (PhysioNet 2019)

## Goal

Produce a reproducible, secure SageMaker pipeline that trains an hourly sepsis risk model on hospital A, evaluates it on hospital B as an external site, registers it conditionally, and serves it. Budget: about 20 hours of the user's time.

## Deliverables

1. Working code in this repo, runnable end to end from documented commands.
2. A model that outputs, for each patient-hour, the probability of meeting sepsis criteria within roughly the next 6 hours.
3. An evaluation report comparing hospital A validation with hospital B, with patient-level bootstrap 95% confidence intervals.
4. A SageMaker Pipeline with conditional model registration.
5. A deployed serverless endpoint (created on demand, then torn down) and a Batch Transform run over hospital B.
6. `README.md` with an architecture diagram, `docs/model_card.md`, `docs/lab_standards.md`, and `scripts/teardown.sh`.

## Repository layout (target)

```
CLAUDE.md
PLAN.md
README.md
pyproject.toml
config/settings.example.yaml
infra/                      # Terraform for VPC, endpoints, S3, KMS, IAM
container/                  # Dockerfile + requirements for processing/training image
src/sepsis/
  io.py                     # read .psv, write/read Parquet
  splits.py                 # patient-level splits
  features.py               # causal feature engineering
  train.py                  # training entry point (baseline + XGBoost)
  evaluate.py               # metrics, bootstrap CIs, report
  metrics.py                # lead time, false alarms/patient-day, wrappers
  inference.py              # endpoint handler
  pipeline.py               # SageMaker Pipeline definition
third_party/evaluation_2019/  # vendored official scoring script, with license/attribution
scripts/
  download_data.sh
  consolidate.py
  teardown.sh
tests/
docs/
  model_card.md
  lab_standards.md
  future_work.md
```

## Architecture decisions (already made)

- **Drive SageMaker from the local machine with the Python SDK**, not from Studio. Note Studio VPC-only mode in `docs/future_work.md`.
- **Infrastructure as code with Terraform** in `infra/`, so setup and teardown are reproducible.
- **Jobs run inside a private VPC** (private subnets, S3 gateway endpoint, no NAT gateway) with network isolation and KMS encryption of S3, job volumes, and model artifacts.
- **One custom container image** in ECR for processing and the evaluation step. Training may use the SageMaker XGBoost framework container in script mode to save time; using the custom image for training is a stretch goal.
- **Models:** logistic regression baseline and XGBoost. No deep learning in the core plan.

## Phases

Hours are the user's time.

### Phase 1: Secure AWS environment (2 h)
- Terraform: VPC with two private subnets, S3 gateway endpoint, security group for SageMaker jobs, KMS key, S3 bucket (SSE-KMS, public access blocked, versioning on), least-privilege SageMaker execution role, ECR repository, AWS Budgets alarm.
- Check whether any SageMaker job type in the plan needs interface endpoints (for example CloudWatch Logs or STS) when run in this VPC; add only what is required and record the cost.
- **Acceptance:** `terraform plan` is clean; a trivial Processing job runs in the VPC with network isolation and writes an encrypted object to S3.

### Phase 2: Ingestion and consolidation (2 h)
- `scripts/download_data.sh` downloads both training sets from PhysioNet.
- `scripts/consolidate.py` merges per-patient files into Parquet partitioned by hospital, adding `patient_id` and `hospital` columns. Upload to S3.
- Short exploration notebook or script: missingness by variable and hospital, label prevalence by patient and by hour, length-of-stay distribution, count of patients positive at their first ICU hour.
- **Acceptance:** row and patient counts reconcile with the raw files; exploration summary saved to `docs/data_summary.md`.

### Phase 3: Causal feature engineering in a custom container (3 h)
- `container/Dockerfile` with pinned dependencies; push to ECR.
- `features.py`: forward-filled values, missingness indicators, hours since last measurement per variable, rolling mean and slope over 6 h and 12 h windows for vitals, ICULOS, demographics.
- `splits.py`: hospital A train/validation split by patient (80/20, stratified on ever-septic); hospital B held out.
- Run as a SageMaker Processing job using the custom image.
- **Required tests:**
  - Causality: changing a patient's rows after hour t does not change any feature at hours ≤ t.
  - Split integrity: no patient appears in more than one split; no hospital B patient in train or validation.
- **Acceptance:** Processing job output matches a local run on the same sample.

### Phase 4: Training (3 h)
- `train.py` trains the logistic regression baseline (with scaler fit on training fold only) and XGBoost, handling class imbalance explicitly (for example `scale_pos_weight`), and logs parameters and metrics.
- Run as SageMaker training jobs with Spot and network isolation.
- Experiment tracking: use SageMaker's managed MLflow if its current pricing fits the budget; otherwise log to S3 and note the choice. Confirm cost with the user first.
- **Acceptance:** both models train on SageMaker; validation AUROC and AUPRC logged.

### Phase 5: Hyperparameter tuning (1.5 h)
- Bayesian tuning job for XGBoost, objective AUPRC on the hospital A validation split, small budget (for example 20 jobs, 4 in parallel).
- **Acceptance:** best job identified and its parameters recorded.

### Phase 6: Evaluation with confidence intervals (1.5 h)
- `evaluate.py` produces, for hospital A validation and hospital B:
  - AUROC, AUPRC, calibration curve and Brier score.
  - Official challenge utility score at a threshold chosen on hospital A validation.
  - Median lead time from first alert to clinical onset among septic patients.
  - False alarms per patient-day among non-septic patients.
  - 95% percentile CIs for each metric from 1,000 bootstrap resamples of patients.
- Output: `evaluation.json` (consumed by the Pipeline) and a Markdown report.
- **Required test:** bootstrap resamples patients, not rows.

### Phase 7: Pipeline with conditional registration (2.5 h)
- `pipeline.py`: Processing (features) → Training → Evaluation → Condition (XGBoost validation AUPRC exceeds baseline) → Register in Model Registry with status `PendingManualApproval`.
- **Acceptance:** one full pipeline execution succeeds from a single command; the registered model package references `evaluation.json`.

### Phase 8: Deployment (2 h)
- `inference.py` handler: accepts a patient's hourly history (JSON), computes features with the same `features.py` code, returns the latest-hour risk score.
- Deploy the approved model to a serverless endpoint; invoke it with a sample patient; then delete it.
- Batch Transform over hospital B; confirm scores match `evaluate.py` output.
- **Acceptance:** endpoint invocation documented in README with sample request/response; endpoint deleted.

### Phase 9: Explainability and subgroups (1 h)
- SageMaker Clarify for SHAP attributions and performance by age band and sex. If Clarify setup in the VPC runs long, fall back to XGBoost's built-in SHAP values and note it.

### Phase 10: Documentation and teardown (1.5 h)
- `README.md`: purpose, architecture diagram (Mermaid), how to run each phase, results table (A vs B with CIs), cost incurred, limitations.
- `docs/model_card.md`: intended use, data, metrics, subgroup results, limitations (retrospective data, label definition, not for clinical use).
- `docs/lab_standards.md`: one page covering naming, tagging, encryption, network isolation, how to add a new model to the pipeline, teardown.
- `scripts/teardown.sh` verified by running it and confirming no tagged resources remain (except the Terraform state bucket if used).

## Cut order if needed

1. Phase 9 (Clarify)
2. Hyperparameter tuning (use defaults plus a small manual grid instead)
3. Serverless endpoint (keep Batch Transform)
4. MLflow (log to S3)

Never cut: patient-level splits, causal features, hospital B external evaluation, bootstrap CIs, network isolation.

## Out of scope

Other datasets, deep sequence models, per-prediction uncertainty intervals (conformal prediction or ensembles), Studio VPC-only mode, CI/CD. List others in `docs/future_work.md` if they come up.
