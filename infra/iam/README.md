# Deployer IAM policy

`deployer-policy.json` is a suggested identity policy for the human/agent identity that runs
`scripts/infra.sh` and the SageMaker SDK scripts. Do not use the account root user.

Create it (account-specific, so not scripted in Terraform to avoid a chicken-and-egg problem):

    aws iam create-policy --policy-name sepsis-deployer --policy-document file://infra/iam/deployer-policy.json

Then attach it to an IAM Identity Center permission set or an IAM user/role used with `aws login`.
The Billing console and Budgets email need no extra permissions.

## What it allows
- EC2/VPC, KMS, SageMaker: any action, only in `us-east-2` (EC2 and KMS resource ARNs do not exist
  before creation, so these cannot be scoped tighter by resource).
- S3: only buckets named `sepsis-*`. ECR: only repositories under `sepsis/`.
- IAM: create/modify only roles named `sepsis-*`, inline policies only, and pass them only to SageMaker.
- Budgets named `sepsis-*`; read-only cost explorer.

## What it denies
- Any use of regional services outside `us-east-2`.
- Creating users or access keys, attaching managed policies, editing policy versions, Organizations.

## Known limits (honest trade-offs)
- A holder can still write an inline admin policy into a `sepsis-*` role and pass it to a job. A
  permissions boundary on those roles would close this; listed as future work.
- If the region in `config/settings.yaml` changes, edit `us-east-2` in this file.
- If Terraform reports AccessDenied, add the one missing action; do not widen to `*`.
