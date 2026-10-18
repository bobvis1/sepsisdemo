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
