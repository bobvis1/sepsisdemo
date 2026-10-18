# SageMaker managed MLflow tracking server (Phase 4+). Created only when enable_mlflow = true.
# Billed per hour while it exists (Small is ~$0.60/hr): apply for a working session, destroy after.
# Artifacts go under s3://<bucket>/mlflow/ in the project bucket, encrypted with the project KMS key.
# Training jobs run with network isolation and cannot reach MLflow; metrics are logged from the
# orchestrating side (scripts / pipeline) after the job, not from inside the container.

locals {
  mlflow_name = "${var.name_prefix}-mlflow"
}

data "aws_iam_policy_document" "mlflow_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "mlflow" {
  statement {
    sid       = "ArtifactPrefix"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.data.arn}/mlflow/*"]
  }
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["mlflow/*", "mlflow"]
    }
  }
  statement {
    sid       = "ProjectKey"
    actions   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:DescribeKey"]
    resources = [aws_kms_key.this.arn]
  }
  # Lets MLflow auto-register models into the SageMaker Model Registry.
  statement {
    sid = "ModelRegistry"
    actions = [
      "sagemaker:AddTags", "sagemaker:CreateModelPackageGroup", "sagemaker:CreateModelPackage",
      "sagemaker:UpdateModelPackage", "sagemaker:DescribeModelPackageGroup",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role" "mlflow" {
  count              = var.enable_mlflow ? 1 : 0
  name               = "${var.name_prefix}-mlflow-server"
  assume_role_policy = data.aws_iam_policy_document.mlflow_trust.json
}

resource "aws_iam_role_policy" "mlflow" {
  count  = var.enable_mlflow ? 1 : 0
  name   = "mlflow-server"
  role   = aws_iam_role.mlflow[0].id
  policy = data.aws_iam_policy_document.mlflow.json
}

resource "aws_sagemaker_mlflow_tracking_server" "this" {
  count                        = var.enable_mlflow ? 1 : 0
  tracking_server_name         = local.mlflow_name
  role_arn                     = aws_iam_role.mlflow[0].arn
  artifact_store_uri           = "s3://${aws_s3_bucket.data.bucket}/mlflow"
  tracking_server_size         = "Small"
  automatic_model_registration = false
  depends_on                   = [aws_iam_role_policy.mlflow]
}

# Callers (the execution role used by jobs/pipelines) may log to this server.
data "aws_iam_policy_document" "mlflow_client" {
  count = var.enable_mlflow ? 1 : 0
  statement {
    sid       = "LogToMlflow"
    actions   = ["sagemaker-mlflow:*"]
    resources = [aws_sagemaker_mlflow_tracking_server.this[0].arn]
  }
}

resource "aws_iam_role_policy" "mlflow_client" {
  count  = var.enable_mlflow ? 1 : 0
  name   = "mlflow-client"
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.mlflow_client[0].json
}
