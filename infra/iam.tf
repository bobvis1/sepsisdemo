data "aws_iam_policy_document" "trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${var.name_prefix}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.trust.json
}

data "aws_iam_policy_document" "sagemaker" {
  statement {
    sid       = "ProjectBucket"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:GetBucketLocation", "s3:AbortMultipartUpload"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
  }

  statement {
    sid       = "ProjectKey"
    actions   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:DescribeKey", "kms:CreateGrant"]
    resources = [aws_kms_key.this.arn]
  }

  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  # Pull-only. Covers our repo and the AWS-owned framework images (e.g. SageMaker XGBoost).
  statement {
    sid = "EcrPull"
    actions = [
      "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability",
    ]
    resources = ["*"]
  }

  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"]
  }

  statement {
    sid       = "Metrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/ProcessingJobs", "/aws/sagemaker/TrainingJobs", "/aws/sagemaker/Endpoints"]
    }
  }

  # Required for jobs/endpoints attached to a VPC.
  statement {
    sid = "VpcAttach"
    actions = [
      "ec2:CreateNetworkInterface", "ec2:CreateNetworkInterfacePermission",
      "ec2:DeleteNetworkInterface", "ec2:DeleteNetworkInterfacePermission",
      "ec2:DescribeNetworkInterfaces", "ec2:DescribeVpcs", "ec2:DescribeDhcpOptions",
      "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups",
    ]
    resources = ["*"]
  }

  # Pipelines / tuning launch jobs, models, endpoints, and registry entries as this role.
  statement {
    sid = "SageMakerWork"
    actions = [
      "sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob", "sagemaker:StopProcessingJob",
      "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob",
      "sagemaker:CreateHyperParameterTuningJob", "sagemaker:DescribeHyperParameterTuningJob",
      "sagemaker:StopHyperParameterTuningJob", "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
      "sagemaker:CreateTransformJob", "sagemaker:DescribeTransformJob", "sagemaker:StopTransformJob",
      "sagemaker:CreateModel", "sagemaker:DescribeModel", "sagemaker:DeleteModel",
      "sagemaker:CreateModelPackage", "sagemaker:DescribeModelPackage",
      "sagemaker:CreateModelPackageGroup", "sagemaker:DescribeModelPackageGroup",
      "sagemaker:UpdateModelPackage", "sagemaker:ListModelPackages",
      "sagemaker:AddTags", "sagemaker:ListTags",
      "sagemaker:CreateExperiment", "sagemaker:CreateTrial", "sagemaker:CreateTrialComponent",
      "sagemaker:UpdateTrialComponent", "sagemaker:AssociateTrialComponent", "sagemaker:DescribeTrialComponent",
      "sagemaker:Search",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "PassSelf"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "sagemaker" {
  name   = "project-access"
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.sagemaker.json
}
