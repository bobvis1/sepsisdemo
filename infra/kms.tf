# One customer-managed key for S3, job volumes, model artifacts, and ECR.
resource "aws_kms_key" "this" {
  description             = "${var.name_prefix} data key"
  enable_key_rotation     = true
  deletion_window_in_days = 7
}

resource "aws_kms_alias" "this" {
  name          = "alias/${var.name_prefix}"
  target_key_id = aws_kms_key.this.key_id
}
