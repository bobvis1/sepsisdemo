output "region" { value = var.region }
output "bucket" { value = aws_s3_bucket.data.bucket }
output "kms_key_arn" { value = aws_kms_key.this.arn }
output "role_arn" { value = aws_iam_role.sagemaker.arn }
output "subnet_ids" { value = aws_subnet.private[*].id }
output "security_group_id" { value = aws_security_group.jobs.id }
output "ecr_repository_url" { value = aws_ecr_repository.main.repository_url }
output "mlflow_tracking_server_arn" {
  value = var.enable_mlflow ? aws_sagemaker_mlflow_tracking_server.this[0].arn : null
}
output "model_package_group_name" { value = aws_sagemaker_model_package_group.this.model_package_group_name }
