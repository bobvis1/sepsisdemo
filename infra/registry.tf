# SageMaker Model Registry group the Pipeline registers into (free; no hourly cost).
resource "aws_sagemaker_model_package_group" "this" {
  model_package_group_name        = "${var.name_prefix}-models"
  model_package_group_description = "Sepsis early-warning models (PhysioNet 2019)"
}
