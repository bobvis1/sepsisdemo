variable "region" { type = string }
variable "aws_profile" { type = string }
variable "owner" { type = string }

variable "name_prefix" {
  type    = string
  default = "sepsis"
}

variable "budget_usd" { type = number }
variable "budget_email" { type = string }

variable "interface_endpoints" {
  description = "Interface VPC endpoints to create (billable, ~$0.01/hr/AZ each). Empty by default."
  type        = list(string)
  default     = []
  validation {
    condition = alltrue([for e in var.interface_endpoints :
    contains(["logs", "sts", "ecr.api", "ecr.dkr", "sagemaker.api", "sagemaker.runtime"], e)])
    error_message = "Unsupported interface endpoint name."
  }
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "enable_mlflow" {
  description = "Create the SageMaker managed MLflow tracking server (billable by the hour while it exists, ~$0.60/hr for Small). Off by default."
  type        = bool
  default     = false
}
