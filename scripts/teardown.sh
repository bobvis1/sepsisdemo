#!/usr/bin/env bash
# Delete everything this project created. DESTRUCTIVE: asks for confirmation.
# 1) stop/delete SageMaker endpoints tagged project=sepsis-sagemaker  2) terraform destroy
# 3) list any tagged resources still remaining.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
PROFILE=$(uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml'))['aws_profile'])")
REGION=$(uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml'))['region'])")
export AWS_PROFILE="${SEPSIS_AWS_PROFILE:-$PROFILE}" AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

read -r -p "Delete ALL sepsis-sagemaker resources in $REGION (profile $AWS_PROFILE)? type 'yes': " ok
[ "$ok" = "yes" ] || { echo aborted; exit 1; }

# Endpoints are not Terraform-managed (created on demand by the SDK), so delete them first.
for ep in $(aws sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text); do
  arn=$(aws sagemaker describe-endpoint --endpoint-name "$ep" --query EndpointArn --output text)
  if aws sagemaker list-tags --resource-arn "$arn" --query "Tags[?Key=='project'].Value" --output text | grep -q sepsis-sagemaker; then
    echo "deleting endpoint $ep"; aws sagemaker delete-endpoint --endpoint-name "$ep"
  fi
done

scripts/infra.sh destroy -auto-approve

echo "Tagged resources still present (expect none):"
aws resourcegroupstaggingapi get-resources --tag-filters Key=project,Values=sepsis-sagemaker \
  --query 'ResourceTagMappingList[].ResourceARN' --output text
