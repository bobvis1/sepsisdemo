#!/usr/bin/env bash
# Delete everything this project created. DESTRUCTIVE: asks for confirmation.
# 1) delete SDK-created SageMaker resources tagged project=sepsis-sagemaker (endpoints, endpoint
#    configs, models, pipelines, model packages)  2) terraform destroy  3) list what remains
# 3) list any tagged resources still remaining.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
PROFILE=$(uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml'))['aws_profile'])")
REGION=$(uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml'))['region'])")
export AWS_PROFILE="${SEPSIS_AWS_PROFILE:-$PROFILE}" AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

read -r -p "Delete ALL sepsis-sagemaker resources in $REGION (profile $AWS_PROFILE)? type 'yes': " ok
[ "$ok" = "yes" ] || { echo aborted; exit 1; }

# The SDK (not Terraform) creates endpoints, endpoint configs, models, pipelines and model
# packages. Delete the ones tagged project=sepsis-sagemaker first; a non-empty model package group
# would also make `terraform destroy` fail.
is_ours() {  # $1 = resource ARN
  aws sagemaker list-tags --resource-arn "$1" --query "Tags[?Key=='project'].Value" --output text 2>/dev/null |
    grep -q sepsis-sagemaker
}

for ep in $(aws sagemaker list-endpoints --query 'Endpoints[].EndpointName' --output text); do
  arn=$(aws sagemaker describe-endpoint --endpoint-name "$ep" --query EndpointArn --output text)
  is_ours "$arn" && { echo "deleting endpoint $ep"; aws sagemaker delete-endpoint --endpoint-name "$ep"; }
done
for cfg in $(aws sagemaker list-endpoint-configs --query 'EndpointConfigs[].EndpointConfigName' --output text); do
  arn=$(aws sagemaker describe-endpoint-config --endpoint-config-name "$cfg" --query EndpointConfigArn --output text)
  is_ours "$arn" && { echo "deleting endpoint config $cfg"; aws sagemaker delete-endpoint-config --endpoint-config-name "$cfg"; }
done
for m in $(aws sagemaker list-models --query 'Models[].ModelName' --output text); do
  arn=$(aws sagemaker describe-model --model-name "$m" --query ModelArn --output text)
  is_ours "$arn" && { echo "deleting model $m"; aws sagemaker delete-model --model-name "$m"; }
done
for p in $(aws sagemaker list-pipelines --pipeline-name-prefix sepsis --query 'PipelineSummaries[].PipelineName' --output text); do
  echo "deleting pipeline $p"; aws sagemaker delete-pipeline --pipeline-name "$p" --client-request-token "teardown-$p-$(date +%s)"
done
GROUP="$(uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml')).get('name_prefix','sepsis'))")-models"
for pkg in $(aws sagemaker list-model-packages --model-package-group-name "$GROUP" --query 'ModelPackageSummaryList[].ModelPackageArn' --output text 2>/dev/null); do
  echo "deleting model package $pkg"; aws sagemaker delete-model-package --model-package-name "$pkg"
done

scripts/infra.sh destroy -auto-approve

echo "Tagged resources still present (expect none; finished training/processing/transform job records cannot be deleted and may be listed):"
aws resourcegroupstaggingapi get-resources --tag-filters Key=project,Values=sepsis-sagemaker \
  --query 'ResourceTagMappingList[].ResourceARN' --output text
