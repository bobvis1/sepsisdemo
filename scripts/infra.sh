#!/usr/bin/env bash
# One entry point for infrastructure: scripts/infra.sh {init|plan|apply|outputs|destroy}
# Renders infra/terraform.tfvars from config/settings.yaml, so settings.yaml is the single source.
# apply/destroy create or delete billable resources: Claude must ask the user first (CLAUDE.md).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

[ -f config/settings.yaml ] || { echo "Copy config/settings.example.yaml to config/settings.yaml and edit it." >&2; exit 1; }

uv run python - <<'PY'
import json, yaml
s = yaml.safe_load(open("config/settings.yaml"))
v = {k: s[k] for k in ("region", "owner", "name_prefix")}
v["aws_profile"] = s["aws_profile"]
v["budget_usd"] = s["budget_usd"]
v["budget_email"] = s["budget_email"]
v["interface_endpoints"] = s.get("interface_endpoints", [])
open("infra/terraform.tfvars.json", "w").write(json.dumps(v, indent=2))
PY

PROFILE=$(uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml'))['aws_profile'])")
export AWS_PROFILE="${SEPSIS_AWS_PROFILE:-$PROFILE}"

cmd="${1:-plan}"; shift || true
case "$cmd" in
  init)    terraform -chdir=infra init -input=false "$@" ;;
  plan)    terraform -chdir=infra init -input=false >/dev/null; terraform -chdir=infra plan -input=false "$@" ;;
  apply)   terraform -chdir=infra apply -input=false "$@"; "$0" outputs ;;
  outputs) terraform -chdir=infra output -json > config/outputs.json; echo "wrote config/outputs.json" ;;
  destroy) terraform -chdir=infra destroy -input=false "$@" ;;
  *) echo "usage: $0 {init|plan|apply|outputs|destroy}" >&2; exit 2 ;;
esac
