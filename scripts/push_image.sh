#!/usr/bin/env bash
# Build the amd64 image and push it to the project's ECR repo as :<git-sha> and :latest.
# Profile, region and repo URL come from config/ (git-ignored); nothing account-specific is committed.
# Usage: scripts/push_image.sh      Prints the pushed image URI (git-sha tag) on the last line.
set -euo pipefail
export PATH="$HOME/.local/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"
cd "$(dirname "$0")/.."

read -r PROFILE REGION REPO < <(uv run python - <<'PY'
from sepsis.config import load_infra, load_settings
s, i = load_settings(), load_infra()
print(s.aws_profile, s.region, i.ecr_repository_url)
PY
)
SHA="$(git rev-parse --short HEAD)"
[[ -z "$(git status --porcelain src container pyproject.toml)" ]] || SHA="${SHA}-dirty"

docker build -q --platform linux/amd64 --provenance=false --sbom=false \
  -f container/Dockerfile -t "$REPO:$SHA" -t "$REPO:latest" .
aws --profile "$PROFILE" --region "$REGION" ecr get-login-password |
  docker login --username AWS --password-stdin "${REPO%%/*}" >/dev/null
docker push -q "$REPO:$SHA" >/dev/null
docker push -q "$REPO:latest" >/dev/null
echo "$REPO:$SHA"
