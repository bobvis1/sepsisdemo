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

# Log in first: buildx pushes straight from the build. Docker Desktop's containerd store would
# otherwise push OCI manifests, which SageMaker CreateModel rejects ("Unsupported manifest media
# type application/vnd.oci.image.manifest.v1+json"), so request Docker v2 media types explicitly.
aws --profile "$PROFILE" --region "$REGION" ecr get-login-password |
  docker login --username AWS --password-stdin "${REPO%%/*}" >/dev/null
docker buildx build -q --platform linux/amd64 --provenance=false --sbom=false \
  --output type=registry,oci-mediatypes=false -t "$REPO:$SHA" -t "$REPO:latest" \
  -f container/Dockerfile . >/dev/null
echo "$REPO:$SHA"
