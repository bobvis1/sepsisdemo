# Fast path from code change to AWS. `make help` lists targets.
PY := uv run python

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'
setup:           ## create venv with pinned deps
	uv sync --extra dev
lint:            ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .
test:            ## unit tests (local, no AWS)
	uv run pytest -q
tf-plan:         ## terraform plan (free)
	scripts/infra.sh plan
tf-apply:        ## terraform apply (BILLABLE, ask first)
	scripts/infra.sh apply
whoami:          ## verify AWS credentials for the configured profile
	$(PY) scripts/whoami.py
smoke:           ## trivial isolated Processing job in the VPC (Phase 1 acceptance)
	$(PY) scripts/smoke_processing_job.py
.PHONY: help setup lint test tf-plan tf-apply whoami smoke
