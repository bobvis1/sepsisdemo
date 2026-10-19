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
smoke:           ## trivial isolated Processing job in the VPC (smoke test)
	$(PY) scripts/smoke_processing_job.py
showcase:        ## build the results page and its data into outputs/showcase (local, free)
	$(PY) -m sepsis.showcase --model-dir outputs/pipe2/model --evaluation outputs/final/eval_xgboost \
	    --features data/features --template showcase/template.html --out outputs/showcase
showcase-test: showcase  ## rebuild, then browser tests of the page (Node 22.12+ and Chrome)
	PATH="$$HOME/.local/share/fnm/aliases/default/bin:$$PATH" npm --prefix showcase/tests ci
	PATH="$$HOME/.local/share/fnm/aliases/default/bin:$$PATH" npm --prefix showcase/tests test
.PHONY: help setup lint test tf-plan tf-apply whoami smoke showcase showcase-test
