# Showcase browser tests

Real mouse, keyboard and rendering checks of the built results page, driven by
[`puppeteer-core`](https://pptr.dev/) against the Chrome you already have installed (no browser is
downloaded). They run `node:test`; the only dependency is `puppeteer-core`, pinned to an exact
version, with install scripts disabled (`.npmrc`). `npm audit`: 0 vulnerabilities.

## Run

```bash
# Node 22.12 or newer. On this Mac it is off the default PATH (installed with fnm):
export PATH=$HOME/.local/share/fnm/aliases/default/bin:$PATH

# 1. build the page and its data (Python, writes outputs/showcase/)
uv run python -m sepsis.showcase --model-dir outputs/pipe2/model \
    --evaluation outputs/final/eval_xgboost --features data/features \
    --template showcase/template.html --out outputs/showcase

# 2. install and test
npm --prefix showcase/tests ci
npm --prefix showcase/tests test            # about 10 seconds
```

The tests open `outputs/showcase/standalone.html` (the page inside a doctype wrapper; `index.html`
is the fragment that gets published as an Artifact) and read the expected numbers from
`outputs/showcase/data.json`. If Chrome is not at the macOS default path, set `CHROME_PATH`; the
tests fail with a clear message instead of skipping.

## What is covered

| Test | Proves |
|---|---|
| 1 | The page loads with no page errors or `console.error`, with and without the intro animation |
| 2 | Counters equal the hospital A threshold row of `data.json`; hero numbers equal `evaluation.json`; the dots the canvas drew per category equal the counters |
| 3 | A real mouse drag moves the threshold; counts only move the right way; every counter equals the grid row for the new threshold, both directions |
| 4, 5 | Arrow keys move exactly one grid point each; reset and the two operating-point buttons snap correctly |
| 6, 7 | Hospital A / B toggle; URL hooks (`t`, `cohort`, `still`) land in the same state as clicking there |
| 8 | All 10 replay patients play to the end and show the banner for their category (hours of warning equal `lead_h`), the alert tag when an alert fired, and their drivers |
| 9 | A slow real mouse click on Pause stops the replay, the hour holds, and Play resumes it |
| 10 | The replay's buttons and labels are not rewritten every frame (rewriting a button's text during a click can swallow it) |
| 11 | The night shift is reproducible per seed, matches an **independent re-implementation** of the simulation in `helpers.js`, and a higher threshold never adds alert hours |
| 12 | Reduced motion skips the intro; dark mode changes tokens and background, not the data |
| 13 | The swimmer wall has one row per septic patient and the never-alerted count matches the grid |
| 14 | Saves review screenshots to `outputs/showcase/screens/` (git-ignored); not assertions |

Test 11 compares alert **episodes** across several seeds in aggregate, not seed by seed: raising
the threshold can split one long episode into two, so a single seed may legitimately start more
episodes at the higher threshold. Alert hours are monotone, so they are checked per seed.

The tests were checked for sensitivity: re-introducing the cohort-toggle bug fixed during
development makes tests 6 and 7 fail.

The suite is desktop only; there are no phone-width tests.
