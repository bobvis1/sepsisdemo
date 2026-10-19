# VIZ_PLAN.md: Results showcase (interactive page)

Read `CLAUDE.md` and `PLAN.md` first. This plan adds one deliverable on top of the finished
Phases 1-10: an interactive, single-page **results showcase** that presents the hospital B
external evaluation with some showmanship and no loss of rigor. It is written for an agent
picking this up cold; everything it needs to know is here or in the files it names.

Budget: about 3 hours of the user's time. No AWS resources are needed; everything runs locally
and costs nothing.

## 1. What we are building and why

The project's audience is a clinical ML research lab. The README already has the standard
academic tables. The showcase is a single HTML page that makes the same numbers *felt*:

- **Scene 1, "The Ward"**: every hospital B patient (20,000) as a dot, placed by peak predicted
  risk and colored by outcome (caught early, caught late, missed, false alarm, quiet). A
  draggable threshold line recolors the dots live and moves the counters.
- **Scene 2, "Bedside replay"**: a dark ICU-monitor animation of one patient's hourly risk
  trace drawing itself, the alert firing, onset arriving, and the "hours of warning" banner,
  with the top SHAP drivers at the alert hour.
- **Scene 3, "Night shift"**: a 20-bed unit running through a 24-hour shift on real hospital B
  patients, beds flashing on alerts, with a running tally. Shows the false-alarm burden in a
  way a clinician understands at a glance.
- **Scene 4 (stretch), "Swimmer wall"**: all septic hospital B patients aligned at onset, one
  heat-strip per patient, sorted by first-alert time.

**The page must be credible.** Showmanship in presentation, not in the numbers. The results
are modest (47.5% of septic hospital B patients alerted before onset, 0.79 false alarms per
patient-day) and the page shows them as they are. A reviewer who checks any number on the page
against `outputs/pipe2/eval_xgboost/evaluation.json` must find it matches exactly; the build
script enforces this (section 5, checks).

## 2. Rules that carry over from CLAUDE.md (do not break)

1. **Nothing is tuned on hospital B.** The page's default threshold is the one in
   `evaluation.json` (fit on hospital A validation). Any second operating point shown (the
   false-alarm-budget threshold, section 5) is also chosen on hospital A validation. The user can
   drag the slider on hospital B for exploration, but the page labels the A-chosen threshold as
   the real operating point and has a "reset to hospital A threshold" control.
2. **Use the frozen calibration and threshold**: Platt `coef`/`intercept` and `threshold` from
   `evaluation.json`, applied with `sepsis.evaluate.platt_apply`. Do not refit anything.
3. **Reuse project code for definitions.** Onset, septic, lead time, false alarms per
   patient-day and utility must come from `sepsis.metrics` (`make_cohort`,
   `per_patient_alert_stats`, `utilities`, `reference_utilities`, `normalized_utility`,
   `ONSET_OFFSET`). Do not reimplement them in a way that could drift.
4. **Patient replays are selected by a written rule, not hand-picked** (section 6). The page says
   so.
5. Local first, small readable modules, ruff clean, type hints on public functions, pytest tests.
6. Commit only when the user asks.

## 3. Inputs (all already on disk, all git-ignored)

| What | Path | Notes |
|---|---|---|
| Model (tuned XGBoost, same as registered package v2; v3 is identical) | `outputs/pipe2/model/model.joblib` | load with `sepsis.evaluate.load_model(Path("outputs/pipe2/model"))` -> `(model, cols)` |
| Frozen calibration, threshold, reference metrics | `outputs/pipe2/eval_xgboost/evaluation.json` | `threshold` = 0.033157..., `calibration` = {coef, intercept} |
| Hospital A validation features | `data/features/validation/features.parquet` | 4,067 patients, `hospital == "A"` |
| Hospital B features | `data/features/test/features.parquet` | 20,000 patients, `hospital == "B"` |
| SHAP summary (for context only) | `outputs/pipe2/explain/explain.json` | |

Feature frames have `patient_id`, `hospital`, `SepsisLabel`, `ICULOS`, static fields,
`<var>_ffill`, `<var>_measured`, `<var>_hrs_since`, rolling stats. Vitals for the replay come
from `HR_ffill`, `Temp_ffill`, `Resp_ffill`, `MAP_ffill`, `O2Sat_ffill` (forward-filled, so
causal; label them "last recorded value").

Reference numbers from `evaluation.json` (hospital B) that the page must reproduce at the
default threshold:

| Metric | Value |
|---|---|
| septic_alerted_before_onset | 0.47548 (543 of 1,142 septic patients) |
| false_alarms_per_patient_day | 0.78654 |
| utility | 0.24518 |
| alert_rate (share of hours) | 0.06102 |
| median_lead_time_h | 33.0 |

Hospital A validation equivalents are in the same file under `hospital_A_validation`. Always
read them from the file; the table above is just for orientation.

## 4. Environment notes

- Tools are off the Bash PATH: prefix commands with
  `export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH`. Run Python via `uv run python`.
- XGBoost runs natively (libomp is installed).
- Google Chrome is **not** installed, so there is no headless-browser screenshot. For visual
  checks, publish the page as a private Artifact (section 9) and ask the user to look, or check
  whether `node` exists (`command -v node`) and at least syntax-check the page's inline script.
- Before writing any HTML, load the **`artifact-design`** and **`dataviz`** skills (Skill tool)
  and follow them: color tokens, light/dark handling, CDN rules, phone-width layout, colorblind-
  safe categorical palette.

## 5. Phase V1: data export (`src/sepsis/showcase.py`), about 1 hour

New module `src/sepsis/showcase.py`, runnable as

```
uv run python -m sepsis.showcase \
    --model-dir outputs/pipe2/model \
    --evaluation outputs/pipe2/eval_xgboost \
    --features data/features \
    --template showcase/template.html \
    --out outputs/showcase
```

It writes `outputs/showcase/data.json` (for inspection) and `outputs/showcase/index.html`
(template with the data injected, section 7). Keep pure functions separate from I/O so they can
be tested on the synthetic frames from `tests/helpers.make_df`.

### 5.1 Scoring

For each cohort (`"B"` = test, `"A_val"` = validation):

1. `p_raw = E.predict(model, cols, df)`; `p_cal = E.platt_apply(p_raw, **ev["calibration"])`.
2. `c = M.make_cohort(df, p_cal)` (rows sorted by patient, ICULOS; keep the sort order and the
   patient-id order, `pd.factorize(..., sort=False)` on the sorted frame, for later lookups).

### 5.2 Per-patient outcome fields (Scene 1)

The key design choice: **x-position is the patient's peak risk over the hours that decide their
category**, so the threshold line partitions the dots exactly.

- `onset = first ICULOS with label 1 + ONSET_OFFSET` (septic patients only), as in
  `per_patient_alert_stats`.
- Septic patient: `x = peak_pre` = max calibrated risk over rows with `ICULOS < onset`
  ("alerted before onset" in `metrics.py` means `first_alert < onset`). Also `peak_all` = max
  over the whole stay.
- Non-septic patient: `x = peak_all`.
- Category at threshold `t` (computed in JS, and in Python for the checks):
  - septic and `peak_pre >= t` -> **caught early**
  - septic and `peak_pre < t <= peak_all` -> **caught late**
  - septic and `peak_all < t` -> **missed**
  - non-septic and `peak_all >= t` -> **false alarm** (patient had at least one alert hour)
  - non-septic and `peak_all < t` -> **quiet**
- Also export per patient: `hours` (rows), and for septic patients `onset` relative hour is not
  needed in Scene 1.

**Sina layout (precomputed in Python, deterministic):** 20,000 dots do not fit a classic
beeswarm. Use a sina plot: x = log10(clip(x, 1e-4, 1)); for each group (septic above the axis,
non-septic below, mirrored), estimate density over x (histogram of ~120 log bins, lightly
smoothed), and give each dot `y = u * density(x) / max_density` with `u ~ Uniform(0, 1)` from
`np.random.default_rng(0)`. Export `y` in [0, 1]; JS scales septic upward and non-septic
downward from the center axis, with independent height scales so the 1,142 septic patients
are not visually crushed (state "heights not to scale between groups" in a tiny caption).

### 5.3 Threshold grid (counters for Scene 1)

Colors can be computed exactly in JS from `x`/`peak_all`, but false alarms per patient-day and
utility need hourly data, so precompute them on a grid:

- Thresholds: 60 log-spaced values from 0.003 to 0.5, plus the A-chosen threshold and the
  budget threshold (below), sorted, deduplicated.
- For each threshold and cohort: `alerts = (c.p >= t)`; `st = M.per_patient_alert_stats(c,
  alerts)`; record `early`, `late`, `missed`, `false_alarm`, `quiet` counts,
  `septic_alerted_before_onset`, `false_alarms_per_patient_day` (same formula as
  `evaluate.metric_values`, unweighted), `alert_rate`, `utility` via
  `M.normalized_utility(M.utilities(c, alerts), best, inact)` with `best, inact =
  M.reference_utilities(c)` computed once per cohort, and `median_lead_time_h` over
  septic patients with non-NaN lead.
- `M.utilities` loops over patients in Python; expect a few seconds per threshold for hospital
  B. If the grid takes more than ~10 minutes, drop to 40 points (`--grid-n`).
- JS snaps the slider to the nearest grid threshold for the counters; colors use the snapped
  threshold too, so dots and counters always agree.

**Budget threshold (second operating point, chosen on hospital A).** From the hospital A
validation grid, take the smallest threshold whose `false_alarms_per_patient_day <= 0.5`. This
answers the limitation already written in `docs/future_work.md` ("choose the threshold under a
false-alarm budget"). It is shown as a second marker on the slider, labelled "0.5 false alarms
per day budget, chosen on hospital A". Record its hospital B results in `data.json` and in the
build log; mention them in the final report to the user.

### 5.4 Built-in consistency checks (fail the build if any fails)

At the A-chosen threshold, for both cohorts, compare against `evaluation.json` with tolerance
1e-6 (relative where values are large):

- `septic_alerted_before_onset`, `false_alarms_per_patient_day`, `utility`, `alert_rate`,
  `median_lead_time_h`.
- `early / n_septic` computed from the per-patient `x` and category rule equals
  `septic_alerted_before_onset` (proves the Scene 1 coloring matches the metric).
- `n_patients` and `n_septic_patients` match.

Print a short table of these checks. If one fails, stop and investigate; do not loosen the
tolerance to make it pass.

### 5.5 Size and precision

- Round risks to 4 significant figures and `y` to 3 decimals; store columns as arrays
  (struct-of-arrays), not one object per patient.
- Target total `data.json` under 3 MB. Do not export patient ids for the 20,000 Scene 1 dots
  (not needed). Replay and shift-pool patients keep their ids (they are public PhysioNet
  record names, useful for a reviewer who wants to check).

## 6. Data for Scenes 2, 3 and 4 (same module, same run)

### 6.1 Replay patients (Scene 2): selection rule

From hospital B, at the A-chosen threshold, restricted to stays of 24-150 hours, using
`np.random.default_rng(0)` for any ties:

| Slot | Rule | Count |
|---|---|---|
| Typical early catch | caught early, lead time closest to the hospital B median lead time | 3 |
| Late catch | caught late, random | 2 |
| Miss | missed, random | 2 |
| False alarm | false alarm, alert-hour count closest to the median among false-alarm patients | 2 |
| Quiet | quiet, random | 1 |

Show the first typical early catch by default. A small caption on the page: "Patients chosen
by a fixed rule (typical lead time, random otherwise), not cherry-picked."

Per replay patient export: `id`, `category`, `iculos[]`, `risk[]` (calibrated), `HR[]`,
`Temp[]`, `Resp[]`, `MAP[]`, `O2Sat[]` (from `_ffill`, null where NaN), `onset` (ICULOS or
null), `first_alert` (ICULOS or null), `lead_h` (or null), and `drivers`.

**Drivers (SHAP):** compute XGBoost `pred_contribs` (see `sepsis.explain.shap_summary` for the
call) on all rows of the replay patients. At the focus hour (first alert hour, or the peak-risk
hour if never alerted), sum contributions per clinical variable with
`sepsis.explain.base_variable`, keep the top 3 positive ones, and export
`[{var, label, contrib}]` with readable labels (HR -> "Heart rate", Resp -> "Respiratory
rate", ICULOS -> "Time in ICU", FiO2 -> "Oxygen therapy (FiO2)", etc.; write a small mapping
dict covering all 40 variables). SHAP here is in log-odds of the uncalibrated model; the page
shows relative bar lengths and the words "pushed the risk up", never raw numbers as
probabilities. If "Time in ICU" is a top driver, show it; it is honest and a reviewer will know
why.

### 6.2 Shift pool (Scene 3)

A random sample of 400 hospital B patients (`default_rng(0)`, uniform over patients, so
prevalence is real, about 23 septic). Per patient: `risk[]` (calibrated, 4 sig figs) and
`onset_idx` (row index of onset, or null; can be past the end of the stay), `septic`.

### 6.3 Swimmer wall (Scene 4, stretch)

All septic hospital B patients. For each, risk over relative hours -48 .. +6 around onset
(null before admission), quantized to uint8 on the page's log scale (0 = 1e-4, 255 = 1), plus
`first_alert_rel` (hours relative to onset, or null). Sort rows by `first_alert_rel` (earliest
first, never-alerted last). About 1,142 x 55 values; encode each row as a base64 string of
bytes to keep it small. Skip this entirely if V1-V3 ran long.

## 7. Page structure (`showcase/template.html`), Phase V2-V3, about 1.5 hours

One committed template, `showcase/template.html`, containing
`<script id="showcase-data" type="application/json">__SHOWCASE_DATA__</script>`. The build
step replaces the placeholder with the JSON, escaping `</` as `<\/` so the data cannot close the
script tag. Output: `outputs/showcase/index.html`, fully self-contained except for the CDN
library.

**Tech:** plain JavaScript and Canvas 2D for the dots and traces (20,000 animated SVG nodes is
too slow). D3 v7 is allowed for scales/axes only, from cdnjs with an exact version
(`https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js`); writing the few scales by hand
is also fine. No build tooling, no frameworks. Handle `devicePixelRatio` for crisp canvases.
Respect `prefers-reduced-motion` (skip animations, show end states).

**Look:** the `artifact-design` skill sets the contract. Within it, aim for a dark,
clinical-monitor aesthetic for Scenes 2 and 3 (near-black panel, phosphor-style trace, mono
numerals) inside a clean page. The five outcome colors must be colorblind-safe (use the
`dataviz` skill's validator) and semantically ordered: correct outcomes in cool hues, errors in
warm hues, quiet in a muted neutral. Never rely on color alone: legend with live counts, hover
tooltips, and category names in the tally.

### Hero (top of page)

Title along the lines of "Can a model see sepsis coming?" and a one-line subtitle: "An hourly
early-warning model trained on one hospital and tested, untouched, on 20,000 patients from
another." Four big numbers that count up on load, all from `evaluation.json` hospital B:
patients evaluated, share of septic patients alerted before onset, median hours of warning,
false alarms per patient-day. A thin line under it: "Trained, registered and served on Amazon
SageMaker" with a link to the repo README if the user provides the URL (leave a placeholder
otherwise).

### Scene 1: The Ward

- Wide canvas, log-scale x axis "peak predicted risk" with ticks at 0.1%, 1%, 10%, 100%.
  Septic patients above the center axis, non-septic below, labels on the left.
- Intro animation: dots fall from the top into their sina positions (staggered, under 2 s
  total), all grey; then a color sweep from left to right as the threshold line slides in from
  the right to the A threshold.
- Draggable vertical threshold line (also keyboard: arrow keys step one grid point), snapping to
  grid thresholds. Two fixed markers: "utility-optimal, chosen on hospital A" and "0.5 false
  alarms/day budget, chosen on hospital A". A "reset" button.
- Counter panel updated live from the grid: caught early / late / missed (with % of septic),
  false-alarm patients / quiet, false alarms per patient-day, challenge utility, median lead
  time.
- Toggle: "Hospital B (external test)" / "Hospital A (validation)". Switching animates dots to
  the other cohort's positions (or cross-fades; whichever is simpler). Default B. A one-line
  note explains that A is where the threshold was chosen.
- Hover a dot: its category and peak risk. No patient ids.

### Scene 2: Bedside replay

- Monitor panel: risk trace (log or linear y; pick what makes the threshold crossing readable,
  probably linear 0-max with the threshold line), drawn left to right at about 8 hours per second
  with a glowing leading point. Small vitals strips underneath (HR, MAP, Resp, Temp, O2Sat), drawn
  in sync.
- When the trace first crosses the threshold: the threshold line flashes, an "ALERT, hour N"
  tag drops in, and the drivers card slides in ("What pushed the risk up at this hour": three
  bars).
- At onset: a vertical "Sepsis onset" marker and a banner "N hours of warning" (or "Alert came
  N hours after onset" / "No alert before onset" for late/missed; "No sepsis: false alarm" or
  "No alert, no sepsis" for the others).
- Controls: play/pause, replay, next/previous patient, patient category chips. Caption with the
  selection rule (section 6.1).

### Scene 3: Night shift

- A grid of 20 beds (5 x 4 desktop, 2 columns on phones). Each bed is a small card with a tiny
  live risk sparkline.
- Simulation (in JS, seeded PRNG such as mulberry32 so "same seed = same shift"): each bed is
  assigned a random pool patient starting at a random hour of their stay; when a stay ends, a new
  pool patient is admitted at hour 1. Run 24 simulated hours at about 1 hour per 0.75 s (speed
  control).
- Bed states: stable (neutral), alert hour (amber pulse), past onset (red outline). The alert
  is "on" for any hour where risk >= the selected threshold.
- Tally strip, labelled precisely: "Alert episodes started" (a new episode = an alert hour after
  a non-alert hour), "...on patients who developed sepsis", "Sepsis onsets this shift", "...that
  had an alert in the 48 h before". A one-line note that the formal metric counts alert hours,
  not episodes.
- Threshold selector: "Hospital A utility-optimal" vs "0.5 false alarms/day budget". Running the
  same seed under both is the punchline: fewer alarms, fewer catches.
- "New shift" button (new seed). Many shifts will have no sepsis at all; that is real, and the
  note says so.

### Scene 4 (stretch): Swimmer wall

Canvas heatmap, one row per septic patient, columns = hours relative to onset (-48 to +6), a
vertical onset line, a marker at each row's first alert. Same log color scale as the risk
legend. Caption: "Each row is one septic patient at hospital B, aligned at clinical onset."

### Footer (required content)

- "Research demonstration. Not for clinical use."
- Method notes in 3-4 lines: label shifted 6 h before onset; onset = first positive hour + 6;
  calibration and threshold fit on hospital A only; "every number on this page reproduces
  `evaluation.json`".
- Data attribution: PhysioNet/Computing in Cardiology Challenge 2019, CC BY 4.0, with the
  citation Reyna MA et al., "Early Prediction of Sepsis From Clinical Data: The PhysioNet/
  Computing in Cardiology Challenge 2019", Critical Care Medicine 48(2):210-217, 2020; Goldberger
  AL et al., PhysioBank, PhysioToolkit, and PhysioNet, Circulation 101(23):e215-e220, 2000.

### Responsive and accessibility

Works at 1440 px and at phone width (390 px, 16 px gutters, no horizontal scroll; Scene 1 canvas
can shrink its height and dot radius). Canvases get `role="img"` and an `aria-label` summarizing
the current state (for example "At threshold 3.3%: 543 of 1,142 septic patients caught early").

## 8. Tests (`tests/test_showcase.py`), part of Phase V1

Use `tests/helpers.make_df` plus hand-built tiny cohorts with known answers.

1. **Category rule matches metrics:** for random thresholds on a synthetic cohort, the count of
   "caught early" from `peak_pre >= t` equals the number of non-NaN `lead` values from
   `M.per_patient_alert_stats` at the same threshold. Same for false alarm vs `alert_hours > 0`
   among non-septic patients.
2. **Hand-built case:** a septic patient whose only alert is at the onset hour is "late", not
   "early"; one whose alert is one hour before onset is "early".
3. **Frozen parameters:** the export function takes threshold and calibration as arguments and
   never fits anything (assert that a stub evaluation dict's values flow through to `meta`).
4. **Replay selection is deterministic:** same inputs give the same patient ids twice.
5. **Template injection** escapes `</script>` inside the data.

Run `uv run pytest -q` and `uv run ruff check . && uv run ruff format --check .`.

## 9. Phase V4: build, verify, publish, about 0.5 hour

1. Run the full build (command in section 5). Paste the consistency-check table in the report.
2. Check `outputs/showcase/index.html` size (target under 4 MB).
3. Publish `outputs/showcase/index.html` with the Artifact tool as a **private** artifact (icon
   "chart"). The data is derived from an open, de-identified CC BY 4.0 dataset, so a private
   link is fine; the user decides whether to share it. Give the user the link and ask them to
   check: intro animation, threshold drag, A/B toggle, replay of all 10 patients, a couple of
   shifts, phone width (narrow the browser), and light/dark mode.
4. Fix what they report; republish to the same file path so the URL stays the same.

## 10. Phase V5: wire into the repo, about 0.25 hour

- `Makefile`: add `showcase:` target running the build command (not billable).
- `README.md`: a short "Results showcase" section (2-3 sentences, the build command, and the
  artifact link if the user wants it public in the README; ask before adding the link).
- `.gitignore`: nothing new needed (`outputs/` is already ignored). The template and module are
  committed; the built HTML is not.
- `docs/future_work.md`: add anything cut (for example Scene 4, an animated GIF for the README,
  live scoring of the replay through the serverless endpoint).

## 11. Files touched (summary)

| File | Change |
|---|---|
| `src/sepsis/showcase.py` | new: scoring, per-patient fields, sina layout, grid, replay/pool/wall export, checks, template injection, CLI |
| `showcase/template.html` | new: the page |
| `tests/test_showcase.py` | new |
| `Makefile` | add `showcase` target |
| `README.md` | short section |
| `docs/future_work.md` | cut items |
| `VIZ_PLAN.md` | this file |

Do not modify `sepsis.metrics`, `sepsis.evaluate` or `sepsis.explain`. If a helper there would
be useful (for example a public `peak_pre` function), add it in `showcase.py` instead.

## 12. Cut order if running long

1. Scene 4 (swimmer wall)
2. Hospital A toggle in Scene 1 (keep B only; keep the A threshold markers)
3. Vitals strips in Scene 2 (keep risk trace, alert, onset, drivers)
4. Scene 3 speed control and threshold selector (keep one threshold)

Never cut: the consistency checks, the "chosen on hospital A" labelling, the selection-rule
caption, the footer disclaimer and attribution.

## 13. Reporting

At the end of each phase report what was done, how it was verified (command plus output), and
any surprises, then continue unless the user has asked to pause between phases. When finished,
report: the artifact link, the consistency-check table, the hospital B results at the 0.5
false-alarm budget threshold (new information the user has not seen), and anything cut.
