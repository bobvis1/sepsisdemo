# Model card: hourly sepsis early-warning model (XGBoost)

**Research demonstration. Not a medical device. Not for clinical use or to inform patient care.**

## Model details
- **Task:** for each ICU patient-hour, estimate the probability that the patient meets the dataset's
  sepsis label within roughly the next 6 hours (the label is already shifted: it is 1 from 6 hours
  before clinical onset).
- **Model:** gradient-boosted trees (XGBoost, histogram method, `scale_pos_weight` for class
  imbalance, early stopping on hospital A validation AUPRC). Baseline: logistic regression with
  median imputation and standardisation fit on the training fold only.
- **Inputs (140 features):** 40 recorded variables (vitals, labs, demographics, `ICULOS`) turned into
  causal features: forward-filled values, "measured this hour" flags, hours since last
  measurement, and trailing 6 h and 12 h mean and slope for vitals.
- **Output:** a probability per hour; the endpoint returns the latest hour's Platt-calibrated
  probability and an alert flag at a frozen threshold.
- **Training environment:** Amazon SageMaker, private VPC, network isolation, KMS encryption.
  Software versions are listed in the README.

## Intended use and out-of-scope use
- **Intended:** demonstrating a secure MLOps pipeline and methodologically careful evaluation
  (patient-level splits, external-site test, bootstrap CIs) for a research audience.
- **Out of scope:** any clinical decision support, triage, or alerting on real patients; use on
  populations, units or hospitals other than the two in the training data; paediatric use.

## Data
PhysioNet/Computing in Cardiology Challenge 2019 training data (CC BY 4.0): hospital A
(20,336 patients, 790,215 patient-hours, 8.8% ever septic) and hospital B (20,000 patients,
761,995 patient-hours, 5.7% ever septic), from different hospital systems.
Hospital A is split 80/20 by patient (stratified on ever-septic) into train and validation;
hospital B is used only for final evaluation. See [data_summary.md](data_summary.md) for
missingness, prevalence and length of stay. 11% of septic patients in A and 20% in B are already
positive at their first recorded hour and cannot be warned in advance.

## Metrics
Calibration (Platt) and the alert threshold are fit on hospital A validation only and frozen for
hospital B. The threshold maximises the challenge's normalised utility score (official scoring code,
vendored unmodified). CIs: percentile bootstrap, patients resampled.

<!-- results:headline:start -->
_Source: local run (models trained locally; SageMaker pipeline run pending). 95% CIs from 1000 patient-level bootstrap resamples. Calibration and alert threshold fit on hospital A validation only; hospital B is an untouched external test set._

| Metric | XGBoost, A validation | XGBoost, B (external) | Logistic regression, A validation | Logistic regression, B |
|---|---|---|---|---|
| Patients (septic) | 4,067 (358) | 20,000 (1,142) | 4,067 (358) | 20,000 (1,142) |
| Hour prevalence | 0.0216 | 0.0141 | 0.0216 | 0.0141 |
| AUROC | 0.800 (0.782-0.820) | 0.776 (0.764-0.787) | 0.754 (0.732-0.776) | 0.711 (0.697-0.725) |
| AUPRC | 0.117 (0.101-0.138) | 0.056 (0.051-0.063) | 0.097 (0.087-0.115) | 0.049 (0.045-0.056) |
| Brier (calibrated) | 0.020 (0.018-0.022) | 0.014 (0.013-0.014) | 0.020 (0.018-0.022) | 0.015 (0.014-0.017) |
| Challenge utility (normalised) | 0.382 (0.336-0.424) | 0.245 (0.221-0.270) | 0.332 (0.287-0.372) | 0.194 (0.169-0.219) |
| Median lead time, h | 35.0 (27.5-42.0) | 32.0 (29.0-37.0) | 38.0 (30.0-45.0) | 30.0 (27.0-37.0) |
| Septic patients alerted before onset | 0.858 (0.822-0.893) | 0.704 (0.676-0.730) | 0.737 (0.693-0.783) | 0.593 (0.564-0.621) |
| False alarms per non-septic patient-day | 4.33 (4.11-4.56) | 1.87 (1.77-1.97) | 2.98 (2.80-3.15) | 2.07 (1.97-2.18) |

XGBoost alert threshold: 0.0243 (calibrated probability).
<!-- results:headline:end -->

How to read it: the model ranks risk well above chance on both hospitals (AUROC about 0.78 to
0.80 for XGBoost) and beats the baseline. Performance is lower on the external hospital.
AUPRC is low in absolute terms because positive hours are rare (about 1 to 2%).

## Subgroups and what drives predictions

Subgroups use the same frozen calibration and threshold. In hospital A, females had lower AUPRC and
utility than males; that gap did **not** replicate in hospital B (where the direction reversed and
the intervals overlap), so it should be treated as unconfirmed, not as a finding. Wide intervals
mean few septic patients in a subgroup.

<!-- results:subgroups:start -->
| Group | A validation: patients (septic) | A AUPRC | A utility | B: patients (septic) | B AUPRC | B utility |
|---|---|---|---|---|---|---|
| age <40 | 400 (35) | 0.156 (0.083-0.288) | 0.383 (0.229-0.503) | 2,347 (130) | 0.054 (0.042-0.072) | 0.257 (0.181-0.334) |
| age 40-59 | 1,203 (92) | 0.134 (0.098-0.179) | 0.374 (0.266-0.465) | 6,392 (359) | 0.053 (0.045-0.064) | 0.242 (0.201-0.288) |
| age 60-74 | 1,365 (131) | 0.114 (0.090-0.149) | 0.424 (0.356-0.487) | 7,256 (422) | 0.059 (0.049-0.073) | 0.258 (0.213-0.302) |
| age 75+ | 1,099 (100) | 0.101 (0.079-0.135) | 0.332 (0.251-0.409) | 4,005 (231) | 0.062 (0.051-0.080) | 0.219 (0.165-0.266) |
| sex female | 1,693 (136) | 0.091 (0.073-0.119) | 0.323 (0.245-0.393) | 9,268 (495) | 0.061 (0.053-0.072) | 0.256 (0.219-0.291) |
| sex male | 2,374 (222) | 0.142 (0.117-0.173) | 0.418 (0.364-0.473) | 10,732 (647) | 0.053 (0.047-0.063) | 0.237 (0.205-0.270) |
<!-- results:subgroups:end -->

Variables that contribute most (mean absolute SHAP, exact TreeSHAP on a 50,000-row sample):

<!-- results:shap:start -->
| Variable | Share of total |SHAP|, A validation | Share, B |
|---|---|---|
| FiO2 | 12.9% | 10.7% |
| ICULOS | 12.9% | 11.2% |
| Temp | 6.9% | 6.1% |
| Resp | 5.6% | 5.2% |
| Lactate | 5.4% | 4.7% |
| Unit1 | 5.2% | 4.6% |
| SBP | 4.7% | 4.5% |
| Creatinine | 3.9% | 4.1% |
<!-- results:shap:end -->

`ICULOS` (time in ICU) and `FiO2` are among the strongest contributors. Time in ICU is a proxy for
risk, not a physiological sign, and whether a lab was measured at all reflects clinician suspicion.
The model may therefore partly learn clinician behaviour and may not transfer to units that order
tests differently.

## Limitations and risks
- **Retrospective, two sites.** No prospective validation, no workflow or human-factors evaluation.
- **Label definition.** The label is a time-shifted sepsis criterion from the challenge, not a
  clinician-adjudicated diagnosis; model quality is bounded by its noise.
- **Distribution shift.** Every metric drops on hospital B; AUPRC roughly halves. Expect further
  degradation elsewhere. Calibration was fit on hospital A and is imperfect on B.
- **Alert burden.** The utility-optimal threshold alerts on 11 to 25% of patient-hours, about 2 to 4 false
  alarms per non-septic patient-day. A clinical deployment would need a false-alarm budget.
- **Input contract.** The serving code assumes one row per ICU hour with no gaps; gaps are rejected.
- **Fairness.** Only age band and sex were examined; race, ethnicity and insurance are not in the
  data, so disparities along those axes could not be checked.
- **Status.** The registered model starts as `PendingManualApproval`; nothing here has been
  approved for use.

## Maintenance
Retraining and re-evaluation are one pipeline run (`scripts/run_pipeline.py`); results tables in
this card are regenerated from `evaluation.json` by `scripts/render_results.py`.
