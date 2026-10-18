# Vendored: PhysioNet/CinC Challenge 2019 official scoring code

`evaluate_sepsis_score.py` and `LICENSE.txt` are copied **unmodified** from
https://github.com/physionetchallenges/evaluation-2019 (BSD 2-Clause, Copyright (c) 2019 PhysioNet).

- Upstream commit: `467c49b514542be7a4a0bafe40fa2c3b064dda2e` (master, last pushed 2019-11-10)
- Retrieved: 2026-10-18
- SHA-256 of `evaluate_sepsis_score.py`: `26b8b26267ed32e8b7a7a27e45201cfc8c6640e717ba4cdc1f452b32f12b99e5`

Used for the utility score (`compute_prediction_utility`) via `src/sepsis/metrics.py`; the utility
function is deliberately not reimplemented. `tests/test_metrics_evaluate.py` checks our cohort
score against the upstream `evaluate_sepsis_score` run on files.

Citation: Reyna MA et al., "Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing
in Cardiology Challenge 2019", Critical Care Medicine 48(2), 2020.
