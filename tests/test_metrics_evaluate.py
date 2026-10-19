import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis import metrics as M
from sepsis.evaluate import bootstrap_counts


def random_cohort(n_patients=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        n = int(rng.integers(8, 60))
        y = np.zeros(n, dtype=int)
        if rng.random() < 0.35:
            y[int(rng.integers(0, n)) :] = 1
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": f"A_p{i:06d}",
                    "ICULOS": np.arange(1, n + 1),
                    "SepsisLabel": y,
                    "p": np.clip(rng.random(n) * 0.5 + 0.4 * y, 0, 1),
                }
            )  # fmt: skip
        )
    df = pd.concat(rows, ignore_index=True)
    return df, M.make_cohort(df, df["p"].to_numpy())


def test_rank_metrics_match_sklearn_including_weights():
    _, c = random_cohort()
    c.p[::7] = 0.5  # force ties
    rank = M.RankMetrics(c.y, c.p)
    auroc, auprc, _ = rank.auroc_auprc_brier()
    assert auroc == pytest.approx(roc_auc_score(c.y, c.p))
    assert auprc == pytest.approx(average_precision_score(c.y, c.p))
    # integer weights == repeating those rows
    w = np.random.default_rng(1).integers(0, 4, len(c.y)).astype(float)
    rep = np.repeat(np.arange(len(c.y)), w.astype(int))
    auroc_w, auprc_w, brier_w = rank.auroc_auprc_brier(w)
    assert auroc_w == pytest.approx(roc_auc_score(c.y[rep], c.p[rep]))
    assert auprc_w == pytest.approx(average_precision_score(c.y[rep], c.p[rep]))
    assert brier_w == pytest.approx(np.mean((c.p[rep] - c.y[rep]) ** 2))


def test_utility_matches_official_evaluate_sepsis_score(tmp_path):
    """Our cohort utility == upstream evaluate_sepsis_score() run on label/prediction files."""
    df, c = random_cohort(30, seed=3)
    alerts = (c.p >= 0.45).astype(np.int8)
    (tmp_path / "labels").mkdir()
    (tmp_path / "preds").mkdir()
    for i, (y, a, p) in enumerate(
        zip(M._split(c, c.y), M._split(c, alerts), M._split(c, c.p), strict=True)
    ):
        pd.DataFrame({"SepsisLabel": y}).to_csv(
            tmp_path / "labels" / f"p{i:06d}.psv", sep="|", index=False
        )
        pd.DataFrame({"PredictedProbability": p, "PredictedLabel": a}).to_csv(
            tmp_path / "preds" / f"p{i:06d}.psv", sep="|", index=False
        )
    *_, official_utility = M.official().evaluate_sepsis_score(
        str(tmp_path / "labels"), str(tmp_path / "preds")
    )
    best, inact = M.reference_utilities(c)
    ours = M.normalized_utility(M.utilities(c, alerts), best, inact)
    assert ours == pytest.approx(official_utility, abs=1e-12)


def test_lead_time_and_false_alarms():
    # septic patient: label 1 from ICULOS 5 -> onset at 11; first alert at ICULOS 8 -> lead 3 h
    df = pd.DataFrame(
        {"patient_id": ["s"] * 12 + ["n"] * 24, "ICULOS": list(range(1, 13)) + list(range(1, 25)),
         "SepsisLabel": [0] * 4 + [1] * 8 + [0] * 24, "p": 0.0}
    )  # fmt: skip
    df.loc[(df.patient_id == "s") & (df.ICULOS >= 8), "p"] = 1.0
    df.loc[(df.patient_id == "n") & (df.ICULOS <= 6), "p"] = 1.0  # 6 false-alarm hours
    c = M.make_cohort(df, df["p"].to_numpy())
    st = M.per_patient_alert_stats(c, (c.p >= 0.5).astype(np.int8))
    # cohort order is sorted by patient id: "n" (non-septic) then "s" (septic)
    assert st["lead"][1] == 3 and np.isnan(st["lead"][0])
    assert 24 * st["alert_hours"][0] / st["hours"][0] == pytest.approx(6.0)
    assert list(st["warned_in_window"]) == [False, True]


def test_warning_window_ignores_alerts_only_long_before_onset():
    # onset at ICULOS 41 (label from 35); window is ICULOS 29-40
    n = 45
    df = pd.DataFrame(
        {"patient_id": ["early"] * n + ["held"] * n, "ICULOS": list(range(1, n + 1)) * 2,
         "SepsisLabel": ([0] * 34 + [1] * 11) * 2, "p": 0.0}
    )  # fmt: skip
    df.loc[(df.patient_id == "early") & (df.ICULOS <= 5), "p"] = 1.0  # alert only on admission
    df.loc[(df.patient_id == "held") & (df.ICULOS >= 3), "p"] = 1.0  # alert from 3, held on
    c = M.make_cohort(df, df["p"].to_numpy())
    st = M.per_patient_alert_stats(c, (c.p >= 0.5).astype(np.int8))
    # sorted ids: "early" then "held"; both count as alerted before onset with lead 40 / 38 h
    assert list(st["lead"]) == [40, 38]
    assert list(st["warned_in_window"]) == [False, True]


def test_bootstrap_resamples_patients_not_rows():
    """Row weights from a bootstrap draw are constant within a patient, and counts sum to the
    number of patients -- i.e. patients are resampled whole."""
    _, c = random_cohort()
    counts = bootstrap_counts(np.random.default_rng(0), c.n_patients)
    assert counts.sum() == c.n_patients and counts.shape == (c.n_patients,)
    w_row = counts[c.codes]
    for k in range(c.n_patients):
        rows = w_row[c.offsets[k] : c.offsets[k] + c.rows_per_patient[k]]
        assert (rows == counts[k]).all()
    # a row-level bootstrap would give different weights to rows of the same patient
    row_level = np.random.default_rng(0).multinomial(len(c.y), np.full(len(c.y), 1 / len(c.y)))
    assert any(
        len(set(row_level[o : o + n])) > 1
        for o, n in zip(c.offsets, c.rows_per_patient, strict=True)
    )
