import json

import joblib
import numpy as np
import pandas as pd
import pytest

from helpers import make_df
from sepsis import evaluate as E
from sepsis import metrics as M
from sepsis import showcase as S
from sepsis.features import STATIC, TIME_VARYING, build_features, feature_columns


class StubModel:
    """Deterministic content-based probabilities in (0, 0.3): no fitting, no OpenMP."""

    def predict_proba(self, X):
        z = np.nan_to_num(np.asarray(X)[:, :6]).sum(axis=1)
        p = 0.3 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


def cohort_from(rows: dict[str, list[tuple[int, int, float]]]) -> M.Cohort:
    """rows: patient -> [(ICULOS, label, risk)]"""
    recs = [(pid, i, y, p) for pid, rr in rows.items() for i, y, p in rr]
    df = pd.DataFrame(recs, columns=["patient_id", "ICULOS", "SepsisLabel", "p"])
    return M.make_cohort(df, df["p"].to_numpy())


def test_category_rule_matches_metrics():
    df = make_df(n_patients=40, hours=40, seed=3)
    p = np.random.default_rng(1).random(len(df))
    c = M.make_cohort(df, p)
    s = S.summarize(c)
    for t in (0.2, 0.5, 0.8, 0.95):
        cat = S.categorize(s.septic, s.peak_pre, s.peak_all, t)
        st = M.per_patient_alert_stats(c, (c.p >= t).astype(np.int8))
        assert (cat == S.EARLY).sum() == (~np.isnan(st["lead"])).sum()
        assert (cat == S.FALSE_ALARM).sum() == ((~st["septic"]) & (st["alert_hours"] > 0)).sum()
        assert (cat == S.QUIET).sum() == ((~st["septic"]) & (st["alert_hours"] == 0)).sum()
        assert np.bincount(cat, minlength=5).sum() == c.n_patients


def test_alert_at_onset_hour_is_late_and_one_hour_before_is_early():
    # label is 1 from ICULOS 11, so clinical onset = 11 + 6 = 17
    def patient(alert_at: int):
        return [(h, int(h >= 11), 0.9 if h == alert_at else 0.1) for h in range(1, 21)]

    c = cohort_from({"a_at_onset": patient(17), "b_before": patient(16), "c_never": patient(99)})
    s = S.summarize(c)
    cat = S.categorize(s.septic, s.peak_pre, s.peak_all, 0.5)
    assert [S.CATEGORIES[i] for i in cat] == ["late", "early", "missed"]
    assert list(s.onset) == [17, 17, 17]


def test_scoring_uses_the_given_calibration_and_fits_nothing():
    df = make_df(n_patients=6, hours=20, seed=4)
    feats = build_features(df)
    cols = feature_columns(feats)
    cal = {"coef": 1.7, "intercept": -0.4}
    sc = S.score_cohort(feats, StubModel(), cols, cal)
    raw = E.predict(StubModel(), cols, feats.iloc[np.lexsort((feats.ICULOS, feats.patient_id))])
    np.testing.assert_allclose(sc.c.p, E.platt_apply(raw, **cal))
    assert list(sc.ids) == sorted(feats.patient_id.unique())


def test_replay_selection_is_deterministic_and_follows_the_rule():
    df = make_df(n_patients=60, hours=40, seed=5)
    c = M.make_cohort(df, np.random.default_rng(2).random(len(df)))
    s = S.summarize(c)
    t = 0.8
    stats = M.per_patient_alert_stats(c, (c.p >= t).astype(np.int8))
    a, b = S.select_replay(s, stats, t), S.select_replay(s, stats, t)
    assert a == b
    cats = S.categorize(s.septic, s.peak_pre, s.peak_all, t)
    assert all(S.CATEGORIES[cats[i]] == name for i, name in a)
    assert all(24 <= s.hours[i] <= 150 for i, _ in a)
    assert len({i for i, _ in a}) == len(a)


def test_labels_cover_every_variable():
    assert set(S.LABELS) >= set(TIME_VARYING) | set(STATIC)


def test_template_injection_cannot_close_the_script_tag():
    data = {"note": "</script><script>alert(1)</script> <!-- x", "n": [1, 2]}
    html = S.inject_data('<script id="d" type="application/json">__SHOWCASE_DATA__</script>', data)
    body = html.split('type="application/json">', 1)[1].rsplit("</script>", 1)[0]
    assert "</" not in body and "<!--" not in body
    assert json.loads(body.replace("<\\/", "</").replace("<\\!--", "<!--")) == data
    with pytest.raises(ValueError):
        S.inject_data("no placeholder", data)


def test_check_flags_a_disagreement():
    c = cohort_from({"p1": [(h, int(h >= 5), 0.1) for h in range(1, 12)]})
    s = S.summarize(c)
    row = {k: 0.5 for k in S.CHECK_METRICS} | {"early": 0}
    ev_h = {"alert_rate": 0.5, "n_patients": 1, "n_septic_patients": 1,
            "metrics": {k: {"value": 0.5} for k in S.CHECK_METRICS}}  # fmt: skip
    res = S.check_against_evaluation("B", row, ev_h, s)
    assert next(r for r in res if r["check"] == "alert_rate")["ok"]
    assert not next(r for r in res if r["check"].startswith("early"))["ok"]
    ev_h["metrics"]["utility"]["value"] = 0.6


def test_end_to_end_build_passes_checks_and_carries_frozen_values(tmp_path):
    cal = {"coef": 1.3, "intercept": -0.2}
    thr = None
    model_dir, ev_dir, feat = tmp_path / "model", tmp_path / "ev", tmp_path / "feat"
    ev = {"threshold": thr, "calibration": cal, "n_bootstrap": 3, "seed": 0}
    for sub, hosp, key, seed in (
        ("test", "B", "hospital_B", 11),
        ("validation", "A", "hospital_A_validation", 12),
    ):
        feats = build_features(make_df(n_patients=45, hours=40, seed=seed, hospital=hosp))
        (feat / sub).mkdir(parents=True)
        feats.to_parquet(feat / sub / "features.parquet")
        cols = feature_columns(feats)
        p_raw = E.predict(StubModel(), cols, feats)
        if thr is None:  # frozen from the first cohort, as the real threshold comes from A
            thr = float(np.quantile(E.platt_apply(p_raw, **cal), 0.9))
            ev["threshold"] = thr
        ev[key] = E.evaluate_hospital(feats, p_raw, E.platt_apply(p_raw, **cal), thr, 3, 0)
    model_dir.mkdir()
    joblib.dump({"model": StubModel(), "columns": cols}, model_dir / "model.joblib")
    ev_dir.mkdir()
    (ev_dir / "evaluation.json").write_text(json.dumps(ev))

    data = S.build(model_dir, ev_dir, feat, grid_n=8, drivers_fn=lambda rows, focus: [])
    assert data["meta"]["threshold"] == thr and data["meta"]["calibration"] == cal
    assert all(k["ok"] for k in data["meta"]["checks"])
    assert thr in data["thresholds"] and data["meta"]["budget_threshold"] in data["thresholds"]
    assert data["cohorts"]["B"]["n"] == 45 and len(data["cohorts"]["B"]["x"]) == 45
    S.to_json(data)  # no NaN anywhere
    # the budget threshold is chosen on hospital A: its A false-alarm rate meets the budget
    assert data["budget_results"]["A_val"]["false_alarms_per_patient_day"] <= S.BUDGET_FA_PER_DAY
