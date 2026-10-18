import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import joblib
import numpy as np
import pytest

from helpers import make_df
from sepsis.evaluate import Platt, platt_apply
from sepsis.features import TIME_VARYING, build_features, feature_columns
from sepsis.inference import Scorer, make_handler
from sepsis.train import build_logreg


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small real model (logreg: no OpenMP needed) trained on synthetic patients."""
    df = make_df(n_patients=30, hours=40, seed=5)
    feats = build_features(df)
    cols = feature_columns(feats)
    model = build_logreg(1.0, 0).fit(feats[cols].to_numpy(np.float32), feats["SepsisLabel"])
    d = tmp_path_factory.mktemp("model")
    joblib.dump({"model": model, "columns": cols}, d / "model.joblib")
    return d, model, cols, df


def records(df, pid):
    cols = ["ICULOS", "Age", "Gender", "HospAdmTime", *TIME_VARYING]
    rows = df[df.patient_id == pid][cols].astype(object)
    return [{k: (None if v != v else v) for k, v in r.items()} for r in rows.to_dict("records")]


def test_scores_match_training_feature_path(trained):
    d, model, cols, df = trained
    pid = df.patient_id.iloc[3]
    out = Scorer(d).score({"patient_id": pid, "records": records(df, pid), "all_hours": True})
    f = build_features(df)
    expected = model.predict_proba(f.loc[f.patient_id == pid, cols].to_numpy(np.float32))[:, 1]
    np.testing.assert_allclose(out["scores"], expected, atol=1e-6)
    assert out["risk_score"] == pytest.approx(expected[-1]) and out["alert"] is None


def test_serving_is_causal_score_at_t_ignores_later_rows(trained):
    d, _, _, df = trained
    pid = df.patient_id.iloc[1]
    recs, scorer = records(df, pid), Scorer(d)
    full = scorer.score({"records": recs, "all_hours": True})["scores"]
    for t in (0, 7, 20):
        part = scorer.score({"records": recs[: t + 1]})
        assert part["risk_score"] == pytest.approx(full[t], abs=1e-6)
        assert part["iculos"] == recs[t]["ICULOS"]


def test_calibration_and_alert_from_evaluation_json(trained, tmp_path):
    d, model, cols, df = trained
    p = model.predict_proba(build_features(df)[cols].to_numpy(np.float32))[:, 1]
    platt = Platt().fit(p, df.sort_values(["patient_id", "ICULOS"]).SepsisLabel.to_numpy())
    coef, icpt = float(platt.lr.coef_[0, 0]), float(platt.lr.intercept_[0])
    np.testing.assert_allclose(platt_apply(p, coef, icpt), platt.transform(p), atol=1e-9)
    (tmp_path / "model.joblib").write_bytes((d / "model.joblib").read_bytes())
    thr = 0.5
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {"model": "m", "threshold": thr, "calibration": {"coef": coef, "intercept": icpt}}
        )
    )
    pid = df.patient_id.iloc[0]
    out = Scorer(tmp_path).score({"records": records(df, pid), "all_hours": True})
    assert out["calibrated"] and out["alert"] == (out["scores"][-1] >= thr)


@pytest.mark.parametrize(
    "payload, msg",
    [
        ({}, "non-empty list"),
        ({"records": []}, "non-empty list"),
        ({"records": [{"HR": 80}]}, "ICULOS"),
        ({"records": [{"ICULOS": 1}, {"ICULOS": 3}]}, "consecutive"),
        ({"records": [{"ICULOS": 1, "HR": "fast"}]}, "numeric"),
        ({"records": [{"ICULOS": i} for i in range(1, 1002)]}, "at most"),
    ],
)
def test_bad_requests_rejected(trained, payload, msg):
    from sepsis.inference import RequestError

    with pytest.raises(RequestError, match=msg):
        Scorer(trained[0]).score(payload)


def test_http_contract_ping_invoke_jsonlines_and_errors(trained):
    d, _, _, df = trained
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Scorer(d)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, body=None, ctype="application/json"):
        req = urllib.request.Request(base + path, data=body, headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        assert call("/ping")[0] == 200
        pids = list(dict.fromkeys(df.patient_id))[:2]
        status, body = call("/invocations", json.dumps({"records": records(df, pids[0])}).encode())
        assert status == 200 and 0 <= json.loads(body)["risk_score"] <= 1
        lines = "\n".join(json.dumps({"patient_id": p, "records": records(df, p)}) for p in pids)
        status, body = call("/invocations", lines.encode(), "application/jsonlines")
        assert status == 200 and [json.loads(x)["patient_id"] for x in body.splitlines()] == pids
        status, body = call("/invocations", b'{"records": []}')
        assert status == 400 and "error" in json.loads(body)
        assert call("/invocations", b"not json")[0] == 400
        assert call("/nope")[0] == 404
    finally:
        server.shutdown()
