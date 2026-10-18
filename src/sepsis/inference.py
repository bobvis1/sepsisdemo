"""Inference handler and SageMaker-compatible HTTP server (GET /ping, POST /invocations).

Request (application/json), one patient's hourly history, rows consecutive in ICULOS:

    {"patient_id": "p1", "records": [{"ICULOS": 1, "HR": 88, "Temp": 37.1, "Age": 63, ...}, ...],
     "all_hours": false}

Response: {"patient_id", "iculos" (latest hour), "risk_score", "alert", "n_hours", "model", and
"scores" (one per input row) when all_hours is true}. `application/jsonlines` takes one such
request per line and returns one response per line (for Batch Transform).

Features come from the same `build_features` used in training, so the score at hour t uses only
rows <= t. If <model dir>/evaluation.json exists, risk_score is the Platt-calibrated probability
and `alert` applies the threshold chosen on hospital A validation; otherwise it is the raw model
probability and `alert` is null. Run: `python -m sepsis.inference` (the image's `serve` script).
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sepsis.evaluate import platt_apply
from sepsis.features import LABEL, STATIC, TIME_VARYING, build_features

MAX_HOURS = 1000
MAX_BODY_BYTES = 4 * 1024 * 1024  # serverless endpoints cap payloads at 4 MB
KNOWN_FIELDS = set(TIME_VARYING) | set(STATIC) | {LABEL}


class RequestError(ValueError):
    """Bad request (HTTP 400)."""


class Scorer:
    def __init__(self, model_dir: Path | str):
        model_dir = Path(model_dir)
        bundle = joblib.load(model_dir / "model.joblib")
        self.model, self.columns = bundle["model"], bundle["columns"]
        ev_path = model_dir / "evaluation.json"
        self.calibration = self.threshold = None
        if ev_path.exists():
            ev = json.loads(ev_path.read_text())
            self.calibration, self.threshold = ev["calibration"], ev["threshold"]
            self.model_name = ev.get("model", "unknown")
        else:
            self.model_name = "unknown"
        self.params = {}
        if (model_dir / "params.json").exists():
            self.params = json.loads((model_dir / "params.json").read_text())

    def history_frame(self, payload: dict) -> pd.DataFrame:
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise RequestError("'records' must be a non-empty list of hourly rows")
        if len(records) > MAX_HOURS:
            raise RequestError(f"at most {MAX_HOURS} hourly rows per request")
        if not all(isinstance(r, dict) for r in records):
            raise RequestError("each record must be an object")
        df = pd.DataFrame(records)
        if "ICULOS" not in df or df["ICULOS"].isna().any():
            raise RequestError("every record needs a numeric 'ICULOS' (hour of ICU stay)")
        for col in df.columns.intersection(list(KNOWN_FIELDS)):
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError) as e:
                raise RequestError(f"field '{col}' must be numeric or null") from e
        if len(df) > 1 and not (df["ICULOS"].diff().dropna() == 1).all():
            raise RequestError("records must be consecutive hourly rows (ICULOS increasing by 1)")
        for col in KNOWN_FIELDS - set(df.columns):
            df[col] = np.nan
        df[LABEL] = 0  # unused by the features; required column
        df.insert(0, "patient_id", str(payload.get("patient_id", "patient")))
        return df

    def score(self, payload: dict) -> dict:
        df = self.history_frame(payload)
        feats = build_features(df)
        raw = self.model.predict_proba(feats[self.columns].to_numpy(np.float32))[:, 1]
        p = platt_apply(raw, **self.calibration) if self.calibration else raw
        out = {
            "patient_id": str(payload.get("patient_id", "patient")),
            "iculos": int(df["ICULOS"].iloc[-1]),
            "n_hours": len(df),
            "risk_score": float(p[-1]),
            "alert": bool(p[-1] >= self.threshold) if self.threshold is not None else None,
            "calibrated": self.calibration is not None,
            "model": self.model_name,
        }
        ignored = sorted(set(pd.DataFrame(payload["records"]).columns) - KNOWN_FIELDS)
        if ignored:
            out["ignored_fields"] = ignored
        if payload.get("all_hours"):
            out["scores"] = [float(x) for x in p]
        return out

    def invoke(self, body: bytes, content_type: str) -> tuple[bytes, str]:
        try:
            if "jsonlines" in content_type or "ndjson" in content_type:
                lines = [ln for ln in body.decode().splitlines() if ln.strip()]
                results = [self.score(json.loads(ln)) for ln in lines]
                return ("\n".join(json.dumps(r) for r in results) + "\n").encode(), content_type
            return json.dumps(self.score(json.loads(body))).encode(), "application/json"
        except json.JSONDecodeError as e:
            raise RequestError(f"invalid JSON: {e}") from e


def make_handler(scorer: Scorer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/ping":
                self._send(200, b'{"status": "ok"}')
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):  # noqa: N802
            if self.path != "/invocations":
                return self._send(404, b'{"error": "not found"}')
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                return self._send(413, b'{"error": "payload too large"}')
            try:
                body, ctype = scorer.invoke(
                    self.rfile.read(length), self.headers.get("Content-Type", "application/json")
                )
                self._send(200, body, ctype)
            except RequestError as e:
                self._send(400, json.dumps({"error": str(e)}).encode())
            except Exception:  # never leak internals to the caller
                self._send(500, b'{"error": "internal error"}')

        def log_message(self, fmt, *args):  # one line per request on stderr (CloudWatch)
            super().log_message(fmt, *args)

    return Handler


def main() -> None:
    scorer = Scorer(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))
    print(f"serving {scorer.model_name} on :{port} (calibrated={scorer.calibration is not None})")
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(scorer)).serve_forever()  # noqa: S104


if __name__ == "__main__":
    main()
