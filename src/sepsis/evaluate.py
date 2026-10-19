"""Evaluate a trained model on hospital A validation and hospital B, with patient-level
bootstrap 95% CIs. Writes evaluation.json (consumed by the Pipeline) and report.md.

    python -m sepsis.evaluate --model-dir outputs/xgboost_full --features data/features \
        --out outputs/eval_xgboost

Everything fitted here (Platt calibration, alert threshold) uses hospital A validation only;
hospital B is scored with those frozen choices. Runs locally and as a SageMaker Processing job
(model under /opt/ml/processing/model, features under /opt/ml/processing/features).
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sepsis import metrics as M

PROC = Path("/opt/ml/processing")
EPS = 1e-6


def load_model(model_dir: Path):
    """Load model.joblib, from a SageMaker model.tar.gz (extracted to a temp dir) or a folder."""
    tar = model_dir / "model.tar.gz"
    if tar.exists():
        with tempfile.TemporaryDirectory() as tmp, tarfile.open(tar) as tf:
            tf.extractall(tmp, filter="data")
            bundle = joblib.load(Path(tmp) / "model.joblib")
    else:
        bundle = joblib.load(model_dir / "model.joblib")
    return bundle["model"], bundle["columns"]


def predict(model, cols: list[str], df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[cols].to_numpy(np.float32))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))[:, None]


def platt_apply(p_raw: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    """Apply a fitted Platt sigmoid: sigmoid(coef * logit(p) + intercept)."""
    z = coef * _logit(p_raw)[:, 0] + intercept
    return 1.0 / (1.0 + np.exp(-z))


class Platt:
    """Two-parameter sigmoid calibration, fit on hospital A validation only."""

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> Platt:
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(p_raw))[:, 1]


def choose_threshold(c: M.Cohort, best, inact, n: int = 60) -> tuple[float, list[dict]]:
    """Threshold maximising normalised utility on hospital A validation (candidates are upper
    quantiles of the score, where alerts are rare enough to be useful)."""
    cands = np.unique(np.quantile(c.p, np.linspace(0.50, 0.9995, n)))
    curve = []
    for t in cands:
        alerts = (c.p >= t).astype(np.int8)
        u = M.normalized_utility(M.utilities(c, alerts), best, inact)
        curve.append({"threshold": float(t), "utility": u, "alert_rate": float(alerts.mean())})
    return max(curve, key=lambda r: r["utility"])["threshold"], curve


def bootstrap_counts(rng: np.random.Generator, n_patients: int) -> np.ndarray:
    """One bootstrap draw: how many times each PATIENT is resampled (never individual rows)."""
    return rng.multinomial(n_patients, np.full(n_patients, 1.0 / n_patients))


def metric_values(c, rank, p_raw, st, obs, best, inact, w_pat=None) -> dict[str, float]:
    """All reported metrics for one (possibly resampled) cohort. w_pat = patient counts."""
    w_row = None if w_pat is None else w_pat[c.codes]
    auroc, auprc, brier = rank.auroc_auprc_brier(w_row)
    ws = np.ones(len(c.y)) if w_row is None else w_row
    wp = np.ones(c.n_patients) if w_pat is None else w_pat
    nonsep = ~st["septic"]
    alerted = ~np.isnan(st["lead"])
    keep = ~st["pos_at_first_row"]
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier_calibrated": brier,
        "brier_raw": float((ws * (p_raw - c.y) ** 2).sum() / ws.sum()),
        "utility": M.normalized_utility(obs, best, inact, wp),
        "median_lead_time_h": M.weighted_median(st["lead"][alerted], wp[alerted]),
        "median_lead_time_excl_first_row_positive_h": M.weighted_median(
            st["lead"][alerted & keep], wp[alerted & keep]
        ),
        "septic_alerted_before_onset": float((wp * alerted).sum() / (wp * st["septic"]).sum()),
        "septic_alerted_in_window": float(
            (wp * st["warned_in_window"]).sum() / (wp * st["septic"]).sum()
        ),
        "first_alert_over_24h_early": float(
            (wp * (st["lead"] > 24))[alerted].sum() / wp[alerted].sum()
        ),
        "false_alarms_per_patient_day": float(
            24 * (wp * st["alert_hours"])[nonsep].sum() / (wp * st["hours"])[nonsep].sum()
        ),
    }


def evaluate_hospital(df, p_raw, p_cal, thr, n_boot, seed) -> dict:
    c = M.make_cohort(df, p_cal)
    raw = M.make_cohort(df, p_raw).p  # same row order as c
    rank = M.RankMetrics(c.y, c.p)
    best, inact = M.reference_utilities(c)
    alerts = (c.p >= thr).astype(np.int8)
    obs = M.utilities(c, alerts)
    st = M.per_patient_alert_stats(c, alerts)

    point = metric_values(c, rank, raw, st, obs, best, inact)
    rng = np.random.default_rng(seed)
    draws = [
        metric_values(c, rank, raw, st, obs, best, inact, bootstrap_counts(rng, c.n_patients))
        for _ in range(n_boot)
    ]
    out = {}
    for k, v in point.items():
        arr = np.array([d[k] for d in draws])
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[k] = {"value": v, "ci_low": float(lo), "ci_high": float(hi)}
    return {
        "n_patients": c.n_patients,
        "n_septic_patients": int(c.septic.sum()),
        "patient_prevalence": float(c.septic.mean()),
        "hour_prevalence": float(c.y.mean()),
        "alert_rate": float(alerts.mean()),
        "metrics": out,
        "calibration_curve": M.calibration_curve(c.y, c.p),
    }


def report_md(ev: dict) -> str:
    names = {
        "auroc": "AUROC", "auprc": "AUPRC", "brier_calibrated": "Brier (Platt-calibrated)",
        "brier_raw": "Brier (raw scores)", "utility": "Challenge normalised utility",
        "median_lead_time_h": "Median lead time, h (septic, alerted before onset)",
        "median_lead_time_excl_first_row_positive_h":
            "Median lead time, h, excl. first-row positives",
        "septic_alerted_before_onset": "Septic patients alerted before onset (any time)",
        "septic_alerted_in_window": "Septic patients alerted in the 12 h before onset",
        "first_alert_over_24h_early": "Of those alerted before onset, first alert > 24 h early",
        "false_alarms_per_patient_day": "False alarms per patient-day (non-septic)",
    }  # fmt: skip
    a, b = ev["hospital_A_validation"], ev["hospital_B"]

    def cell(m):
        return f"{m['value']:.3f} ({m['ci_low']:.3f} to {m['ci_high']:.3f})"

    lines = [
        f"# Evaluation: {ev['model']}", "",
        f"Alert threshold {ev['threshold']:.4f} (calibrated probability), chosen to maximise "
        "utility on hospital A validation. Platt calibration and the threshold were fit on "
        "hospital A validation only; hospital B was never used for fitting or tuning. "
        f"95% CIs: percentile bootstrap, {ev['n_bootstrap']} resamples of patients.", "",
        "| | Hospital A validation (selection set, optimistic) | Hospital B (external) |",
        "|---|---|---|",
        f"| Patients (septic) | {a['n_patients']:,} ({a['n_septic_patients']:,}) "
        f"| {b['n_patients']:,} ({b['n_septic_patients']:,}) |",
        f"| Hour prevalence | {a['hour_prevalence']:.4f} | {b['hour_prevalence']:.4f} |",
        f"| Alert rate (share of hours) | {a['alert_rate']:.4f} | {b['alert_rate']:.4f} |",
    ]  # fmt: skip
    for k, label in names.items():
        lines.append(f"| {label} | {cell(a['metrics'][k])} | {cell(b['metrics'][k])} |")
    lines += ["", "## Calibration (equal-count bins, calibrated probabilities)", ""]
    for key, title in (
        ("hospital_A_validation", "Hospital A validation (selection set, optimistic)"),
        ("hospital_B", "Hospital B"),
    ):
        lines += [f"**{title}**", "", "| mean predicted | observed rate | hours |", "|---|---|---|"]
        lines += [
            f"| {r['mean_predicted']:.4f} | {r['observed']:.4f} | {r['n']:,} |"
            for r in ev[key]["calibration_curve"]
        ]
        lines.append("")
    lines += [
        "Hour prevalence differs between hospitals, so AUPRC is not directly comparable across "
        "them; read it against each hospital's prevalence.", "",
    ]  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=PROC / "model")
    ap.add_argument("--features", type=Path, default=PROC / "features")
    ap.add_argument("--out", type=Path, default=PROC / "evaluation")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    model, cols = load_model(args.model_dir)
    val = pd.read_parquet(args.features / "validation" / "features.parquet")
    test = pd.read_parquet(args.features / "test" / "features.parquet")
    assert set(val["hospital"]) == {"A"} and set(test["hospital"]) == {"B"}
    p_val, p_test = predict(model, cols, val), predict(model, cols, test)

    platt = Platt().fit(p_val, val["SepsisLabel"].to_numpy())
    cal_val, cal_test = platt.transform(p_val), platt.transform(p_test)
    cval = M.make_cohort(val, cal_val)
    best, inact = M.reference_utilities(cval)
    thr, curve = choose_threshold(cval, best, inact)
    print(f"threshold={thr:.4f}")

    ev = {
        "model": args.model_name or args.model_dir.name,
        "threshold": thr,
        "calibration": {
            "coef": float(platt.lr.coef_[0, 0]),
            "intercept": float(platt.lr.intercept_[0]),
        },
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
        "threshold_curve": curve,
        "hospital_A_validation": evaluate_hospital(
            val, p_val, cal_val, thr, args.n_boot, args.seed
        ),
        "hospital_B": evaluate_hospital(test, p_test, cal_test, thr, args.n_boot, args.seed),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(json.dumps(ev, indent=2))
    (args.out / "report.md").write_text(report_md(ev))
    print(report_md(ev))


if __name__ == "__main__":
    main()
