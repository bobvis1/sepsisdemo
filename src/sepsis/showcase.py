"""Data export for the interactive results showcase (VIZ_PLAN.md, Phase V1).

    uv run python -m sepsis.showcase --model-dir outputs/pipe2/model \
        --evaluation outputs/pipe2/eval_xgboost --features data/features \
        --template showcase/template.html --out outputs/showcase

Scores hospital B (and hospital A validation) with the frozen model, calibration and threshold from
evaluation.json, derives the per-patient and per-threshold numbers the page draws, and fails the
build if any of them disagrees with evaluation.json. Nothing here fits or tunes anything. The
definitions of onset, lead time, false alarms and utility come from sepsis.metrics.

Outputs: data.json (struct-of-arrays, for inspection) and index.html (the template with the data
injected) when the template exists.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis import evaluate as E
from sepsis import metrics as M
from sepsis.explain import base_variable

EARLY, LATE, MISSED, FALSE_ALARM, QUIET = range(5)
CATEGORIES = ["early", "late", "missed", "false_alarm", "quiet"]
RISK_FLOOR = 1e-4  # risks below this share the left edge of the log axis
BUDGET_FA_PER_DAY = 0.5
REPLAY_VITALS = ["HR", "Temp", "Resp", "MAP", "O2Sat"]
REPLAY_HOURS = (24, 150)
REPLAY_SLOTS = [("early", 3), ("late", 2), ("missed", 2), ("false_alarm", 2), ("quiet", 1)]
POOL_SIZE = 400
WALL_HOURS = (-48, 6)
PLACEHOLDER = "__SHOWCASE_DATA__"
CHECK_METRICS = (
    "septic_alerted_before_onset",
    "false_alarms_per_patient_day",
    "utility",
    "alert_rate",
    "median_lead_time_h",
)

LABELS = {
    "HR": "Heart rate", "O2Sat": "Oxygen saturation", "Temp": "Temperature",
    "SBP": "Systolic blood pressure", "MAP": "Mean arterial pressure",
    "DBP": "Diastolic blood pressure", "Resp": "Respiratory rate", "EtCO2": "End-tidal CO2",
    "BaseExcess": "Base excess", "HCO3": "Bicarbonate", "FiO2": "Oxygen therapy (FiO2)",
    "pH": "Blood pH", "PaCO2": "Arterial CO2 (PaCO2)", "SaO2": "Arterial oxygen saturation",
    "AST": "Liver enzyme (AST)", "BUN": "Blood urea nitrogen",
    "Alkalinephos": "Alkaline phosphatase",
    "Calcium": "Calcium", "Chloride": "Chloride", "Creatinine": "Creatinine",
    "Bilirubin_direct": "Direct bilirubin", "Glucose": "Glucose", "Lactate": "Lactate",
    "Magnesium": "Magnesium", "Phosphate": "Phosphate", "Potassium": "Potassium",
    "Bilirubin_total": "Total bilirubin", "TroponinI": "Troponin I", "Hct": "Hematocrit",
    "Hgb": "Hemoglobin", "PTT": "Clotting time (PTT)", "WBC": "White blood cell count",
    "Fibrinogen": "Fibrinogen", "Platelets": "Platelets", "Age": "Age", "Gender": "Sex",
    "Unit1": "ICU type (unit 1)", "Unit2": "ICU type (unit 2)",
    "HospAdmTime": "Time since hospital admission", "ICULOS": "Time in ICU",
}  # fmt: skip


# ---- scoring and per-patient fields ------------------------------------------------------------


@dataclass
class Scored:
    """A cohort with calibrated risks, plus the matching sorted frame and patient ids."""

    c: M.Cohort
    frame: pd.DataFrame  # same row order as c
    ids: np.ndarray  # patient id per patient index


@dataclass
class Summary:
    """Per-patient quantities Scene 1 is built from (arrays, one entry per patient)."""

    septic: np.ndarray
    onset: np.ndarray  # ICULOS of clinical onset (first label-1 hour + 6); inf if never septic
    peak_pre: np.ndarray  # max risk over rows before onset (== peak_all for non-septic)
    peak_all: np.ndarray
    hours: np.ndarray

    @property
    def x(self) -> np.ndarray:
        """Scene 1 x position: the peak risk over the hours that decide the patient's category."""
        return np.where(self.septic, self.peak_pre, self.peak_all)


def score_cohort(df: pd.DataFrame, model, cols: list[str], calibration: dict) -> Scored:
    """Frozen model and Platt calibration; nothing is fitted here."""
    p_cal = E.platt_apply(E.predict(model, cols, df), **calibration)
    c = M.make_cohort(df, p_cal)
    order = np.lexsort((df["ICULOS"].to_numpy(), df["patient_id"].to_numpy()))  # as make_cohort
    frame = df.iloc[order].reset_index(drop=True)
    assert np.array_equal(frame["ICULOS"].to_numpy(dtype=np.float64), c.iculos)
    return Scored(c=c, frame=frame, ids=frame["patient_id"].to_numpy()[c.offsets])


def summarize(c: M.Cohort) -> Summary:
    first_pos = np.minimum.reduceat(np.where(c.y > 0, c.iculos, np.inf), c.offsets)
    onset = first_pos + M.ONSET_OFFSET
    before_onset = c.iculos < onset[c.codes]
    return Summary(
        septic=c.septic,
        onset=onset,
        peak_pre=np.maximum.reduceat(np.where(before_onset, c.p, -np.inf), c.offsets),
        peak_all=np.maximum.reduceat(c.p, c.offsets),
        hours=c.rows_per_patient.astype(np.float64),
    )


def categorize(
    septic: np.ndarray, peak_pre: np.ndarray, peak_all: np.ndarray, t: float
) -> np.ndarray:
    """Outcome category per patient at threshold t (indices into CATEGORIES)."""
    cat = np.full(len(septic), QUIET, dtype=np.int8)
    cat[~septic & (peak_all >= t)] = FALSE_ALARM
    cat[septic & (peak_all < t)] = MISSED
    cat[septic & (peak_pre < t) & (peak_all >= t)] = LATE
    cat[septic & (peak_pre >= t)] = EARLY
    return cat


def sina_layout(x: np.ndarray, septic: np.ndarray, seed: int = 0) -> np.ndarray:
    """Deterministic jitter in [0, 1]: u * density(x), per group, density on a log-risk grid."""
    lx = np.log10(np.clip(x, RISK_FLOOR, 1.0))
    edges = np.linspace(np.log10(RISK_FLOOR), 0.0, 121)
    u = np.random.default_rng(seed).random(len(x))
    y = np.zeros(len(x))
    for grp in (septic, ~septic):
        if not grp.any():
            continue
        h = np.histogram(lx[grp], bins=edges)[0].astype(np.float64)
        for _ in range(2):
            h = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
        idx = np.clip(np.digitize(lx[grp], edges) - 1, 0, len(h) - 1)
        y[grp] = u[grp] * h[idx] / h.max()
    return y


# ---- threshold grid ----------------------------------------------------------------------------


def grid_row(sc: Scored, s: Summary, t: float, best: np.ndarray, inact: np.ndarray) -> dict:
    """Everything the counters show at threshold t; formulas as in evaluate.metric_values."""
    c = sc.c
    alerts = (c.p >= t).astype(np.int8)
    st = M.per_patient_alert_stats(c, alerts)
    nonsep = ~st["septic"]
    alerted = ~np.isnan(st["lead"])
    counts = np.bincount(categorize(s.septic, s.peak_pre, s.peak_all, t), minlength=5)
    return {
        "t": float(t),
        **{name: int(n) for name, n in zip(CATEGORIES, counts, strict=True)},
        "septic_alerted_before_onset": float(alerted.sum() / st["septic"].sum()),
        "false_alarms_per_patient_day": float(
            24 * st["alert_hours"][nonsep].sum() / st["hours"][nonsep].sum()
        ),
        "alert_rate": float(alerts.mean()),
        "utility": M.normalized_utility(M.utilities(c, alerts), best, inact),
        "median_lead_time_h": float(np.median(st["lead"][alerted])) if alerted.any() else None,
    }


def false_alarms_per_day(c: M.Cohort, t: float) -> float:
    """False alarms per non-septic patient-day at threshold t (cheap: no utility)."""
    alert_hours = np.add.reduceat((c.p >= t).astype(np.float64), c.offsets)
    nonsep = ~c.septic
    return float(24 * alert_hours[nonsep].sum() / c.rows_per_patient[nonsep].sum())


def budget_threshold(c_a: M.Cohort, budget: float = BUDGET_FA_PER_DAY) -> float:
    """Smallest threshold with at most `budget` false alarms per patient-day, on hospital A."""
    for t in np.geomspace(1e-3, 0.9, 600):
        if false_alarms_per_day(c_a, float(t)) <= budget:
            return float(t)
    raise ValueError("no threshold meets the false-alarm budget on hospital A")


def threshold_values(n: int, extras: list[float]) -> list[float]:
    return sorted({float(t) for t in np.geomspace(0.003, 0.5, n)} | {float(t) for t in extras})


# ---- export of Scene 1 arrays, with a check that rounding does not move any dot ----------------


def _round_sig(a: np.ndarray, sig: int) -> list[float]:
    return [float(f"{v:.{sig}g}") for v in a]


def export_cohort(s: Summary, y: np.ndarray, grid: list[dict]) -> dict:
    """Scene 1 arrays at the fewest significant figures that reproduce every grid count."""
    for sig in range(4, 10):
        x_r, pk_r = np.array(_round_sig(s.x, sig)), np.array(_round_sig(s.peak_all, sig))
        if all(
            np.array_equal(
                np.bincount(categorize(s.septic, x_r, pk_r, g["t"]), minlength=5),
                [g[name] for name in CATEGORIES],
            )
            for g in grid
        ):
            return {
                "n": len(x_r),
                "n_septic": int(s.septic.sum()),
                "sig_figs": sig,
                "x": x_r.tolist(),
                "peak_all": pk_r.tolist(),
                "septic": s.septic.astype(int).tolist(),
                "y": [round(v, 3) for v in y.tolist()],
            }
    raise AssertionError("rounded risks changed a category at a grid threshold")


# ---- Scene 2: replay patients ------------------------------------------------------------------


def select_replay(
    s: Summary, stats: dict[str, np.ndarray], t: float, seed: int = 0
) -> list[tuple[int, str]]:
    """Patient indices by the fixed rule in VIZ_PLAN.md section 6.1 (never hand-picked)."""
    rng = np.random.default_rng(seed)
    cats = categorize(s.septic, s.peak_pre, s.peak_all, t)
    ok = (s.hours >= REPLAY_HOURS[0]) & (s.hours <= REPLAY_HOURS[1])
    lead, alert_hours = stats["lead"], stats["alert_hours"]

    def closest(cand: np.ndarray, key: np.ndarray, target: float, k: int) -> list[int]:
        shuffled = rng.permutation(cand)  # random order breaks ties
        return shuffled[np.argsort(np.abs(key[shuffled] - target), kind="stable")][:k].tolist()

    picks: list[tuple[int, str]] = []
    for name, k in REPLAY_SLOTS:
        cand = np.flatnonzero(ok & (cats == CATEGORIES.index(name)))
        if len(cand) == 0:
            continue  # nothing of this kind in a tiny cohort; real data has all five
        if name == "early":
            chosen = closest(cand, lead, float(np.nanmedian(lead)), k)
        elif name == "false_alarm":
            target = float(np.median(alert_hours[cats == FALSE_ALARM]))
            chosen = closest(cand, alert_hours, target, k)
        else:
            chosen = rng.permutation(cand)[:k].tolist()
        picks += [(i, name) for i in chosen]
    return picks


def _nullable(a: np.ndarray, sig: int = 4) -> list[float | None]:
    return [None if not math.isfinite(v) else float(f"{v:.{sig}g}") for v in a]


def top_drivers(contribs: np.ndarray, cols: list[str], k: int = 3) -> list[dict]:
    """Top-k variables that pushed the risk up (summed over a variable's derived features)."""
    by_var = pd.Series(contribs[: len(cols)], index=cols).groupby(base_variable).sum()
    top = by_var[by_var > 0].nlargest(k)
    return [
        {"var": v, "label": LABELS.get(v, v), "contrib": round(float(c), 3)} for v, c in top.items()
    ]


def replay_patient(sc: Scored, s: Summary, idx: int, category: str, t: float, drivers_fn) -> dict:
    c = sc.c
    lo = int(c.offsets[idx])
    hi = lo + int(c.rows_per_patient[idx])
    rows = sc.frame.iloc[lo:hi]
    risk, iculos = c.p[lo:hi], c.iculos[lo:hi]
    alert_rows = np.flatnonzero(risk >= t)
    first_alert = float(iculos[alert_rows[0]]) if len(alert_rows) else None
    onset = float(s.onset[idx]) if s.septic[idx] else None
    focus = int(alert_rows[0]) if len(alert_rows) else int(np.argmax(risk))
    return {
        "id": str(sc.ids[idx]),
        "category": category,
        "iculos": iculos.astype(int).tolist(),
        "risk": _nullable(risk),
        **{v: _nullable(rows[f"{v}_ffill"].to_numpy(dtype=np.float64)) for v in REPLAY_VITALS},
        "onset": onset,
        "first_alert": first_alert,
        "lead_h": (onset - first_alert) if onset is not None and first_alert is not None
        and first_alert < onset else None,
        "focus_iculos": int(iculos[focus]),
        "drivers": drivers_fn(rows, focus),
    }  # fmt: skip


def xgb_drivers_fn(model, cols: list[str]):
    """SHAP (pred_contribs) drivers at one row of a patient's frame, in log-odds of the model."""
    import xgboost as xgb

    def fn(rows: pd.DataFrame, focus: int) -> list[dict]:
        X = rows[cols].to_numpy(np.float32)[focus : focus + 1]
        contribs = model.get_booster().predict(
            xgb.DMatrix(X, feature_names=cols), pred_contribs=True
        )
        return top_drivers(contribs[0], cols)

    return fn


# ---- Scene 3: shift pool, Scene 4: swimmer wall ------------------------------------------------


def shift_pool(sc: Scored, s: Summary, n: int = POOL_SIZE, seed: int = 0) -> list[dict]:
    c = sc.c
    n = min(n, c.n_patients)
    pick = np.sort(np.random.default_rng(seed).choice(c.n_patients, size=n, replace=False))
    out = []
    for i in pick:
        lo = int(c.offsets[i])
        hi = lo + int(c.rows_per_patient[i])
        out.append(
            {
                "id": str(sc.ids[i]),
                "septic": bool(s.septic[i]),
                "risk": _nullable(c.p[lo:hi]),
                "onset_idx": int(s.onset[i] - c.iculos[lo]) if s.septic[i] else None,
            }
        )
    return out


def quantize_risk(p: np.ndarray) -> np.ndarray:
    """uint8 on a log scale: 0 = no data, 1..255 map 1e-4..1."""
    z = (np.log10(np.clip(p, RISK_FLOOR, 1.0)) - np.log10(RISK_FLOOR)) / -np.log10(RISK_FLOOR)
    return (1 + np.rint(254 * z)).astype(np.uint8)


def swimmer_wall(sc: Scored, s: Summary, t: float) -> dict:
    """Septic patients' risk from 48 h before to 6 h after onset, sorted by first-alert time."""
    c = sc.c
    rel = np.arange(WALL_HOURS[0], WALL_HOURS[1] + 1)
    rows, first_alert_rel = [], []
    for i in np.flatnonzero(s.septic):
        lo = int(c.offsets[i])
        hi = lo + int(c.rows_per_patient[i])
        iculos, risk = c.iculos[lo:hi], c.p[lo:hi]
        want = s.onset[i] + rel
        pos = np.clip(np.searchsorted(iculos, want), 0, len(iculos) - 1)
        have = iculos[pos] == want
        q = np.where(have, quantize_risk(risk[pos]), 0).astype(np.uint8)
        alerts = np.flatnonzero(risk >= t)
        first_alert_rel.append(float(iculos[alerts[0]] - s.onset[i]) if len(alerts) else None)
        rows.append(q)
    order = sorted(
        range(len(rows)), key=lambda j: (first_alert_rel[j] is None, first_alert_rel[j] or 0)
    )
    return {
        "hours_from": WALL_HOURS[0],
        "hours_to": WALL_HOURS[1],
        "scale": "0 = no data; 1..255 = 10**(-4 + 4*(q-1)/254)",
        "rows": [base64.b64encode(rows[j].tobytes()).decode() for j in order],
        "first_alert_rel": [first_alert_rel[j] for j in order],
    }


# ---- checks against evaluation.json ------------------------------------------------------------


def check_against_evaluation(cohort: str, row: dict, ev_h: dict, s: Summary) -> list[dict]:
    """Rows of (name, got, expected, ok); the caller fails the build if any is not ok."""
    out = []

    def add(name: str, got, want) -> None:
        ok = (got is None and want is None) or (
            got is not None
            and want is not None
            and math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)
        )
        out.append({"cohort": cohort, "check": name, "got": got, "expected": want, "ok": bool(ok)})

    for k in CHECK_METRICS:
        want = ev_h["alert_rate"] if k == "alert_rate" else ev_h["metrics"][k]["value"]
        add(k, row[k], want)
    add("n_patients", len(s.septic), ev_h["n_patients"])
    add("n_septic_patients", int(s.septic.sum()), ev_h["n_septic_patients"])
    add("early / n_septic == septic_alerted_before_onset", row["early"] / s.septic.sum(),
        ev_h["metrics"]["septic_alerted_before_onset"]["value"])  # fmt: skip
    return out


def print_checks(checks: list[dict]) -> bool:
    print(f"{'cohort':7s} {'check':52s} {'got':>14s} {'expected':>14s}  ok")
    for k in checks:
        g = "None" if k["got"] is None else f"{k['got']:.8g}"
        e = "None" if k["expected"] is None else f"{k['expected']:.8g}"
        print(f"{k['cohort']:7s} {k['check']:52s} {g:>14s} {e:>14s}  {'ok' if k['ok'] else 'FAIL'}")
    return all(k["ok"] for k in checks)


# ---- template ----------------------------------------------------------------------------------


def to_json(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), allow_nan=False)


def inject_data(template: str, data: dict) -> str:
    """Put the data into the template. `</` and `<!--` are escaped so the data cannot end the
    script element or open an HTML comment inside it."""
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"template must contain {PLACEHOLDER} exactly once")
    safe = to_json(data).replace("</", "<\\/").replace("<!--", "<\\!--")
    return template.replace(PLACEHOLDER, safe)


STANDALONE_HEAD = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
    "<style>:root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 "
    "env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}"
    "img{max-width:100%}[hidden]{display:none!important}</style></head><body>"
)


def standalone(fragment: str) -> str:
    """Wrap the page the way the Artifact host does (doctype, viewport, reset), for local viewing
    and browser tests. The fragment itself is what gets published as an Artifact."""
    return STANDALONE_HEAD + fragment + "</body></html>"


# ---- build -------------------------------------------------------------------------------------


def build(
    model_dir: Path, evaluation: Path, features: Path, grid_n: int = 60, drivers_fn=None
) -> dict:
    t0 = time.time()
    ev = json.loads((evaluation / "evaluation.json").read_text())
    thr, calibration = float(ev["threshold"]), ev["calibration"]
    model, cols = E.load_model(model_dir)
    frames = {
        "B": pd.read_parquet(features / "test" / "features.parquet"),
        "A_val": pd.read_parquet(features / "validation" / "features.parquet"),
    }
    ev_keys = {"B": "hospital_B", "A_val": "hospital_A_validation"}
    assert set(frames["B"]["hospital"]) == {"B"} and set(frames["A_val"]["hospital"]) == {"A"}

    scored = {k: score_cohort(df, model, cols, calibration) for k, df in frames.items()}
    del frames
    summaries = {k: summarize(sc.c) for k, sc in scored.items()}
    print(f"scored both cohorts ({time.time() - t0:.0f}s)")

    budget_t = budget_threshold(scored["A_val"].c)  # chosen on hospital A only
    thresholds = threshold_values(grid_n, [thr, budget_t])
    cohorts, grid, checks = {}, {}, []
    for k, sc in scored.items():
        s = summaries[k]
        best, inact = M.reference_utilities(sc.c)
        grid[k] = []
        for i, t in enumerate(thresholds):
            grid[k].append(grid_row(sc, s, t, best, inact))
            if i % 10 == 0:
                print(f"  grid {k}: {i + 1}/{len(thresholds)} ({time.time() - t0:.0f}s)")
        cohorts[k] = export_cohort(s, sina_layout(s.x, s.septic), grid[k])
        row = next(g for g in grid[k] if g["t"] == thr)
        checks += check_against_evaluation(k, row, ev[ev_keys[k]], s)
    if not print_checks(checks):
        raise SystemExit("consistency checks failed; not writing output")

    sc_b, s_b = scored["B"], summaries["B"]
    stats_b = M.per_patient_alert_stats(sc_b.c, (sc_b.c.p >= thr).astype(np.int8))
    drivers = drivers_fn or xgb_drivers_fn(model, cols)
    replay = [
        replay_patient(sc_b, s_b, i, name, thr, drivers)
        for i, name in select_replay(s_b, stats_b, thr)
    ]
    return {
        "meta": {
            "threshold": thr,
            "budget_threshold": budget_t,
            "budget_fa_per_patient_day": BUDGET_FA_PER_DAY,
            "calibration": calibration,
            "categories": CATEGORIES,
            "risk_floor": RISK_FLOOR,
            "evaluation": {
                k: {
                    "n_patients": ev[v]["n_patients"],
                    "n_septic_patients": ev[v]["n_septic_patients"],
                    "alert_rate": ev[v]["alert_rate"],
                    **{m: ev[v]["metrics"][m]["value"] for m in CHECK_METRICS if m != "alert_rate"},
                }
                for k, v in ev_keys.items()
            },
            "checks": checks,
        },
        "thresholds": thresholds,
        "cohorts": cohorts,
        "grid": grid,
        "budget_results": {k: next(g for g in grid[k] if g["t"] == budget_t) for k in grid},
        "replay": replay,
        "shift_pool": shift_pool(sc_b, s_b),
        "wall": swimmer_wall(sc_b, s_b, thr),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True, help="dir with evaluation.json")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=Path("showcase/template.html"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid-n", type=int, default=60)
    args = ap.parse_args()

    data = build(args.model_dir, args.evaluation, args.features, args.grid_n)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data.json").write_text(to_json(data))
    size = (args.out / "data.json").stat().st_size / 1e6
    print(f"wrote {args.out / 'data.json'} ({size:.2f} MB)")
    if args.template.exists():
        page = inject_data(args.template.read_text(), data)
        (args.out / "index.html").write_text(page)  # fragment: this is what is published
        (args.out / "standalone.html").write_text(standalone(page))  # local viewing and tests
        print(f"wrote {args.out / 'index.html'} and standalone.html")
    else:
        print(f"no template at {args.template}; wrote data.json only")
    b = data["budget_results"]
    for k in ("A_val", "B"):
        r = b[k]
        print(
            f"budget threshold {data['meta']['budget_threshold']:.5f} on {k}: "
            f"early {r['early']}, late {r['late']}, missed {r['missed']}, "
            f"septic alerted before onset {r['septic_alerted_before_onset']:.4f}, "
            f"false alarms/day {r['false_alarms_per_patient_day']:.4f}, "
            f"utility {r['utility']:.4f}, "
            f"median lead {r['median_lead_time_h']}"
        )


if __name__ == "__main__":
    main()
