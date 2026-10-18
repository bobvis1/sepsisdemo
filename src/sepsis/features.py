"""Causal feature engineering for hourly sepsis prediction.

Every feature at row t depends only on rows <= t of the same patient: forward-fill, trailing
cumulative sums, and shifts by a positive lag. No back-fill, no interpolation, no whole-stay
aggregates. Assumes rows are one per ICU hour in ICULOS order (checked in data_summary.md);
windows are counted in rows.

Fitted preprocessing (scaling, imputation constants) is deliberately NOT here: it must be fit
on the hospital A training fold only and lives in train.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
    "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]  # fmt: skip
TIME_VARYING = VITALS + LABS
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
WINDOWS = (6, 12)
ID, LABEL = "patient_id", "SepsisLabel"


def _trailing_sum(x: pd.DataFrame, pid: pd.Series, w: int) -> pd.DataFrame:
    """Sum over the last w rows (including the current one) within each patient."""
    c = x.groupby(pid, sort=False).cumsum()
    lagged = c.groupby(pid, sort=False).shift(w).fillna(0.0)
    return c - lagged


def _window_stats(y: pd.DataFrame, pid: pd.Series, t: pd.Series, w: int) -> dict[str, pd.DataFrame]:
    """Trailing mean and least-squares slope (per hour) of forward-filled y against time t."""
    valid = y.notna().astype("float64")
    y0 = y.fillna(0.0)
    tt = t.to_numpy()[:, None]
    n = _trailing_sum(valid, pid, w)
    st = _trailing_sum(valid * tt, pid, w)
    stt = _trailing_sum(valid * tt**2, pid, w)
    sy = _trailing_sum(y0, pid, w)
    sty = _trailing_sum(y0 * tt, pid, w)
    mean = sy / n.where(n > 0)
    denom = n * stt - st**2
    slope = (n * sty - st * sy) / denom.where((n >= 2) & (denom > 1e-9))
    return {"mean": mean, "slope": slope}


def _hours_since_measured(raw: pd.DataFrame, pid: pd.Series, t: pd.Series) -> pd.DataFrame:
    """Hours since the variable was last observed (0 if observed now); NaN if never yet."""
    t_obs = raw.notna().mul(t, axis=0).where(raw.notna())
    last = t_obs.groupby(pid, sort=False).ffill()
    return last.rsub(t, axis=0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per input row, preserving order.

    Input needs patient_id, ICULOS, SepsisLabel and the 40 raw variables. Output columns:
    ids/label passthrough, static fields, `<var>_ffill`, `<var>_measured`, `<var>_hrs_since`,
    and `<vital>_mean{w}` / `<vital>_slope{w}` for w in WINDOWS.
    """
    df = df.sort_values([ID, "ICULOS"], kind="stable").reset_index(drop=True)
    pid, t = df[ID], df["ICULOS"].astype("float64")
    raw = df[TIME_VARYING]
    ffill = raw.groupby(pid, sort=False).ffill()

    parts = [
        df[[ID, "hospital", LABEL] + STATIC] if "hospital" in df else df[[ID, LABEL] + STATIC],
        ffill.add_suffix("_ffill"),
        raw.notna().astype("int8").add_suffix("_measured"),
        _hours_since_measured(raw, pid, t).add_suffix("_hrs_since"),
    ]
    for w in WINDOWS:
        stats = _window_stats(ffill[VITALS], pid, t, w)
        for name, frame in stats.items():
            parts.append(frame.add_suffix(f"_{name}{w}"))
    out = pd.concat(parts, axis=1)
    return out.astype({c: np.float32 for c in out.select_dtypes("float64").columns})


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model input columns: everything except ids, hospital, and the label."""
    return [c for c in features.columns if c not in (ID, "hospital", LABEL)]
