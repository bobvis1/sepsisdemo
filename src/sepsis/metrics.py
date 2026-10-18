"""Metrics for hourly sepsis alerts: official utility, lead time, false alarms, and fast
patient-weighted ranking metrics for the bootstrap.

Official utility: per-patient utilities come from the vendored challenge code
(third_party/evaluation_2019); we only add the cohort normalisation, which is a ratio of sums
and so can be bootstrapped over patients without re-running the slow per-row loop.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

ONSET_OFFSET = 6  # label is 1 from 6 h before clinical onset: onset = first label-1 hour + 6


@cache
def official():
    """Load the vendored challenge scoring module (SEPSIS_THIRD_PARTY overrides the location)."""
    root = Path(
        os.environ.get("SEPSIS_THIRD_PARTY", Path(__file__).resolve().parents[2] / "third_party")
    )
    path = root / "evaluation_2019" / "evaluate_sepsis_score.py"
    spec = importlib.util.spec_from_file_location("evaluate_sepsis_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Cohort:
    """Rows sorted by (patient, ICULOS), with per-patient offsets for vectorised reductions."""

    y: np.ndarray  # SepsisLabel per row
    p: np.ndarray  # predicted probability per row
    iculos: np.ndarray
    codes: np.ndarray  # patient index per row, 0..n_patients-1
    offsets: np.ndarray  # first row of each patient

    @property
    def n_patients(self) -> int:
        return len(self.offsets)

    @property
    def rows_per_patient(self) -> np.ndarray:
        return np.diff(np.append(self.offsets, len(self.y)))

    @property
    def septic(self) -> np.ndarray:
        return np.maximum.reduceat(self.y, self.offsets) > 0


def make_cohort(df: pd.DataFrame, p: np.ndarray) -> Cohort:
    order = np.lexsort((df["ICULOS"].to_numpy(), df["patient_id"].to_numpy()))
    d = df.iloc[order]
    codes, _ = pd.factorize(d["patient_id"], sort=False)
    offsets = np.flatnonzero(np.diff(codes, prepend=-1))
    return Cohort(
        y=d["SepsisLabel"].to_numpy().astype(np.int8),
        p=np.asarray(p)[order],
        iculos=d["ICULOS"].to_numpy().astype(np.float64),
        codes=codes,
        offsets=offsets,
    )


def _split(c: Cohort, x: np.ndarray) -> list[np.ndarray]:
    return np.split(x, c.offsets[1:])


def best_predictions(labels: np.ndarray, dt_early=-12, dt_optimal=-6, dt_late=3) -> np.ndarray:
    """Optimal alerts per patient, exactly as in upstream evaluate_sepsis_score()."""
    n = len(labels)
    best = np.zeros(n)
    if np.any(labels):
        t_sepsis = np.argmax(labels) - dt_optimal
        best[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, n)] = 1
    return best


def utilities(c: Cohort, alerts: np.ndarray) -> np.ndarray:
    """Per-patient observed utility for the given binary alerts (official function)."""
    f = official().compute_prediction_utility
    return np.array(
        [
            f(y, a, check_errors=False)
            for y, a in zip(_split(c, c.y), _split(c, alerts), strict=True)
        ]
    )


def reference_utilities(c: Cohort) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient (best, inaction) utilities; independent of the model and threshold."""
    f = official().compute_prediction_utility
    best, inaction = [], []
    for y in _split(c, c.y):
        best.append(f(y, best_predictions(y), check_errors=False))
        inaction.append(f(y, np.zeros(len(y)), check_errors=False))
    return np.array(best), np.array(inaction)


def normalized_utility(
    obs: np.ndarray, best: np.ndarray, inact: np.ndarray, w: np.ndarray | None = None
) -> float:
    """(observed - inaction) / (best - inaction) over a cohort; w = patient resample counts."""
    w = np.ones(len(obs)) if w is None else w
    return float(((w * obs).sum() - (w * inact).sum()) / ((w * best).sum() - (w * inact).sum()))


def per_patient_alert_stats(c: Cohort, alerts: np.ndarray) -> dict[str, np.ndarray]:
    """Per-patient: lead time (h, NaN unless septic and first alert before onset), alert hours,
    hours, and whether the patient was positive at their first row."""
    inf = np.inf
    first_alert = np.minimum.reduceat(np.where(alerts > 0, c.iculos, inf), c.offsets)
    first_pos = np.minimum.reduceat(np.where(c.y > 0, c.iculos, inf), c.offsets)
    onset = first_pos + ONSET_OFFSET
    septic = c.septic
    lead = np.where(septic & (first_alert < onset), onset - first_alert, np.nan)
    return {
        "lead": lead,
        "alert_hours": np.add.reduceat(alerts.astype(np.float64), c.offsets),
        "hours": c.rows_per_patient.astype(np.float64),
        "septic": septic,
        "pos_at_first_row": c.y[c.offsets] > 0,
    }


def weighted_median(values: np.ndarray, counts: np.ndarray) -> float:
    v = np.repeat(values, counts.astype(int))
    return float(np.median(v)) if len(v) else float("nan")


class RankMetrics:
    """AUROC / AUPRC / Brier with per-row weights, sorting once. Weights are patient resample
    counts expanded to rows, so each bootstrap draw costs a few cumsums instead of a sort.
    Matches sklearn's roc_auc_score / average_precision_score (verified in tests)."""

    def __init__(self, y: np.ndarray, p: np.ndarray):
        self.order = np.argsort(-p, kind="stable")
        self.y = y[self.order].astype(np.float64)
        ps = p[self.order]
        self.last = np.append(
            np.flatnonzero(np.diff(ps)), len(ps) - 1
        )  # last row of each tie group
        self.err2 = (p[self.order] - self.y) ** 2

    def auroc_auprc_brier(self, w: np.ndarray | None = None) -> tuple[float, float, float]:
        ws = np.ones_like(self.y) if w is None else w[self.order]
        tps = np.cumsum(ws * self.y)[self.last]
        fps = np.cumsum(ws * (1 - self.y))[self.last]
        pos, neg = tps[-1], fps[-1]
        tpr = np.append(0.0, tps / pos)
        fpr = np.append(0.0, fps / neg)
        auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        precision = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) > 0)
        auprc = float(np.sum(np.diff(tpr) * precision))
        brier = float((ws * self.err2).sum() / ws.sum())
        return auroc, auprc, brier


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Equal-count bins by predicted probability: mean predicted vs observed event rate."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [
        {"mean_predicted": float(p[idx == b].mean()), "observed": float(y[idx == b].mean()),
         "n": int((idx == b).sum())}
        for b in range(len(edges) - 1) if (idx == b).any()
    ]  # fmt: skip
