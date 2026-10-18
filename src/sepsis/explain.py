"""Explainability and subgroup performance (Phase 9, XGBoost built-in SHAP fallback for Clarify).

    python -m sepsis.explain --model-dir outputs/xgboost_full --evaluation outputs/eval_xgboost \
        --features data/features --out outputs/explain

* SHAP: exact TreeSHAP contributions from XGBoost (`pred_contribs`) on a random sample of rows,
  per hospital; reported per derived feature and summed per underlying clinical variable.
* Subgroups: age band and sex, using the SAME frozen calibration and threshold as the main
  evaluation (fit on hospital A validation), with patient-level bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis import evaluate as E
from sepsis.features import STATIC, TIME_VARYING, feature_columns

AGE_BINS = [0, 40, 60, 75, 200]
AGE_LABELS = ["<40", "40-59", "60-74", "75+"]
SEX = {0: "female", 1: "male"}  # PhysioNet 2019: Gender 0 = female, 1 = male
BASES = sorted(TIME_VARYING + STATIC, key=len, reverse=True)  # longest first: Bilirubin_direct


def base_variable(feature: str) -> str:
    return next(b for b in BASES if feature == b or feature.startswith(b + "_"))


def shap_summary(model, cols: list[str], df: pd.DataFrame, n: int, seed: int) -> dict:
    import xgboost as xgb

    sample = df.sample(min(n, len(df)), random_state=seed)
    X = sample[cols].to_numpy(np.float32)
    contribs = model.get_booster().predict(xgb.DMatrix(X, feature_names=cols), pred_contribs=True)
    mean_abs = pd.Series(np.abs(contribs[:, :-1]).mean(axis=0), index=cols)
    by_var = mean_abs.groupby(base_variable).sum()
    total = float(mean_abs.sum())
    return {
        "rows_sampled": len(sample),
        "top_features": {k: float(v) for k, v in mean_abs.nlargest(20).items()},
        "top_variables": {k: float(v) for k, v in by_var.nlargest(15).items()},
        "variable_share_of_total": {k: float(v / total) for k, v in by_var.nlargest(15).items()},
    }


def subgroup_table(df, p_raw, p_cal, thr, n_boot, seed, groups: dict[str, pd.Series]) -> list[dict]:
    """Evaluate each subgroup (a per-row mask) with the frozen threshold."""
    rows = []
    for name, mask in groups.items():
        m = mask.to_numpy()
        if df.loc[m, "SepsisLabel"].sum() == 0 or df.loc[m, "patient_id"].nunique() < 50:
            rows.append({"group": name, "skipped": "too few patients or no septic patients"})
            continue
        res = E.evaluate_hospital(df[m], p_raw[m], p_cal[m], thr, n_boot, seed)
        rows.append({"group": name, **{k: res[k] for k in ("n_patients", "n_septic_patients",
                     "hour_prevalence", "alert_rate", "metrics")}})  # fmt: skip
    return rows


def group_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    age = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS, right=False)
    masks = {f"age {lab}": age == lab for lab in AGE_LABELS}
    for code, label in SEX.items():
        masks[f"sex {label}"] = df["Gender"] == code
    return masks


def ci(m: dict) -> str:
    return f"{m['value']:.3f} ({m['ci_low']:.3f} to {m['ci_high']:.3f})"


def report_md(shap: dict, subs: dict) -> str:
    L = ["# Explainability and subgroup performance", "",
         "SHAP values are XGBoost's exact TreeSHAP (`pred_contribs`), in log-odds units of the "
         "uncalibrated model; Clarify was not used (see docs/future_work.md). Importance = "
         "mean |SHAP| over sampled rows.", ""]  # fmt: skip
    for hosp, key in (
        ("Hospital A validation", "hospital_A_validation"),
        ("Hospital B", "hospital_B"),
    ):
        s = shap[key]
        L += [f"## Top clinical variables, {hosp} ({s['rows_sampled']:,} rows sampled)", "",
              "| variable | mean abs SHAP | share of total |", "|---|---|---|"]  # fmt: skip
        L += [f"| {k} | {v:.4f} | {s['variable_share_of_total'][k]:.1%} |"
              for k, v in s["top_variables"].items()]  # fmt: skip
        L.append("")
    cols = [
        "auroc",
        "auprc",
        "utility",
        "false_alarms_per_patient_day",
        "septic_alerted_before_onset",
    ]
    for hosp, key in (
        ("Hospital A validation", "hospital_A_validation"),
        ("Hospital B", "hospital_B"),
    ):
        L += [f"## Performance by subgroup, {hosp} (95% patient-bootstrap CIs)", "",
              "| group | patients (septic) | hour prev. | AUROC | AUPRC | utility "
              "| false alarms/pt-day | septic alerted |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in subs[key]:
            if "skipped" in r:
                L.append(f"| {r['group']} | {r['skipped']} | | | | | | |")
                continue
            cells = [ci(r["metrics"][c]) for c in cols]
            L.append(f"| {r['group']} | {r['n_patients']:,} ({r['n_septic_patients']:,}) | "
                     f"{r['hour_prevalence']:.4f} | " + " | ".join(cells) + " |")  # fmt: skip
        L.append("")
    L += [
        "Subgroup differences in AUPRC partly reflect different prevalence; compare utility and "
        "false alarms alongside. Wide intervals mean few septic patients: do not over-read them.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True, help="dir with evaluation.json")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-shap", type=int, default=50_000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cols = E.load_model(args.model_dir)
    ev = json.loads((args.evaluation / "evaluation.json").read_text())
    cal, thr = ev["calibration"], ev["threshold"]
    data = {
        "hospital_A_validation": pd.read_parquet(args.features / "validation" / "features.parquet"),
        "hospital_B": pd.read_parquet(args.features / "test" / "features.parquet"),
    }
    assert set(cols) == set(feature_columns(next(iter(data.values()))))

    shap, subs = {}, {}
    for key, df in data.items():
        shap[key] = shap_summary(model, cols, df, args.n_shap, args.seed)
        p_raw = E.predict(model, cols, df)
        p_cal = E.platt_apply(p_raw, **cal)
        subs[key] = subgroup_table(df, p_raw, p_cal, thr, args.n_boot, args.seed, group_masks(df))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "explain.json").write_text(json.dumps({"shap": shap, "subgroups": subs}, indent=2))
    md = report_md(shap, subs)
    (args.out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
