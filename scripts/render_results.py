"""Fill the results blocks in README.md and docs/model_card.md from evaluation outputs.

Blocks are delimited by `<!-- results:NAME:start -->` / `<!-- results:NAME:end -->` and are
overwritten in place, so numbers are never hand-copied.

    uv run python scripts/render_results.py [--eval-dir outputs] [--source "local run"]

Reads <eval-dir>/eval_xgboost, eval_logreg (evaluation.json) and explain/explain.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("brier_calibrated", "Brier (calibrated)"),
    ("utility", "Challenge utility (normalised)"),
    ("median_lead_time_h", "Median lead time, h"),
    ("septic_alerted_before_onset", "Septic patients alerted before onset"),
    ("false_alarms_per_patient_day", "False alarms per non-septic patient-day"),
]


def ci(m: dict, d: int = 3) -> str:
    return f"{m['value']:.{d}f} ({m['ci_low']:.{d}f}-{m['ci_high']:.{d}f})"


def headline(evs: dict[str, dict], source: str) -> str:
    x = evs["xgboost"]
    L = [
        f"_Source: {source}. 95% CIs from {x['n_bootstrap']} patient-level bootstrap resamples. "
        "Calibration and alert threshold fit on hospital A validation only; hospital B is an "
        "untouched external test set._",
        "",
        "| Metric | XGBoost, A validation | XGBoost, B (external) "
        "| Logistic regression, A validation | Logistic regression, B |",
        "|---|---|---|---|---|",
    ]
    a, b = x["hospital_A_validation"], x["hospital_B"]
    base = evs["logreg"]
    L.append(
        f"| Patients (septic) | {a['n_patients']:,} ({a['n_septic_patients']:,}) "
        f"| {b['n_patients']:,} ({b['n_septic_patients']:,}) "
        f"| {a['n_patients']:,} ({a['n_septic_patients']:,}) "
        f"| {b['n_patients']:,} ({b['n_septic_patients']:,}) |"
    )
    L.append(
        f"| Hour prevalence | {a['hour_prevalence']:.4f} | {b['hour_prevalence']:.4f} "
        f"| {a['hour_prevalence']:.4f} | {b['hour_prevalence']:.4f} |"
    )
    for key, label in ROWS:
        d = 1 if key == "median_lead_time_h" else 2 if key.startswith("false_alarms") else 3
        cells = [
            ci(ev[h]["metrics"][key], d)
            for ev in (x, base)
            for h in ("hospital_A_validation", "hospital_B")
        ]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", f"XGBoost alert threshold: {x['threshold']:.4f} (calibrated probability)."]
    return "\n".join(L)


def shap_block(ex: dict) -> str:
    a = ex["shap"]["hospital_A_validation"]["variable_share_of_total"]
    b = ex["shap"]["hospital_B"]["variable_share_of_total"]
    L = ["| Variable | Share of total |SHAP|, A validation | Share, B |", "|---|---|---|"]
    for k in list(a)[:8]:
        L.append(f"| {k} | {a[k]:.1%} | {b.get(k, 0):.1%} |")
    return "\n".join(L)


def subgroup_block(ex: dict) -> str:
    L = [
        "| Group | A validation: patients (septic) | A AUPRC | A utility | B: patients (septic)"
        " | B AUPRC | B utility |",
        "|---|---|---|---|---|---|---|",
    ]
    for ra, rb in zip(
        ex["subgroups"]["hospital_A_validation"], ex["subgroups"]["hospital_B"], strict=True
    ):
        if "skipped" in ra or "skipped" in rb:
            continue
        L.append(
            f"| {ra['group']} | {ra['n_patients']:,} ({ra['n_septic_patients']:,}) "
            f"| {ci(ra['metrics']['auprc'])} | {ci(ra['metrics']['utility'])} "
            f"| {rb['n_patients']:,} ({rb['n_septic_patients']:,}) "
            f"| {ci(rb['metrics']['auprc'])} | {ci(rb['metrics']['utility'])} |"
        )
    return "\n".join(L)


def fill(path: Path, blocks: dict[str, str]) -> int:
    text, n = path.read_text(), 0
    for name, body in blocks.items():
        pat = re.compile(
            rf"(<!-- results:{name}:start -->\n).*?(\n<!-- results:{name}:end -->)", re.S
        )
        text, k = pat.subn(lambda m, b=body: m.group(1) + b + m.group(2), text)
        n += k
    path.write_text(text)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "outputs")
    ap.add_argument(
        "--source", default="local run (models trained locally; SageMaker pipeline run pending)"
    )
    args = ap.parse_args()
    d = args.eval_dir
    evs = {
        m: json.loads((d / f"eval_{m}" / "evaluation.json").read_text())
        for m in ("xgboost", "logreg")
    }
    ex = json.loads((d / "explain" / "explain.json").read_text())
    blocks = {
        "headline": headline(evs, args.source),
        "shap": shap_block(ex),
        "subgroups": subgroup_block(ex),
    }
    for f in (ROOT / "README.md", ROOT / "docs" / "model_card.md"):
        if f.exists():
            print(f"{f.name}: {fill(f, blocks)} block(s) updated")


if __name__ == "__main__":
    main()
