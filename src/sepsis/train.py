"""Train the logistic-regression baseline or XGBoost on hospital A features.

Runs identically locally and as a SageMaker training job (channels under /opt/ml/input/data,
model to /opt/ml/model). Hyperparameters arrive as `--name value` arguments, or, in a tuning job
(which cannot override the container entrypoint), as SageMaker's hyperparameters.json via the
`--from-sm-config` flag that container/train passes.

    python -m sepsis.train --model xgboost --train data/features_500/train \
        --validation data/features_500/validation --model-dir outputs/xgb

Only the hospital A train fold is used to fit anything (scaler, imputation constants, trees);
the validation fold is used for early stopping and reporting. Hospital B is never read here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sepsis.features import LABEL, feature_columns

SM = Path("/opt/ml")


def load(channel: Path) -> pd.DataFrame:
    return pd.read_parquet(channel / "features.parquet")


def build_logreg(C: float, seed: int):
    # Imputer and scaler live inside the pipeline, so they are fit on the train fold only.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed),
    )


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


def sm_hyperparameters_argv(path: Path = SM / "input/config/hyperparameters.json") -> list[str]:
    """Turn SageMaker's hyperparameters.json into `--name value` arguments.

    Values are strings, sometimes JSON-quoted. SageMaker adds housekeeping keys (`_tuning_*`,
    `sagemaker_*`) that are not ours; skip them.
    """
    argv: list[str] = []
    for key, value in json.loads(path.read_text()).items():
        if key.startswith(("_", "sagemaker_")):
            continue
        argv += [f"--{key.replace('_', '-')}", str(value).strip('"')]
    return argv


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", "xgboost"], required=True)
    ap.add_argument("--train", type=Path, default=SM / "input/data/train")
    ap.add_argument("--validation", type=Path, default=SM / "input/data/validation")
    ap.add_argument(
        "--model-dir", type=Path, default=Path(os.environ.get("SM_MODEL_DIR", SM / "model"))
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)  # logreg
    ap.add_argument("--max-depth", type=int, default=6)  # xgboost
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--num-round", type=int, default=500)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    args = ap.parse_args(argv)

    tr, va = load(args.train), load(args.validation)
    cols = feature_columns(tr)
    Xtr, ytr = tr[cols].to_numpy(np.float32), tr[LABEL].to_numpy()
    Xva, yva = va[cols].to_numpy(np.float32), va[LABEL].to_numpy()
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    params: dict = {"model": args.model, "seed": args.seed, "n_features": len(cols)}

    if args.model == "logreg":
        model = build_logreg(args.C, args.seed).fit(Xtr, ytr)
        pva = model.predict_proba(Xva)[:, 1]
        params["C"] = args.C
    else:
        import xgboost as xgb  # lazy: needs the OpenMP runtime, absent on a bare Mac

        model = xgb.XGBClassifier(
            n_estimators=args.num_round,
            learning_rate=args.eta,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_weight=args.min_child_weight,
            scale_pos_weight=pos_weight,  # explicit class-imbalance handling
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=args.early_stopping_rounds,
            random_state=args.seed,
            n_jobs=os.cpu_count(),
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pva = model.predict_proba(Xva)[:, 1]
        params |= {
            "eta": args.eta, "max_depth": args.max_depth, "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree, "min_child_weight": args.min_child_weight,
            "scale_pos_weight": pos_weight, "best_iteration": int(model.best_iteration),
        }  # fmt: skip

    metrics = {f"validation:{k}": v for k, v in evaluate(yva, pva).items()}
    metrics["train:auroc"] = evaluate(ytr, model.predict_proba(Xtr)[:, 1])["auroc"]
    # One line per metric: the SageMaker metric_definitions regexes match these.
    for k, v in metrics.items():
        print(f"{k}={v:.6f}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": cols}, args.model_dir / "model.joblib")
    (args.model_dir / "params.json").write_text(json.dumps(params, indent=2))
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    cli = sys.argv[1:]
    main(sm_hyperparameters_argv() if cli == ["--from-sm-config"] else cli)
