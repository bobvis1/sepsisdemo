"""Feature-building entry point. Runs identically locally and as a SageMaker Processing job.

    python -m sepsis.process --input data/parquet_sample --output data/features_sample
    # in SageMaker: defaults are /opt/ml/processing/input and /opt/ml/processing/output

Reads the consolidated Parquet (hospital=A, hospital=B), assigns patient-level splits, builds
causal features, and writes <output>/{train,validation,test}/features.parquet plus splits.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sepsis import io
from sepsis.features import build_features
from sepsis.splits import assign_splits, patient_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/opt/ml/processing/input"))
    ap.add_argument("--output", type=Path, default=Path("/opt/ml/processing/output"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd

    raw = pd.concat([io.read_hospital(args.input, h) for h in io.HOSPITAL_DIRS], ignore_index=True)
    split = assign_splits(patient_table(raw), args.val_frac, args.seed)
    feats = build_features(raw)
    for name, part in feats.groupby(feats["patient_id"].map(split)):
        d = args.output / str(name)
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / "features.parquet", index=False)
        print(f"{name}: {part.patient_id.nunique()} patients, {len(part)} rows")
    args.output.mkdir(parents=True, exist_ok=True)
    split.rename_axis("patient_id").reset_index().to_csv(args.output / "splits.csv", index=False)


if __name__ == "__main__":
    main()
