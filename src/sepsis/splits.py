"""Patient-level splits. Hospital A -> train/validation (stratified on ever-septic);
hospital B -> test, never used for fitting or tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: hospital and whether they ever have a positive label."""
    g = df.groupby("patient_id", sort=True)
    return pd.DataFrame(
        {"hospital": g["hospital"].first(), "ever_septic": g["SepsisLabel"].max().astype(bool)}
    )


def assign_splits(patients: pd.DataFrame, val_frac: float = 0.2, seed: int = 42) -> pd.Series:
    """Map patient_id -> 'train' | 'validation' | 'test'. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    split = pd.Series("test", index=patients.index, name="split")
    a = patients[patients.hospital == "A"]
    for _, stratum in a.groupby("ever_septic"):
        ids = np.array(sorted(stratum.index))
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_frac))
        split.loc[ids[:n_val]] = "validation"
        split.loc[ids[n_val:]] = "train"
    unknown = set(patients.hospital) - {"A", "B"}
    if unknown:
        raise ValueError(f"unexpected hospitals: {unknown}")
    return split
