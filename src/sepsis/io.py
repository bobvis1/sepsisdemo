"""Read PhysioNet 2019 .psv files and read/write the consolidated Parquet layout.

Layout: <root>/hospital=A/data.parquet and <root>/hospital=B/data.parquet. Each file carries
`patient_id` and `hospital` columns, so a single file can be read on its own.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HOSPITAL_DIRS = {"A": "training_setA", "B": "training_setB"}


def patient_id(hospital: str, psv: Path) -> str:
    """Globally unique id: file stems (p000001) repeat across the two hospitals."""
    return f"{hospital}_{psv.stem}"


def list_psv(raw_dir: Path, hospital: str, limit: int | None = None) -> list[Path]:
    files = sorted((raw_dir / HOSPITAL_DIRS[hospital]).glob("p*.psv"))
    return files[:limit] if limit else files


def read_psv(path: Path, hospital: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", na_values=["NaN"])
    df.insert(0, "patient_id", patient_id(hospital, path))
    df.insert(1, "hospital", hospital)
    return df


def read_hospital_raw(raw_dir: Path, hospital: str, limit: int | None = None) -> pd.DataFrame:
    frames = [read_psv(p, hospital) for p in list_psv(raw_dir, hospital, limit)]
    return pd.concat(frames, ignore_index=True)


def parquet_path(root: Path, hospital: str) -> Path:
    return root / f"hospital={hospital}" / "data.parquet"


def write_hospital(df: pd.DataFrame, root: Path, hospital: str) -> Path:
    path = parquet_path(root, hospital)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_hospital(
    root: Path | str, hospital: str, columns: list[str] | None = None
) -> pd.DataFrame:
    return pd.read_parquet(parquet_path(Path(root), hospital), columns=columns)
