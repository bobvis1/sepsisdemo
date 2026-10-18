from pathlib import Path

from sepsis import io

HEADER = "HR|Age|ICULOS|SepsisLabel\n"


def _write_patient(raw: Path, set_name: str, stem: str, rows: list[str]) -> None:
    d = raw / set_name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.psv").write_text(HEADER + "\n".join(rows) + "\n")


def test_patient_ids_unique_across_hospitals_and_nan_parsed(tmp_path):
    _write_patient(tmp_path, "training_setA", "p000001", ["80|60|1|0", "NaN|60|2|1"])
    _write_patient(tmp_path, "training_setB", "p000001", ["70|50|1|0"])
    a = io.read_hospital_raw(tmp_path, "A")
    b = io.read_hospital_raw(tmp_path, "B")
    assert a.patient_id.iloc[0] == "A_p000001" and b.patient_id.iloc[0] == "B_p000001"
    assert a.HR.isna().sum() == 1
    assert set(a.hospital) == {"A"}


def test_parquet_round_trip_and_limit(tmp_path):
    for i in range(3):
        _write_patient(
            tmp_path / "raw", "training_setA", f"p00000{i + 1}", ["80|60|1|0", "81|60|2|0"]
        )
    df = io.read_hospital_raw(tmp_path / "raw", "A", limit=2)
    assert df.patient_id.nunique() == 2 and len(df) == 4
    io.write_hospital(df, tmp_path / "pq", "A")
    back = io.read_hospital(tmp_path / "pq", "A")
    assert back.equals(df)
