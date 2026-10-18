"""Write docs/sample_request.json: a hospital B septic patient's first N hours, in the
endpoint's request format (only non-null fields kept, to keep the example readable).

    uv run python scripts/make_sample_request.py [--hours 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sepsis import io

ROOT = Path(__file__).resolve().parents[1]
FIELDS_DROP = {"patient_id", "hospital", "SepsisLabel"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "sample_request.json")
    args = ap.parse_args()
    df = io.read_hospital(ROOT / "data" / "parquet", "B")
    septic = df.groupby("patient_id")["SepsisLabel"].max()
    pid = septic[septic == 1].index[0]
    rows = df[df.patient_id == pid].sort_values("ICULOS").head(args.hours)
    records = [
        {k: (int(v) if k == "ICULOS" else float(v)) for k, v in r.items()
         if k not in FIELDS_DROP and v == v}
        for r in rows.to_dict("records")
    ]  # fmt: skip
    payload = {"patient_id": pid, "records": records, "all_hours": False}
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out} ({pid}, {len(records)} hours)")


if __name__ == "__main__":
    main()
