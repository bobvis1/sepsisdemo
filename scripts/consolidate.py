"""Merge per-patient .psv files into Parquet partitioned by hospital, reconcile counts against
the raw files, and optionally upload to S3 (SSE-KMS).

    uv run python scripts/consolidate.py --limit 500            # local sample, no AWS
    uv run python scripts/consolidate.py                        # full local run
    uv run python scripts/consolidate.py --upload               # full run + upload to S3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sepsis import io

ROOT = Path(__file__).resolve().parents[1]


def raw_counts(raw_dir: Path, hospital: str, limit: int | None) -> dict:
    """Count patients and data rows straight from the files (line count minus header),
    independent of pandas so the reconciliation is a real check."""
    files = io.list_psv(raw_dir, hospital, limit)
    rows = 0
    for f in files:
        with f.open("rb") as fh:
            rows += sum(1 for _ in fh) - 1
    return {"patients": len(files), "rows": rows}


def upload(out_dir: Path, prefix: str) -> None:
    from sepsis.config import boto_session, load_infra

    infra = load_infra()
    s3 = boto_session().client("s3")
    extra = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": infra.kms_key_arn}
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            key = f"{prefix}/{f.relative_to(out_dir)}"
            s3.upload_file(str(f), infra.bucket, key, ExtraArgs=extra)
            print(f"uploaded s3://<bucket>/{key}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: data/parquet (data/parquet_sample with --limit)",
    )
    ap.add_argument("--limit", type=int, default=None, help="patients per hospital (local sample)")
    ap.add_argument("--upload", action="store_true", help="upload to s3://<bucket>/data/parquet/")
    args = ap.parse_args()
    out_dir = args.out_dir or ROOT / "data" / ("parquet_sample" if args.limit else "parquet")

    manifest = {}
    for h in io.HOSPITAL_DIRS:
        raw = raw_counts(args.raw_dir, h, args.limit)
        df = io.read_hospital_raw(args.raw_dir, h, args.limit)
        path = io.write_hospital(df, out_dir, h)
        back = io.read_hospital(out_dir, h)
        got = {"patients": int(back["patient_id"].nunique()), "rows": len(back)}
        ok = got == raw
        manifest[h] = {"raw": raw, "parquet": got, "reconciled": ok}
        print(f"hospital {h}: raw={raw} parquet={got} -> {'OK' if ok else 'MISMATCH'} ({path})")
        if not ok:
            raise SystemExit(f"Row/patient counts do not reconcile for hospital {h}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.upload:
        if args.limit:
            raise SystemExit("Refusing to upload a --limit sample as the dataset")
        upload(out_dir, "data/parquet")


if __name__ == "__main__":
    main()
