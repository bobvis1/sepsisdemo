#!/usr/bin/env bash
# Download PhysioNet/CinC Challenge 2019 training sets A and B (open access, CC BY 4.0)
# into data/raw/training_set{A,B}/ as one .psv per patient. Resumable: existing files are skipped.
# Usage: scripts/download_data.sh [--limit N]   (N = patients per set, for a quick local sample)
set -euo pipefail

BASE="https://physionet.org/files/challenge-2019/1.0.0/training"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${DATA_DIR:-$ROOT/data/raw}"
JOBS="${JOBS:-8}"
LIMIT=0
[[ "${1:-}" == "--limit" ]] && LIMIT="${2:?--limit needs a number}"

for set in training_setA training_setB; do
  dir="$OUT/$set"
  mkdir -p "$dir"
  curl -fsS "$BASE/$set/" | grep -oE 'p[0-9]{6}\.psv' | sort -u > "$dir/.index"
  [[ "$LIMIT" -gt 0 ]] && head -n "$LIMIT" "$dir/.index" > "$dir/.index.tmp" && mv "$dir/.index.tmp" "$dir/.index"
  expected=$(wc -l < "$dir/.index" | tr -d ' ')
  echo "$set: $expected files listed"
  # Download only what is missing; write to .part then rename so interrupted runs leave no partial files.
  while read -r f; do [[ -s "$dir/$f" ]] || echo "$f"; done < "$dir/.index" |
    xargs -P "$JOBS" -I{} sh -c \
      'curl -fsS --retry 8 --retry-all-errors --retry-delay 3 -o "$1/$2.part" "$3/$4/$2" && mv "$1/$2.part" "$1/$2"' \
      _ "$dir" {} "$BASE" "$set"
  actual=$(find "$dir" -name 'p*.psv' | wc -l | tr -d ' ')
  echo "$set: $actual / $expected downloaded"
  [[ "$actual" -eq "$expected" ]] || { echo "ERROR: count mismatch in $set" >&2; exit 1; }
done
