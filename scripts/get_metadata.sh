#!/bin/bash
# Fetch the isolate table for the Moradigaravand et al. (2018) E. coli collection:
# one row per isolate with its ENA run accession, year, and S/I/R phenotypes
# for 12 antibiotics (S1 Table of the paper, doi:10.1371/journal.pcbi.1006258).
#
#   bash scripts/get_metadata.sh [out_dir]   (default: $MORAD_DIR)
# Writes metadata.csv and runs.txt (one ENA run accession per line, same order).
set -euo pipefail

OUT="${1:-${MORAD_DIR:?set MORAD_DIR or pass an output dir}}"
URL="https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=10.1371/journal.pcbi.1006258.s010"

mkdir -p "$OUT"
if [ ! -s "$OUT/metadata.csv" ]; then
    curl -L --fail --silent --show-error -o "$OUT/metadata.csv.partial" "$URL"
    mv "$OUT/metadata.csv.partial" "$OUT/metadata.csv"
fi
# Column 3 is "Lane.accession", the ENA run ID (e.g. ERR434268).
tail -n +2 "$OUT/metadata.csv" | cut -d, -f3 > "$OUT/runs.txt"
echo "$(wc -l < "$OUT/runs.txt") isolates listed in $OUT/runs.txt"
