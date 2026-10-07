#!/bin/bash
# Download the Giessen E. coli dataset shared by Ren et al. (2022):
# 809 isolates, a SNP matrix (~61k positions) and R/S labels for CIP, CTX, CTZ, GEN.
#
#   bash scripts/get_data.sh [data_dir]     (default: $AMR_DATA_DIR, else ./data)
set -euo pipefail

DATA_DIR="${1:-${AMR_DATA_DIR:-data}}"
RAW="$DATA_DIR/raw"
# Pinned to a specific commit so the data can't silently change under us.
URL="https://github.com/YunxiaoRen/ML-iAMR/raw/d7688c6b04ca930081a93777ec57b47bf12ff81e/Giessen_dataset.zip"

if [ -f "$RAW/Giessen_dataset/cip_ctx_ctz_gen_multi_data.csv" ]; then
    echo "Data already in $RAW, skipping download."
    exit 0
fi

mkdir -p "$RAW"
echo "Downloading Giessen dataset to $RAW ..."
# Unpack into a temporary folder and only move it into place once it's complete,
# so a half-finished download can never be mistaken for the real thing.
TMP="$RAW/.incomplete"
rm -rf "$TMP" "$RAW/Giessen_dataset"
mkdir -p "$TMP"
curl -L --fail --silent --show-error -o "$TMP/Giessen_dataset.zip" "$URL"
unzip -q "$TMP/Giessen_dataset.zip" -d "$TMP"
mv "$TMP/Giessen_dataset" "$RAW/Giessen_dataset"
rm -rf "$TMP"
echo "Done."
