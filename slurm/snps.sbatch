#!/bin/bash
#SBATCH --job-name=amr-snps
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/snps/%x_%A_%a.out
# One isolate per array task: Snippy lines its reads up against the K-12
# reference genome and lists every single-letter difference (SNP).
#
# This is Phase 1's view of the genome (typos against the reference edition),
# computed for the same isolates as the Phase 2 gene table, so the two can be
# compared head to head and combined. Restartable: finished isolates are skipped.
set -eo pipefail
source slurm/common.sh

line=$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))
RUN=$(sed -n "${line}p" "$MORAD_DIR/passed.txt")
OUT="$MORAD_DIR/snps/$RUN.tab"
echo "Isolate $line: $RUN on $(hostname)"
if [ -s "$OUT" ]; then echo "Already called, skipping."; exit 0; fi

READS="$MORAD_DIR/reads/$RUN"
R1=$(ls "$READS"/*_1.fastq.gz | head -1); R2=$(ls "$READS"/*_2.fastq.gz | head -1)
WORK="/tmp/$USER-snippy-$RUN"
trap 'rm -rf "$WORK"' EXIT
rm -rf "$WORK"

tool snippy snippy --R1 "$R1" --R2 "$R2" --ref "$MORAD_DIR/reference/MG1655.gbk" \
    --outdir "$WORK" --prefix "$RUN" --cpus "$SLURM_CPUS_PER_TASK" --ram 14 --tmpdir /tmp

mkdir -p "$(dirname "$OUT")"
cp "$WORK/$RUN.tab" "$OUT.partial"
mv "$OUT.partial" "$OUT"
echo "Done: $(awk -F'\t' '$3=="snp"' "$OUT" | wc -l) SNPs vs K-12"
