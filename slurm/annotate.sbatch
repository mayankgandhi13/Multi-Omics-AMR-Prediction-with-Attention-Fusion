#!/bin/bash
#SBATCH --job-name=amr-annotate
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/annotate/%x_%A_%a.out
# One genome per array task: Bakta finds every gene and gives it a name.
#
# Like a librarian labelling every book in one library, so that later (Panaroo)
# we can compare libraries shelf by shelf. Only genomes that passed assembly QC
# (passed.txt) get here. Restartable: finished genomes are skipped.
set -eo pipefail
source slurm/common.sh

line=$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))
RUN=$(sed -n "${line}p" "$MORAD_DIR/passed.txt")
OUT="$MORAD_DIR/annotations/$RUN"
echo "Genome $line: $RUN on $(hostname)"
if [ -s "$OUT/$RUN.gff3" ]; then echo "Already annotated, skipping."; exit 0; fi

WORK="/tmp/$USER-bakta-$RUN"
trap 'rm -rf "$WORK"' EXIT
rm -rf "$WORK"

# --skip-crispr: CRISPR arrays (a bacterial immune memory) play no part in our
# resistance gene table, and PilerCR, the tool that finds them, crashed on 23 genomes.
tool bakta bakta --db "$BAKTA_DB" --output "$WORK" --prefix "$RUN" --locus-tag "$RUN" \
    --genus Escherichia --species coli --translation-table 11 \
    --threads "$SLURM_CPUS_PER_TASK" --skip-plot --skip-crispr --tmp-dir /tmp \
    "$MORAD_DIR/assemblies/$RUN.fasta"

mkdir -p "$(dirname "$OUT")"
rm -rf "$OUT.partial" && cp -r "$WORK" "$OUT.partial"
rm -rf "$OUT" && mv "$OUT.partial" "$OUT"   # only a finished annotation gets the real name
echo "Done: $(grep -c "	CDS	" "$OUT/$RUN.gff3") protein-coding genes"
