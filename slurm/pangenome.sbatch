#!/bin/bash
#SBATCH --job-name=amr-pangenome
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Panaroo: compare every annotated genome and build one gene presence/absence table.
#
# Like merging 1,900 library catalogues into a single checklist of which library
# owns which book, while recognising the same book even when one copy is torn
# (a gene split across two contigs) or missing its label (a gene Bakta skipped).
#
#   sbatch slurm/pangenome.sbatch                 # every annotated genome
#   sbatch --export=ALL,N=100 slurm/pangenome.sbatch   # first 100: a memory/time test
set -eo pipefail
source slurm/common.sh

OUT="$MORAD_DIR/pangenome${N:+_test$N}"
WORK="/tmp/$USER-panaroo-$SLURM_JOB_ID"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"

# Input list: genomes that passed QC and have a finished annotation.
# (Build the full list first, then trim: piping straight into `head` would end the
# loop early with SIGPIPE, which strict mode treats as a failure.)
while read -r run; do
    gff="$MORAD_DIR/annotations/$run/$run.gff3"
    if [ -s "$gff" ]; then echo "$gff"; fi
done < "$MORAD_DIR/passed.txt" > "$WORK/all_inputs.txt"
head -n "${N:-1000000}" "$WORK/all_inputs.txt" > "$WORK/inputs.txt"
echo "Building pangenome from $(wc -l < "$WORK/inputs.txt") genomes"

# --clean-mode strict: remove genes that look like contamination or assembly junk.
#   Recommended when the gene table feeds an association/prediction model,
#   because a fake "present here, absent there" gene is a fake feature.
tool panaroo panaroo -i "$WORK/inputs.txt" -o "$WORK/out" \
    --clean-mode strict --remove-invalid-genes -t "$SLURM_CPUS_PER_TASK"

rm -rf "$OUT.partial" && cp -r "$WORK/out" "$OUT.partial"
rm -rf "$OUT" && mv "$OUT.partial" "$OUT"
echo "Done: $(head -1 "$OUT/gene_presence_absence.Rtab" | tr '\t' '\n' | tail -n +2 | wc -l) genomes x $(($(wc -l < "$OUT/gene_presence_absence.Rtab") - 1)) gene families"
