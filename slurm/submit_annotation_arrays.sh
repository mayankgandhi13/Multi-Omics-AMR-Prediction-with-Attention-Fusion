#!/bin/bash
# Called by run_annotation.sh once passed.txt exists: one Bakta task per genome,
# in chunks of 1,000 (SLURM's array cap), at most MAX_PARALLEL at a time.
# Bakta needs no downloads, so it can run wider than the assembly step.
set -euo pipefail
source slurm/common.sh
MAX_PARALLEL=60

total=$(wc -l < "$MORAD_DIR/passed.txt")
n=${1:-$total}
chunks=$(( (n + 999) / 1000 ))
for (( c = 0; c < chunks; c++ )); do
    offset=$(( c * 1000 ))
    size=$(( n - offset < 1000 ? n - offset : 1000 ))
    job=$(sbatch --parsable --array=0-$((size - 1))%$(( MAX_PARALLEL / chunks )) \
          --export=ALL,OFFSET=$offset slurm/annotate.sbatch)
    echo "Annotation: job $job (genomes $((offset + 1))-$((offset + size)) of $total passing QC)"
done
