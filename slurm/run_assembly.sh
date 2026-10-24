#!/bin/bash
# Phase 2, step 1: assemble every Moradigaravand isolate with one command.
#
#   bash slurm/run_assembly.sh        # all 1,936 isolates
#   bash slurm/run_assembly.sh 3      # just the first 3 (a quick test)
#
# Pulls the tool images, then fans out one array task per isolate. At most
# MAX_PARALLEL run at once so we share the cluster politely.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/common.sh
mkdir -p logs/assemble
MAX_PARALLEL=50

bash scripts/get_metadata.sh "$MORAD_DIR"
total=$(wc -l < "$MORAD_DIR/runs.txt")
n=${1:-$total}

images=$(sbatch --parsable slurm/bio_images.sbatch)
echo "Images: job $images"

# SLURM arrays hold at most 1,001 tasks, so split into chunks of 1,000.
chunks=$(( (n + 999) / 1000 ))
for (( c = 0; c < chunks; c++ )); do
    offset=$(( c * 1000 ))
    size=$(( n - offset < 1000 ? n - offset : 1000 ))
    job=$(sbatch --parsable --dependency=afterok:$images --array=0-$((size - 1))%$(( MAX_PARALLEL / chunks )) \
          --export=ALL,OFFSET=$offset slurm/assemble.sbatch)
    echo "Assembly: job $job (isolates $((offset + 1))-$((offset + size)))"
done
echo "Watch: squeue -u $USER"
