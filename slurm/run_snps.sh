#!/bin/bash
# Phase 2, step 4: call SNPs vs K-12 for every isolate that passed assembly QC.
#
#   bash slurm/run_snps.sh        # all
#   bash slurm/run_snps.sh 3      # just the first 3 (a quick test)
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/common.sh
mkdir -p logs/snps
MAX_PARALLEL=60   # reads are already on disk, so no download throttling here

images=$(sbatch --parsable slurm/bio_images.sbatch)   # pulls Snippy + the reference if missing
echo "Images + reference: job $images"
total=$(wc -l < "$MORAD_DIR/passed.txt")
n=${1:-$total}
chunks=$(( (n + 999) / 1000 ))
for (( c = 0; c < chunks; c++ )); do
    offset=$(( c * 1000 ))
    size=$(( n - offset < 1000 ? n - offset : 1000 ))
    job=$(sbatch --parsable --dependency=afterok:$images --array=0-$((size - 1))%$(( MAX_PARALLEL / chunks )) \
          --export=ALL,OFFSET=$offset slurm/snps.sbatch)
    echo "SNP calling: job $job (isolates $((offset + 1))-$((offset + size)))"
done
