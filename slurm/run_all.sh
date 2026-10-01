#!/bin/bash
# Submit the whole Phase 1 pipeline with one command:
#
#   bash slurm/run_all.sh
#
# The jobs form a relay race: prepare runs first, baselines and CNNs start
# together once it finishes, and the summary runs after both are done.
# `--dependency=afterok` is the baton: a job waits until the previous one succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."   # SLURM jobs run from the directory they were submitted from
mkdir -p logs

prep=$(sbatch --parsable slurm/prepare.sbatch)
base=$(sbatch --parsable --dependency=afterok:$prep slurm/baselines.sbatch)
cnn=$(sbatch --parsable --dependency=afterok:$prep slurm/cnn.sbatch)
summ=$(sbatch --parsable --dependency=afterok:$base:$cnn slurm/summarize.sbatch)

echo "Submitted: prepare=$prep  baselines=$base  cnn=$cnn  summary=$summ"
echo "Watch:     squeue -u $USER"
echo "Logs:      tail -f logs/<job>.out"
