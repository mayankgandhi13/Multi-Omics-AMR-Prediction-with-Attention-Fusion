#!/bin/bash
# Submit the whole Phase 1 pipeline with one command:
#
#   bash slurm/run_all.sh
#
# The jobs form a relay race: setup checks the conda env, prepare builds the
# dataset, baselines and CNNs start together, and the summary runs after both.
# `--dependency=afterok` is the baton: a job waits until the previous one succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."   # SLURM jobs run from the directory they were submitted from
mkdir -p logs

setup=$(sbatch --parsable slurm/setup.sbatch)
prep=$(sbatch --parsable --dependency=afterok:$setup slurm/prepare.sbatch)
base=$(sbatch --parsable --dependency=afterok:$prep slurm/baselines.sbatch)
cnn=$(sbatch --parsable --dependency=afterok:$prep slurm/cnn.sbatch)
summ=$(sbatch --parsable --dependency=afterok:$base:$cnn slurm/summarize.sbatch)

echo "Submitted: setup=$setup  prepare=$prep  baselines=$base  cnn=$cnn  summary=$summ"
echo "Watch:     squeue -u $USER"
echo "Logs:      tail -f logs/<job>.out"
