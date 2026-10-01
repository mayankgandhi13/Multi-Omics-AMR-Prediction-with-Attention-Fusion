#!/bin/bash
#SBATCH --job-name=amr-baselines
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
# LR / SVM / RF baselines. A job array: eight copies run side by side, one per
# antibiotic x split, like eight cooks each making one dish instead of one cook making all eight.
set -eo pipefail
source slurm/common.sh
source activate "$ENV_PREFIX"

echo "Task $SLURM_ARRAY_TASK_ID: $AB / $SPLIT on $SLURM_CPUS_PER_TASK CPUs"
python -m src.baselines --antibiotic "$AB" --split "$SPLIT"
