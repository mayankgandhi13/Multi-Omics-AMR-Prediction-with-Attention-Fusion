#!/bin/bash
#SBATCH --job-name=amr-cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
# CNN baselines on a GPU, one array task per antibiotic x split.
# Want a specific card? e.g. --gres=gpu:a100:1 (list types with `sinfo -p gpu -o "%G"`),
# but a named card can mean a longer wait in the queue than "any GPU".
set -eo pipefail
source slurm/common.sh
source activate "$ENV_PREFIX"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Task $SLURM_ARRAY_TASK_ID: $AB / $SPLIT"
for INPUT in fcgr label; do
    python -m src.train_cnn --antibiotic "$AB" --split "$SPLIT" --input "$INPUT"
done
