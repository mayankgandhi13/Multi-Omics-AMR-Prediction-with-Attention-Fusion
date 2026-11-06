#!/bin/bash
#SBATCH --job-name=amr-fusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
# Phase 4 attention fusion (gene table + SNPs) on a GPU, one task per antibiotic x split.
set -eo pipefail
source slurm/common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Task $SLURM_ARRAY_TASK_ID: $AB / $SPLIT"
run python -m src.fusion --antibiotic "$AB" --split "$SPLIT"
