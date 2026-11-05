#!/bin/bash
#SBATCH --job-name=amr-baselines
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
# LR / SVM / RF baselines. A job array: eight copies run side by side, one per
# antibiotic x split, like eight cooks each making one dish instead of one cook making all eight.
set -eo pipefail
source slurm/common.sh

echo "Task $SLURM_ARRAY_TASK_ID: ${DATASET:-giessen} $AB / $SPLIT on $SLURM_CPUS_PER_TASK CPUs"
# Phase 2 gene table: sbatch --export=ALL,DATASET=pangenome slurm/baselines.sbatch
# Shortlist features first: sbatch --export=ALL,DATASET=snps,TOP=1000 slurm/baselines.sbatch
run python -m src.baselines --dataset "${DATASET:-giessen}" --antibiotic "$AB" --split "$SPLIT" ${TOP:+--top "$TOP"}
