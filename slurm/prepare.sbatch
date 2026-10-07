#!/bin/bash
#SBATCH --job-name=amr-prepare
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Download the data and build data/processed/dataset.npz (SNPs, labels, families, FCGR images).
set -eo pipefail
source slurm/common.sh

bash scripts/get_data.sh "$AMR_DATA_DIR"
run python -m src.prepare
