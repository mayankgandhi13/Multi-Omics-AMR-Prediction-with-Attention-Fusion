#!/bin/bash
#SBATCH --job-name=amr-summary
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# Collect every fold's scores into results/summary.csv.
set -eo pipefail
source slurm/common.sh

run python -m src.summarize
