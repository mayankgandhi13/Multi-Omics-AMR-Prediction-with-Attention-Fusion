#!/bin/bash
#SBATCH --job-name=amr-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Check (and if needed build) the conda env before anything else runs.
set -eo pipefail
bash slurm/setup_env.sh
