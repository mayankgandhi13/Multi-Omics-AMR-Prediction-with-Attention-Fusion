#!/bin/bash
#SBATCH --job-name=amr-qc
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Measure every assembly and write passed.txt (see src/assembly_qc.py).
set -eo pipefail
source slurm/common.sh
run python -m src.assembly_qc
