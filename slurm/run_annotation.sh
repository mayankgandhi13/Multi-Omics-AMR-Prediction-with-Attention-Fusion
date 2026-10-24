#!/bin/bash
# Phase 2, step 2: QC the assemblies, then annotate every genome that passed.
# Run it after run_assembly.sh has finished.
#
#   bash slurm/run_annotation.sh       # all genomes that pass QC
#   bash slurm/run_annotation.sh 3     # just the first 3 (a quick test)
#
# The QC job decides how many genomes pass, so a small "launcher" job reads
# passed.txt after QC and only then submits the annotation arrays.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/common.sh
mkdir -p logs/annotate

db=$(sbatch --parsable slurm/bakta_db.sbatch)
qc=$(sbatch --parsable slurm/qc.sbatch)
echo "Bakta DB: job $db   QC: job $qc"

launch=$(sbatch --parsable --dependency=afterok:$db:$qc --job-name=amr-annotate-launch \
    --partition=short --cpus-per-task=1 --mem=1G --time=00:10:00 --output=logs/%x_%j.out \
    --wrap="bash slurm/submit_annotation_arrays.sh ${1:-}")
echo "Annotation arrays get submitted by job $launch once QC and the DB are ready."
echo "Watch: squeue -u $USER"
