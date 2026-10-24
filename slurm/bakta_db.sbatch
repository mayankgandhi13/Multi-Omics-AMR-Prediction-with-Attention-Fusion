#!/bin/bash
#SBATCH --job-name=amr-bakta-db
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Download Bakta's full annotation database once (skipped if already there).
# This is the "catalogue" Bakta checks every gene against: proteins from UniProt
# and friends, plus NCBI's AMRFinderPlus database of resistance genes.
set -eo pipefail
source slurm/common.sh

if [ -f "$BAKTA_DB/version.json" ]; then
    echo "Bakta DB already at $BAKTA_DB"
else
    parent="$(dirname "$BAKTA_DB")"
    rm -rf "$parent" && mkdir -p "$parent"
    tool bakta bakta_db download --output "$parent" --type full
fi
# AMRFinderPlus needs its own index built inside the database folder.
tool bakta amrfinder_update --force_update --database "$BAKTA_DB/amrfinderplus-db/"
cat "$BAKTA_DB/version.json"
