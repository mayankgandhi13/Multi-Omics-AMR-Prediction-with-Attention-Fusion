#!/bin/bash
#SBATCH --job-name=amr-images
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Pull one BioContainers image per Phase 2 tool (skips any already present).
set -eo pipefail
source slurm/common.sh

declare -A IMAGE_TAGS=(
    [fastp]="quay.io/biocontainers/fastp:1.4.0--h43da1c4_0"
    [shovill]="quay.io/biocontainers/shovill:1.4.2--hdfd78af_1"
    [bakta]="quay.io/biocontainers/bakta:1.12.2--pyhdfd78af_0"
    [panaroo]="quay.io/biocontainers/panaroo:1.8.0--pyhdfd78af_1"
)
mkdir -p "$IMAGES"
# Unpack on the node's local disk (fast); only the finished .sif lands on /scratch.
export APPTAINER_TMPDIR="/tmp/$USER-apptainer-tmp" APPTAINER_CACHEDIR="/tmp/$USER-apptainer-cache"
trap 'rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"' EXIT
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

for name in "${!IMAGE_TAGS[@]}"; do
    if [ -s "$IMAGES/$name.sif" ]; then
        echo "$name: already there"
    else
        apptainer pull --force "$IMAGES/$name.sif.partial" "docker://${IMAGE_TAGS[$name]}"
        mv "$IMAGES/$name.sif.partial" "$IMAGES/$name.sif"
        echo "$name: pulled"
    fi
done
tool fastp fastp --version
tool shovill shovill --version
