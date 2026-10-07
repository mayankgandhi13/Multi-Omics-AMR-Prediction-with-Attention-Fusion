#!/bin/bash
# Make sure the container image exists and works; build it if not.
# run_all.sh submits this for you (slurm/setup.sbatch) at the start of every run:
# a few seconds when the image is healthy, ~15 minutes when it has to build.
#
# To run it by hand instead, from the repo root:
#   srun --partition=short --cpus-per-task=8 --mem=16G --time=01:00:00 bash slurm/setup_env.sh
set -eo pipefail
source slurm/common.sh

if [ -f "$SIF" ] && run python -c "import torch, sklearn, yaml" 2>/dev/null; then
    echo "Image $SIF is healthy, nothing to do."
else
    echo "Building $SIF from container/amr.def ..."
    # Do the messy unpacking on the node's own local disk (fast) and write only
    # the finished single-file image to /scratch.
    export APPTAINER_TMPDIR="/tmp/$USER-apptainer-tmp" APPTAINER_CACHEDIR="/tmp/$USER-apptainer-cache"
    trap 'rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"' EXIT
    mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$SIF")"
    apptainer build --force "$SIF.partial" container/amr.def
    mv "$SIF.partial" "$SIF"   # only a complete image ever gets the real name
fi
mkdir -p logs "$AMR_DATA_DIR"

run python -c "import torch, sklearn; print('torch', torch.__version__, '| sklearn', sklearn.__version__)"
