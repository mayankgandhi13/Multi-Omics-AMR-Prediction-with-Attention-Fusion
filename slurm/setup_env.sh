#!/bin/bash
# Make sure the conda env exists and works; build it if not.
# run_all.sh submits this for you (slurm/setup.sbatch) at the start of every run:
# a few seconds when the env is healthy, ~10 minutes when it has to build.
#
# To run it by hand instead, from the repo root on a compute node:
#   srun --partition=short --cpus-per-task=4 --mem=16G --time=01:00:00 bash slurm/setup_env.sh
set -eo pipefail
source slurm/common.sh

# conda and pip keep a cache of every download (several GB for PyTorch).
# Park conda's cache on /scratch next to the env, and skip pip's.
export CONDA_PKGS_DIRS="/scratch/$USER/conda-pkgs"
export PIP_NO_CACHE_DIR=1

if "$ENV_PREFIX/bin/python" -c "import torch, sklearn, yaml" 2>/dev/null; then
    echo "Env at $ENV_PREFIX is healthy, nothing to do."
else
    echo "Building env at $ENV_PREFIX ..."
    rm -rf "$ENV_PREFIX"   # a half-built env from a crashed install is worse than none
    mkdir -p "$(dirname "$ENV_PREFIX")"
    conda env create --prefix "$ENV_PREFIX" --file environment.yml
fi
mkdir -p logs "$AMR_DATA_DIR"

source activate "$ENV_PREFIX"
python -c "import torch, sklearn; print('torch', torch.__version__, '| sklearn', sklearn.__version__)"
