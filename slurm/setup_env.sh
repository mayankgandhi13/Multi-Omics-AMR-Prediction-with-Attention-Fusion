#!/bin/bash
# One-time setup on Explorer. Run it from the repo root on a compute node,
# because conda installs crawl on the shared login nodes:
#
#   srun --partition=short --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
#   bash slurm/setup_env.sh
set -eo pipefail
source slurm/common.sh

# conda and pip keep a cache of every download (several GB for PyTorch).
# Park conda's cache on /scratch and skip pip's, so /home's small quota stays free.
export CONDA_PKGS_DIRS="/scratch/$USER/conda-pkgs"
export PIP_NO_CACHE_DIR=1

if [ -d "$ENV_PREFIX" ]; then
    echo "Env already exists at $ENV_PREFIX (delete it to rebuild)."
else
    conda env create --prefix "$ENV_PREFIX" --file environment.yml
fi
mkdir -p logs "$AMR_DATA_DIR"

source activate "$ENV_PREFIX"
python -c "import torch, sklearn; print('torch', torch.__version__, '| sklearn', sklearn.__version__)"
echo "Ready. Next: bash slurm/run_all.sh"
