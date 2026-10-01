# Shared settings for every Explorer job. Sourced by the other scripts in slurm/.
# If your paths differ, edit these two lines (or export the variables before submitting).

# Conda env location. /home has a small fixed quota; if your PI has a /projects
# space, put the env there instead, e.g. /projects/<lab>/envs/amr
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/amr}"

# Data lives on /scratch: fast and roomy, but purged now and then. That's fine
# here, because scripts/get_data.sh re-downloads it and src.prepare rebuilds the rest.
export AMR_DATA_DIR="${AMR_DATA_DIR:-/scratch/$USER/amr-data}"

# Use exactly the cores SLURM gave us. Too few wastes the allocation; too many
# and threads elbow each other for the same cores, which is slower, not faster.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

ANTIBIOTICS=(CIP CTX CTZ GEN)
SPLITS=(random lineage)

# Array jobs have 8 tasks: one per antibiotic x split.
# Task 0 = CIP/random, 1 = CIP/lineage, 2 = CTX/random, ... 7 = GEN/lineage.
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    AB=${ANTIBIOTICS[$((SLURM_ARRAY_TASK_ID / 2))]}
    SPLIT=${SPLITS[$((SLURM_ARRAY_TASK_ID % 2))]}
fi

module load anaconda3/2024.06
