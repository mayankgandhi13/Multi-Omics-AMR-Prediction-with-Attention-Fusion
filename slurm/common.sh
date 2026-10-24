# Shared settings for every Explorer job. Sourced by the other scripts in slurm/.

# Every job runs inside one container image (see container/amr.def): Python,
# PyTorch + CUDA and scikit-learn packed into a single file on /scratch.
# setup_env.sh builds it; run_all.sh checks it at the start of every run, so a
# /scratch purge just means one rebuild.
export SIF="${SIF:-/scratch/$USER/amr/amr.sif}"

# Data lives on /scratch too: fast and roomy, but purged now and then. That's fine
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
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    AB=${ANTIBIOTICS[$((SLURM_ARRAY_TASK_ID / 2))]}
    SPLIT=${SPLITS[$((SLURM_ARRAY_TASK_ID % 2))]}
fi

# Run a command inside the container.
#   --nv              hand over the GPU, if SLURM gave this job one
#   --bind /scratch   let the container see the data
#   PYTHONNOUSERSITE  ignore anything pip-installed in ~/.local, so the
#                     container's packages are the only ones in play
run() {
    local gpu=""
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && gpu="--nv"
    apptainer exec $gpu --bind /scratch --env PYTHONNOUSERSITE=1 "$SIF" "$@"
}

# --- Phase 2 (pangenome) -----------------------------------------------------
# Moradigaravand et al. 2018: 1,936 E. coli with public reads + phenotypes.
export MORAD_DIR="$AMR_DATA_DIR/moradigaravand"
# One ready-made BioContainers image per tool (fastp, shovill, bakta, panaroo),
# pulled once by slurm/bio_images.sbatch. Like borrowing each specialist's own
# toolbox instead of cramming every tool into one drawer.
export IMAGES="/scratch/$USER/amr/images"

# Run a bioinformatics tool from its image:  tool shovill shovill --help
tool() {
    local image="$1"; shift
    apptainer exec --bind /scratch "$IMAGES/$image.sif" "$@"
}
