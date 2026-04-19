#!/bin/bash
#SBATCH --job-name=o2vae_mnist
#SBATCH --partition=gpu-l40s
#SBATCH --account=kzlinlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kzlin@uw.edu
#SBATCH --output=/gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin/slurm_%j.out
#SBATCH --error=/gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin/slurm_%j.err

# ── Paths ──
DATA_DIR="/gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin"
OUT_DIR="/gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin"
REPO_DIR="/gscratch/kzlinlab/projects/microglia3d/git/o2vae"
SIF="/gscratch/kzlinlab/containers/microglia3d.sif"

# ── Setup ──
mkdir -p "${OUT_DIR}"
module load singularity
cd "${REPO_DIR}"

# Step 1: generate the O2-MNIST dataset (skip if already done)
if [ ! -f "${DATA_DIR}/X_train.sav" ]; then
    echo "Generating O2-MNIST dataset..."
    time -p singularity run --nv --bind /gscratch "${SIF}" \
        python data/generate_o2mnist.py "${DATA_DIR}"
else
    echo "Dataset already exists, skipping generation."
fi

# Step 2: train the model
echo "Starting training..."
time -p singularity run --nv --bind /gscratch "${SIF}" \
    python examples/train_o2mnist_hyak.py \
        --data-dir "${DATA_DIR}" \
        --out-dir  "${OUT_DIR}" \
        --epochs 41

touch "${OUT_DIR}/job_finished.flag"
echo "Done."
