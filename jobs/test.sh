#!/bin/bash
#SBATCH --job-name=Test
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dy23a@fsu.edu
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=96:00:00
#SBATCH --output=NONE
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# sbatch jobs/test.sh
# c; jobs/test.sh

date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/flow
PORJ=Test
LOG=$BASE/output/Test
mkdir -p $LOG
ARGS="--bs 1024 --dataset nyc_mobility --proj $PORJ --years 2024 --max_epochs 1"
# ARGS="--bs 1024 --dataset chicago_mobility --proj $PORJ --years 2025 --max_epochs 1"

# MODELS=(
#     stgcn
# )
MODELS=(
    agcrn astgcn d2stgnn dgcrn dstagnn gluonts gwnet
    hl lstm #mamba2 mamba3 mamba4 mamba5 mamba6 mamba7
    patchtst stgcn stgode stllm stllm2 sttn transformer uqgnn
    dcrnn mamba
)

for m in "${MODELS[@]}"; do
    echo "=== Running $m ==="
    python3 $SRC/$m/main.py $ARGS 2>&1 | tee $LOG/${m}.log
    echo "$m finished with exit code $?"
done

date
