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
LOG=$BASE/output
ARGS="--bs 64 --dataset nyc_mobility --proj Test --years 2024 --max_epochs 2"

MODELS=(
    stgcn
)
# MODELS=(
#     agcrn astgcn d2stgnn dcrnn dgcrn dstagnn gluonts gwnet
#     hl lstm mamba #mamba2 mamba3 mamba4 mamba5 mamba6 mamba7
#     patchtst stgcn stgode stllm stllm2 sttn transformer uqgnn
# )

for m in "${MODELS[@]}"; do
    echo "=== Running $m ==="
    python3 $SRC/$m/main.py $ARGS 2>&1 | tee $LOG/Test_${m}.log
    echo "$m finished with exit code $?"
done

date
