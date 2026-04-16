#!/bin/bash
#SBATCH --job-name=nyc
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dy23a@fsu.edu
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=6-23:00:00
#SBATCH --output=NONE
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# sbatch jobs/base_nyc.sh
# c; jobs/base_nyc.sh

date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/flow
PORJ=NYC_Mobi_15min
LOG=$BASE/output/$PORJ
mkdir -p $LOG
ARGS="--bs 1024 --dataset nyc_mobility --proj $PORJ --years 2024"
# ARGS="--bs 64 --dataset chicago_mobility --proj $PORJ --years 2025"

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
