#!/bin/bash
#SBATCH --job-name=dc_f_stllm
#SBATCH --account=fsu-compsci-dept
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

# sbatch /home/dy23a.fsu/st/jobs/dc_flow_stllm.sh
# c; /home/dy23a.fsu/st/jobs/dc_flow_stllm.sh

date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/flow
PORJ=DC_Flow
LOG=$BASE/output/$PORJ
mkdir -p $LOG

BASE_ARGS="--bs 512 --dataset dc_60min"

YEARS=(
    "2025_12to1"
    "2025_12to3"
    "2025_12to6"
    "2025_12to9"
    "2025_12to12"
)

# MODELS=(
#     stllm6 stllm7
# )

MODELS=(
    # stllm stllm2 stllm3 stllm4 stllm5 
    stllm6 stllm7 stllm8 stllm9
)

for m in "${MODELS[@]}"; do
    for y in "${YEARS[@]}"; do
        echo "=== Running $m for year $y ==="
        CURRENT_ARGS="$BASE_ARGS --years $y --proj $y"
        python3 $SRC/$m/main.py $CURRENT_ARGS 2>&1 | tee $LOG/${m}_${y}.log
        echo "$m for year $y finished with exit code $?"
    done
done

date
