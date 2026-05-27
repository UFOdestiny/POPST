#!/bin/bash
#SBATCH --job-name=stllm_chi
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

# sbatch jobs/stllm_chi.sh
# c; jobs/stllm_chi.sh

date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/flow
PORJ=Chi_Mobi
LOG=$BASE/output/$PORJ
mkdir -p $LOG
ARGS="--bs 512 --dataset chicago_mobility_dense --proj $PORJ --years 2025" # --engine_mode flow_matching

MODELS=(
    stllm stllm2 stllm3 stllm4 stllm5 ST-LLM-plus
)

for m in "${MODELS[@]}"; do
    echo "=== Running $m ==="
    python3 $SRC/$m/main.py $ARGS 2>&1 | tee $LOG/${m}.log
    echo "$m finished with exit code $?"
done

date
