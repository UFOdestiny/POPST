#!/bin/bash
#SBATCH --job-name=chi_f_b
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

# sbatch /home/dy23a.fsu/st/jobs/chi_flow_base.sh
# c; /home/dy23a.fsu/st/jobs/chi_flow_base.sh

date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/flow
PORJ=Chi_Flow
LOG=$BASE/output/$PORJ
mkdir -p $LOG
ARGS="--bs 512 --dataset chicago_15min --proj $PORJ --years 2025"

MODELS=(
    healthmamba uqgnn
)
# MODELS=(
#     agcrn astgcn d2stgnn dgcrn dstagnn gluonts gwnet
#     hl lstm transformer patchtst stgcn stgode ST-LLM-plus sttn 
#     mamba uqgnn trustenergy healthmamba energymamba dcrnn
# )

for m in "${MODELS[@]}"; do
    echo "=== Running $m ==="
    python3 $SRC/$m/main.py $ARGS 2>&1 | tee $LOG/${m}.log
    echo "$m finished with exit code $?"
done

date
