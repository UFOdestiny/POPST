#!/bin/bash
#SBATCH --job-name=dc_o_b
#SBATCH --account=fsu-compsci-dept
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dy23a@fsu.edu
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --time=6-23:00:00
#SBATCH --output=NONE
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# sbatch /home/dy23a.fsu/st/jobs/dc_od_base.sh
# c; /home/dy23a.fsu/st/jobs/dc_od_base.sh

date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/od
PORJ=DC_OD
LOG=$BASE/output/$PORJ
mkdir -p $LOG
ARGS="--bs 8 --dataset dc_od_60min --proj $PORJ --years 2025"

# MODELS=(
#     stgcn
# )
MODELS=(
    agcrn astgcn gmel gwnet ha hl hmdlf lstm odmixer stgcn stgode sttn stzinb
)

for m in "${MODELS[@]}"; do
    echo "=== Running $m ==="
    python3 $SRC/$m/main.py $ARGS 2>&1 | tee $LOG/${m}.log
    echo "$m finished with exit code $?"
done

date
