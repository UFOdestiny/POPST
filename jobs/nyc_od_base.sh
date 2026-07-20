#!/bin/bash
#SBATCH --job-name=nyc_o_b
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

# sbatch /home/dy23a.fsu/st/jobs/nyc_od_base.sh
# c; /home/dy23a.fsu/st/jobs/nyc_od_base.sh

date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/od
PORJ=NYC_OD
LOG=$BASE/output/$PORJ
mkdir -p $LOG
DATASETS=(
    nyc_manhattan_od_15min_fhv
    nyc_manhattan_od_15min_taxi
    nyc_manhattan_od_15min_bike
)
BASE_ARGS="--bs 64 --proj $PORJ --years 2025_12to1 --export"

# MODELS=(
#     stgcn
# )
MODELS=(
    pdr_reg_post pdr_reg pdr pdr_no_context pdr_no_zone_embed pdr_no_spatial pdr_no_moe
    stzinb agcrn astgcn gmel gwnet ha hl hmdlf lstm stgcn stgode
)

for dataset in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        echo "=== Running $m on $dataset ==="
        python3 $SRC/$m/main.py $BASE_ARGS --dataset "$dataset" 2>&1 | tee "$LOG/${m}_${dataset}.log"
        echo "$m on $dataset finished with exit code ${PIPESTATUS[0]}"
    done
done



date
