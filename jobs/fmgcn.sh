#!/bin/bash
#SBATCH --job-name=fmgcn
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

# sbatch jobs/fmgcn.sh
# c; jobs/fmgcn.sh


date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/flow
PORJ=Chi_Mobi_15min_FM
LOG=$BASE/output/$PORJ
mkdir -p $LOG
ARGS="--bs 1024 --dataset chicago_mobility --proj $PORJ --years 2025 --engine_mode flow_matching"


MODELS=(
    fmgcn
)

for m in "${MODELS[@]}"; do
    echo "=== Running $m ==="
    python3 $SRC/$m/main.py $ARGS 2>&1 | tee $LOG/${m}.log
    echo "$m finished with exit code $?"
done

date
