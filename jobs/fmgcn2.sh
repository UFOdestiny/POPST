#!/bin/bash
#SBATCH --job-name=fmgcn2
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

# sbatch jobs/fmgcn2.sh
# c; jobs/fmgcn2.sh

date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/flow/fmgcn2
# PORJ=Chi_Mobi_15min
PORJ=NYC_Mobi_15min
LOG=$BASE/output/$PORJ
mkdir -p $LOG

# ARGS="--bs 1024 --dataset chicago_mobility --proj $PORJ --years 2025"
ARGS="--bs 512 --dataset nyc_mobility --proj $PORJ --years 2024"
ARGS="$ARGS --adj_type doubletransition"
ARGS="$ARGS --hidden_dim 32 --skip_dim 192 --end_dim 256 --graph_dim 16 --num_layers 6 --diffusion_order 2"
ARGS="$ARGS --highway_window 6 --dropout 0.2 --lrate 0.001 --wdecay 1e-4 --clip_grad_norm 5 --step_size 120 --gamma 0.5"

echo "=== Running fmgcn2 ==="
python3 $SRC/main.py $ARGS 2>&1 | tee $LOG/fmgcn2.log
echo "fmgcn2 finished with exit code $?"

date
