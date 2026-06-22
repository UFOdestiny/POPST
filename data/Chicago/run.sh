#!/bin/bash
#SBATCH --job-name=Donwload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=48:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=0
#SBATCH --output=/home/dy23a.fsu/st/data/Chicago/download.log
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept

# sbatch /home/dy23a.fsu/st/data/Chicago/run.sh

date

module load cuda python conda
conda activate st

cd /home/dy23a.fsu/st/data/Chicago

python3 -m nbconvert \
--to notebook \
--execute Chicago_Download.ipynb \
--output Chicago_Download_res.ipynb

date
