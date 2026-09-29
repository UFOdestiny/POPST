#!/usr/bin/env bash
#SBATCH --account=gtyson.fsu
#SBATCH --partition=hpg-b200
#SBATCH --job-name=st
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-23:00:00
#SBATCH --gpus=3
#SBATCH --chdir=/blue/fsu-compsci-dept/dahai/proj/POPST
#SBATCH --output=results/%x-%j.out

set -euo pipefail

python_bin=/blue/fsu-compsci-dept/dahai/conda/envs/llm/bin/python

cd /blue/fsu-compsci-dept/dahai/proj/POPST

"$python_bin" -u run.py config/suites/od_baselines.yaml "$@"
exec "$python_bin" -u run.py config/suites/od_zeropdr.yaml "$@"
