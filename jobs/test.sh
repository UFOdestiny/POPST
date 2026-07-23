#!/usr/bin/env bash
#SBATCH --job-name=od_pdr_ablation
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

# sbatch /home/dy23a.fsu/st/jobs/test.sh

set -o pipefail
date
set +u
module load cuda conda
conda activate st
set -u

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/od
PYTHON=${PYTHON:-python3}
YEARS=${YEARS:-2025_12to1}
BS=${BS:-64}
RUN_ID=$(date +%Y-%m-%d_%H-%M-%S)

# Four PDR component ablations plus the three distributional-regression
# baselines.  Keep this list aligned with the calibration call below.
MODELS=(
    pdr_no_context
    pdr_no_zone_embed
    pdr_no_spatial
    pdr_no_moe
    pdr_reg_gau
    pdr_reg_lap
    pdr_reg_t
)

DATASETS=(
    chicago_od_15min_tnp
    nyc_manhattan_od_15min_fhv
    nyc_manhattan_od_15min_taxi
    dc_od_60min_bike
    chicago_od_15min_taxi
    chicago_od_15min_bike
)

project_for_dataset() {
    case "$1" in
        chicago_*) echo Chi_OD ;;
        nyc_*) echo NYC_OD ;;
        dc_*) echo DC_OD ;;
        *) return 1 ;;
    esac
}

for dataset in "${DATASETS[@]}"; do
    project=$(project_for_dataset "$dataset") || {
        echo "Unknown dataset: $dataset"
        continue
    }
    log_dir="$BASE/output/$project"
    mkdir -p "$log_dir"

    for model in "${MODELS[@]}"; do
        echo "=== Training $model on $dataset ($project) ==="
        run_log="$log_dir/${model}_${dataset}_${RUN_ID}.log"
        "$PYTHON" "$SRC/$model/main.py" \
            --bs "$BS" --proj "$project" --years "$YEARS" --export \
            --dataset "$dataset" 2>&1 | tee "$run_log"
        status=${PIPESTATUS[0]}
        echo "Training $model on $dataset finished with exit code $status"
    done
done

# Run calibration only after every training run has completed.  The calibration
# script finds the newest checkpoint for every model/dataset pair and skips a
# pair whose training failed or did not save a checkpoint.
echo "=== Starting OD calibration for completed PDR ablations ==="
DATASETS_OVERRIDE="${DATASETS[*]}" \
MODELS_OVERRIDE="${MODELS[*]}" \
CALIBRATION_ENGINES_OVERRIDE="od_split_cp_horizon od_split_cp_global od_aci_horizon od_aci_global" \
YEARS="$YEARS" \
BS="$BS" \
"$BASE/jobs/od_calibration.sh"
calibration_status=$?
echo "OD calibration finished with exit code $calibration_status"
date
