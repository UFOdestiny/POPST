#!/usr/bin/env bash
#SBATCH --job-name=od_calibration
#SBATCH --account=fsu-compsci-dept
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --time=3-00:00:00
#SBATCH --output=NONE
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Calibrate existing OD checkpoints after the four *_od_base.sh jobs finish.
# Examples:
#   sbatch jobs/od_calibration.sh
#   CALIBRATION_ENGINE=zero_cqr_8bin sbatch jobs/od_calibration.sh
# To restrict a run, edit DATASETS=(...), MODELS=(...), and
# CALIBRATION_ENGINES=(...) below.

set -uo pipefail
date
module load cuda conda
conda activate st

BASE=/home/dy23a.fsu/st
SRC=$BASE/src/od
PYTHON=${PYTHON:-python3}
YEARS=${YEARS:-2025_12to1}
BS=${BS:-64}
ALPHA=${ALPHA:-0.05}
CQR_MODE=${CQR_MODE:-horizon}
ZERO_CQR_BINS=${ZERO_CQR_BINS:-8}

# Each entry is run and logged separately for every compatible model/dataset.
#   od_cqr_horizon: generic OD split conformal, one radius per horizon
#   od_cqr_global:  generic OD split conformal, one shared radius
#   zero_cqr_Nbin:  sparse OD ZeroCQR, N positive-demand Mondrian bins
CALIBRATION_ENGINES=(
    od_cqr_horizon
    od_cqr_global
    zero_cqr_4bin
    zero_cqr_8bin
    zero_cqr_16bin
    zero_cqr
)
# Backward-compatible one-method override, e.g.
# CALIBRATION_ENGINE=zero_cqr_4bin sbatch jobs/od_calibration.sh
if [[ -n ${CALIBRATION_ENGINE:-} ]]; then
    CALIBRATION_ENGINES=("$CALIBRATION_ENGINE")
fi

DATASETS=(
    chicago_od_15min_taxi
    chicago_od_15min_tnp
    chicago_od_15min_bike
    nyc_manhattan_od_15min_fhv
    nyc_manhattan_od_15min_taxi
    nyc_manhattan_od_15min_bike
    dc_od_60min_taxi
    dc_od_60min_bike
    sf_od_15min_taxi
    sf_od_15min_bike
)

# These are the OD models that currently opt into the generic OD-CQR engine.
MODELS=(agcrn stgcn stzinb pdr pdr_reg pdr_reg_post)

project_for_dataset() {
    case "$1" in
        chicago_*) echo Chi_OD ;;
        nyc_*) echo NYC_OD ;;
        dc_*) echo DC_OD ;;
        sf_*) echo SF_OD ;;
        *) return 1 ;;
    esac
}

result_model_name() {
    case "$1" in
        agcrn) echo AGCRN_OD ;;
        stgcn) echo STGCN_OD ;;
        stzinb) echo STZINB ;;
        pdr) echo PDR ;;
        pdr_reg) echo PDR_REG ;;
        pdr_reg_post) echo PDR_REG_POST ;;
        *) return 1 ;;
    esac
}

latest_checkpoint() {
    local directory=$1
    find "$directory" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

method_args() {
    local method=$1
    case "$method" in
        od_cqr_horizon)
            METHOD_ARGS=(--cqr horizon --quantile_alpha "$ALPHA")
            METHOD_KIND=od_cqr
            ;;
        od_cqr_global)
            METHOD_ARGS=(--cqr global --quantile_alpha "$ALPHA")
            METHOD_KIND=od_cqr
            ;;
        zero_cqr_*bin)
            local bins=${method#zero_cqr_}
            bins=${bins%bin}
            [[ "$bins" =~ ^[1-9][0-9]*$ ]] || return 1
            METHOD_ARGS=(--zero_cqr_alpha "$ALPHA" --zero_cqr_active_bins "$bins" --zero_cqr_aux_epochs 0)
            METHOD_KIND=zero_cqr
            ;;
        zero_cqr)
            METHOD_ARGS=(--zero_cqr_alpha "$ALPHA" --zero_cqr_active_bins "$ZERO_CQR_BINS" --zero_cqr_aux_epochs 0)
            METHOD_KIND=zero_cqr
            ;;
        *) return 1 ;;
    esac
}

for dataset in "${DATASETS[@]}"; do
    project=$(project_for_dataset "$dataset") || { echo "Unknown dataset: $dataset"; continue; }
    for m in "${MODELS[@]}"; do
        model_name=$(result_model_name "$m") || { echo "Unknown model: $m"; continue; }
        checkpoint_dir="$BASE/result/$project/$model_name/$dataset"
        checkpoint=$(latest_checkpoint "$checkpoint_dir")
        if [[ -z "$checkpoint" ]]; then
            echo "Skipping $m on $dataset: no checkpoint under $checkpoint_dir"
            continue
        fi
        log_dir="$BASE/output/${project}_Calibration"
        mkdir -p "$log_dir"
        common=(--bs "$BS" --dataset "$dataset" --proj "$project" --years "$YEARS" --mode test --model_path "$checkpoint" --export)

        for calibration_method in "${CALIBRATION_ENGINES[@]}"; do
            if ! method_args "$calibration_method"; then
                echo "Skipping unknown calibration method: $calibration_method"
                continue
            fi
            if [[ "$METHOD_KIND" == "zero_cqr" && "$m" != "pdr_reg_post" ]]; then
                echo "Skipping $calibration_method for $m on $dataset: supported by pdr_reg_post only."
                continue
            fi
            echo "=== $calibration_method: $m on $dataset ==="
            "$PYTHON" "$SRC/$m/main.py" "${common[@]}" --calibration_tag "$calibration_method" "${METHOD_ARGS[@]}" 2>&1 | tee "$log_dir/${m}_${dataset}_${calibration_method}.log"
            echo "$calibration_method: $m on $dataset finished with exit code ${PIPESTATUS[0]}"
        done
    done
done
date
