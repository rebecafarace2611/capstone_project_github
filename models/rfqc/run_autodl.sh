#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
THREADS="${RFQC_THREADS:-20}"
RUNS_DIR="${RFQC_RUNS_DIR:-$ROOT/runs}"
LOGS_DIR="${RFQC_LOGS_DIR:-$ROOT/logs}"

TRAIN="${RFQC_TRAIN:-$ROOT/data/train_model_dataset.parquet}"
TEST="${RFQC_TEST:-$ROOT/data/test_model_dataset.parquet}"
FEATURES="${RFQC_FEATURES:-$ROOT/outputs/leakage_analysis/approved_features.json}"
FOLDS="${RFQC_FOLDS:-$ROOT/outputs/rfqc/folds/fold_assignments.parquet}"
RFQC_SCRIPT="$SCRIPT_DIR/run.R"
FINAL_PREP_SCRIPT="$SCRIPT_DIR/prepare_final_run.py"
FINAL_SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_results.py"
WORKFLOW="$SCRIPT_DIR/workflow.R"

mkdir -p "$RUNS_DIR" "$LOGS_DIR"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

for file in "$TRAIN" "$FEATURES" "$FOLDS" "$RFQC_SCRIPT" "$WORKFLOW"; do
  require_file "$file"
done

export RF_CORES="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MC_CORES=1

common_args=(
  --train "$TRAIN"
  --test "$TEST"
  --approved-features "$FEATURES"
  --folds "$FOLDS"
  --threads "$THREADS"
)

run_logged() {
  local name="$1"
  shift
  local log="$LOGS_DIR/${name}.log"
  echo "Starting $name with $THREADS RF cores"
  echo "Log: $log"
  Rscript "$RFQC_SCRIPT" "$@" "${common_args[@]}" 2>&1 | tee -a "$log"
}

absolute_run_dir() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT/$path"
  fi
}

command="${1:-help}"
case "$command" in
  install)
    Rscript "$SCRIPT_DIR/install_packages.R"
    ;;
  smoke)
    run_logged smoke smoke \
      --output-dir "$RUNS_DIR/smoke"
    ;;
  baseline-500)
    run_logged baseline_500 baseline \
      --baseline-trees 500 \
      --output-dir "$RUNS_DIR/baseline_500" \
      --no-save-model
    ;;
  baseline-1000)
    run_logged baseline_1000 baseline \
      --baseline-trees 1000 \
      --output-dir "$RUNS_DIR/baseline_1000" \
      --no-save-model
    ;;
  quick)
    run_logged tune_quick_500 tune \
      --profile quick \
      --output-dir "$RUNS_DIR/tune_quick_500"
    ;;
  refine-local)
    run_logged tune_local_gini_500 tune \
      --profile local \
      --output-dir "$RUNS_DIR/tune_local_gini_500"
    ;;
  refine-gini)
    run_logged tune_gini_1000 tune \
      --profile paper \
      --tuning-trees 1000 \
      --output-dir "$RUNS_DIR/tune_gini_1000"
    ;;
  refine-auc)
    run_logged tune_auc_1000 tune \
      --profile auc \
      --tuning-trees 1000 \
      --output-dir "$RUNS_DIR/tune_auc_1000"
    ;;
  full-1000)
    run_logged tune_full_1000 tune \
      --profile full \
      --tuning-trees 1000 \
      --output-dir "$RUNS_DIR/tune_full_1000"
    ;;
  prepare-final-qstar)
    source_dir="$(absolute_run_dir "${2:-runs/tune_local_gini_500}")"
    result_dir="$(absolute_run_dir "${3:-runs/final_local_qstar_3000}")"
    baseline_summary="$(absolute_run_dir "${4:-baseline_summary.csv}")"
    require_file "$FINAL_PREP_SCRIPT"
    require_file "$source_dir/cv_ranking.csv"
    require_file "$source_dir/run_context.json"
    require_file "$baseline_summary"
    python3 "$FINAL_PREP_SCRIPT" \
      --cv-ranking "$source_dir/cv_ranking.csv" \
      --run-context "$source_dir/run_context.json" \
      --baseline-summary "$baseline_summary" \
      --output-dir "$result_dir" \
      --final-trees 3000
    ;;
  final-qstar)
    source_dir="$(absolute_run_dir "${2:-runs/tune_local_gini_500}")"
    result_dir="$(absolute_run_dir "${3:-runs/final_local_qstar_3000}")"
    baseline_summary="$(absolute_run_dir "${4:-baseline_summary.csv}")"
    require_file "$TEST"
    require_file "$result_dir/best_configuration.json"
    require_file "$source_dir/cv_ranking.csv"
    require_file "$baseline_summary"
    require_file "$FINAL_SUMMARY_SCRIPT"
    run_logged "final_local_qstar_3000" final \
      --final-trees 3000 \
      --output-dir "$result_dir"
    python3 "$FINAL_SUMMARY_SCRIPT" \
      --baseline-summary "$baseline_summary" \
      --cv-ranking "$source_dir/cv_ranking.csv" \
      --final-metrics "$result_dir/final_test_metrics.json" \
      --output "$result_dir/rfqc_results_summary.csv"
    ;;
  retry-final-qstar)
    source_dir="$(absolute_run_dir "${2:-runs/tune_local_gini_500}")"
    result_dir="$(absolute_run_dir "${3:-runs/final_local_qstar_3000}")"
    baseline_summary="$(absolute_run_dir "${4:-baseline_summary.csv}")"
    if [[ -f "$result_dir/.test_access_started.json" ]]; then
      echo "Test access already started; refusing to retry final evaluation." >&2
      exit 1
    fi
    require_file "$TEST"
    require_file "$result_dir/best_configuration.json"
    require_file "$source_dir/cv_ranking.csv"
    require_file "$baseline_summary"
    require_file "$FINAL_SUMMARY_SCRIPT"
    rm -f \
      "$result_dir/final_test_metrics.json" \
      "$result_dir/final_test_predictions.parquet" \
      "$result_dir/final_threshold_curve.csv" \
      "$result_dir/final_tree_convergence.csv" \
      "$result_dir/rfqc_native_model.rds" \
      "$result_dir/rfqc_native_model.rds.tmp" \
      "$result_dir/.final_evaluation_complete.json"
    run_logged "final_local_qstar_3000_retry" final \
      --final-trees 3000 \
      --output-dir "$result_dir"
    python3 "$FINAL_SUMMARY_SCRIPT" \
      --baseline-summary "$baseline_summary" \
      --cv-ranking "$source_dir/cv_ranking.csv" \
      --final-metrics "$result_dir/final_test_metrics.json" \
      --output "$result_dir/rfqc_results_summary.csv"
    ;;
  recover-final-qstar)
    source_dir="$(absolute_run_dir "${2:-runs/tune_local_gini_500}")"
    result_dir="$(absolute_run_dir "${3:-runs/final_local_qstar_3000}")"
    baseline_summary="$(absolute_run_dir "${4:-baseline_summary.csv}")"
    if [[ -f "$result_dir/.test_access_started.json" ]]; then
      echo "Test access already started; refusing saved-model recovery." >&2
      exit 1
    fi
    require_file "$TEST"
    require_file "$result_dir/best_configuration.json"
    require_file "$result_dir/final_threshold_curve.csv"
    require_file "$result_dir/final_tree_convergence.csv"
    require_file "$source_dir/cv_ranking.csv"
    require_file "$baseline_summary"
    require_file "$FINAL_SUMMARY_SCRIPT"
    model_count=0
    [[ -f "$result_dir/rfqc_native_model.rds" ]] && model_count=$((model_count + 1))
    [[ -f "$result_dir/rfqc_native_model.rds.tmp" ]] && model_count=$((model_count + 1))
    if [[ "$model_count" -ne 1 ]]; then
      echo "Recovery needs exactly one .rds or .rds.tmp model file." >&2
      exit 1
    fi
    run_logged "final_local_qstar_3000_recover" recover \
      --final-trees 3000 \
      --output-dir "$result_dir"
    python3 "$FINAL_SUMMARY_SCRIPT" \
      --baseline-summary "$baseline_summary" \
      --cv-ranking "$source_dir/cv_ranking.csv" \
      --final-metrics "$result_dir/final_test_metrics.json" \
      --output "$result_dir/rfqc_results_summary.csv"
    ;;
  resume-final-qstar)
    source_dir="$(absolute_run_dir "${2:-runs/tune_local_gini_500}")"
    result_dir="$(absolute_run_dir "${3:-runs/final_local_qstar_3000}")"
    baseline_summary="$(absolute_run_dir "${4:-baseline_summary.csv}")"
    require_file "$TEST"
    require_file "$result_dir/.test_access_started.json"
    require_file "$result_dir/best_configuration.json"
    require_file "$result_dir/rfqc_native_model.rds"
    require_file "$result_dir/final_threshold_curve.csv"
    require_file "$result_dir/final_tree_convergence.csv"
    require_file "$source_dir/cv_ranking.csv"
    require_file "$baseline_summary"
    require_file "$FINAL_SUMMARY_SCRIPT"
    if [[ -f "$result_dir/final_test_metrics.json" ]] ||
       [[ -f "$result_dir/final_test_predictions.parquet" ]] ||
       [[ -f "$result_dir/.final_evaluation_complete.json" ]] ||
       [[ -f "$result_dir/.test_access_retry_started.json" ]]; then
      echo "Final outputs or a retry marker already exist; refusing resume." >&2
      exit 1
    fi
    run_logged "final_local_qstar_3000_resume" resume \
      --final-trees 3000 \
      --output-dir "$result_dir"
    python3 "$FINAL_SUMMARY_SCRIPT" \
      --baseline-summary "$baseline_summary" \
      --cv-ranking "$source_dir/cv_ranking.csv" \
      --final-metrics "$result_dir/final_test_metrics.json" \
      --output "$result_dir/rfqc_results_summary.csv"
    ;;
  help|*)
    cat <<'EOF'
Usage: bash models/rfqc/run_autodl.sh COMMAND

Commands:
  install          Install project-local R packages
  smoke            Run the 100-tree workflow check
  baseline-500     Run the 500-tree OOB baseline
  baseline-1000    Run the 1000-tree OOB baseline
  quick            Run 8 candidates x 5 folds at 500 trees
  refine-local     Run 4 local Gini candidates x 5 folds at 500 trees
  refine-gini      Run the 32-candidate Gini grid at 1000 trees
  refine-auc       Run the 32-candidate AUC grid at 1000 trees
  full-1000        Run both rules: 64 candidates at 1000 trees
  prepare-final-qstar [SOURCE_DIR] [FINAL_DIR] [BASELINE_SUMMARY]
                   Lock candidate 3 with q_star_prevalence at 3000 trees
  final-qstar [SOURCE_DIR] [FINAL_DIR] [BASELINE_SUMMARY]
                   Run the locked one-time final test and write the summary
  retry-final-qstar [SOURCE_DIR] [FINAL_DIR] [BASELINE_SUMMARY]
                   Retry only after a pre-test failure; refuses after test access
  recover-final-qstar [SOURCE_DIR] [FINAL_DIR] [BASELINE_SUMMARY]
                   Continue from one saved .rds/.tmp model without retraining
  resume-final-qstar [SOURCE_DIR] [FINAL_DIR] [BASELINE_SUMMARY]
                   Resume after the documented pre-prediction test-read failure

Set RFQC_THREADS to change the default of 20 cores.
EOF
    ;;
esac
