#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

RUN_DIR="${LGBM_RUN_DIR:-runs/lightgbm/baseline}"
SCREEN_DIR="${LGBM_SCREEN_DIR:-runs/lightgbm/imbalance_screen}"
TUNE_DIR="${LGBM_TUNE_DIR:-runs/lightgbm/optuna_rus}"
REFINE_DIR="${LGBM_REFINE_DIR:-runs/lightgbm/optuna_refine}"
CONFIRM_DIR="${LGBM_CONFIRM_DIR:-runs/lightgbm/seed_confirmation}"
LOCK_OOF_DIR="${LGBM_LOCK_OOF_DIR:-runs/lightgbm/fixed_oof_lock}"
FINAL_TEST_DIR="${LGBM_FINAL_TEST_DIR:-runs/lightgbm/final_test}"
FINAL_SHAP_DIR="${LGBM_FINAL_SHAP_DIR:-runs/lightgbm/final_test/shap_top_score}"
THREADS="${LGBM_THREADS:-16}"
N_TRIALS="${LGBM_N_TRIALS:-60}"
REFINE_TRIALS="${LGBM_REFINE_TRIALS:-40}"

case "$ACTION" in
  install)
    python -m pip install -r models/lightgbm/requirements.txt
    ;;
  baseline)
    mkdir -p "$RUN_DIR"
    python -u -m models.lightgbm.run_baseline \
      --output-dir "$RUN_DIR" \
      --threads "$THREADS" \
      "$@" 2>&1 | tee -a "$RUN_DIR/console.log"
    ;;
  screen)
    mkdir -p "$SCREEN_DIR"
    python -u -m models.lightgbm.run_imbalance_screen \
      --output-dir "$SCREEN_DIR" \
      --threads "$THREADS" \
      "$@" 2>&1 | tee -a "$SCREEN_DIR/console.log"
    ;;
  tune)
    mkdir -p "$TUNE_DIR"
    python -u -m models.lightgbm.run_optuna_tuning \
      --output-dir "$TUNE_DIR" \
      --threads "$THREADS" \
      --n-trials "$N_TRIALS" \
      "$@" 2>&1 | tee -a "$TUNE_DIR/console.log"
    ;;
  refine)
    mkdir -p "$REFINE_DIR"
    python -u -m models.lightgbm.run_optuna_tuning \
      --search-profile local \
      --study-name lightgbm_rus_fpr_refinement \
      --output-dir "$REFINE_DIR" \
      --threads "$THREADS" \
      --n-trials "$REFINE_TRIALS" \
      "$@" 2>&1 | tee -a "$REFINE_DIR/console.log"
    ;;
  confirm)
    mkdir -p "$CONFIRM_DIR"
    python -u -m models.lightgbm.run_seed_confirmation \
      --output-dir "$CONFIRM_DIR" \
      --threads "$THREADS" \
      "$@" 2>&1 | tee -a "$CONFIRM_DIR/console.log"
    ;;
  lock-oof)
    mkdir -p "$LOCK_OOF_DIR"
    python -u -m models.lightgbm.run_fixed_oof_lock \
      --confirmation-dir "$CONFIRM_DIR" \
      --output-dir "$LOCK_OOF_DIR" \
      --threads "$THREADS" \
      "$@" 2>&1 | tee -a "$LOCK_OOF_DIR/console.log"
    ;;
  final-test)
    mkdir -p "$FINAL_TEST_DIR"
    python -u -m models.lightgbm.run_final_test \
      --lock-dir "$LOCK_OOF_DIR" \
      --output-dir "$FINAL_TEST_DIR" \
      "$@" 2>&1 | tee -a "$FINAL_TEST_DIR/console.log"
    ;;
  shap)
    mkdir -p "$FINAL_SHAP_DIR"
    python -u -m models.lightgbm.run_final_shap \
      --final-dir "$FINAL_TEST_DIR" \
      --output-dir "$FINAL_SHAP_DIR" \
      "$@" 2>&1 | tee -a "$FINAL_SHAP_DIR/console.log"
    ;;
  check)
    python -c "import lightgbm, numpy, pandas, pyarrow, sklearn; print('LightGBM', lightgbm.__version__); print('NumPy', numpy.__version__); print('pandas', pandas.__version__); print('PyArrow', pyarrow.__version__); print('scikit-learn', sklearn.__version__)"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage:
  bash models/lightgbm/run_autodl.sh install
  bash models/lightgbm/run_autodl.sh check
  bash models/lightgbm/run_autodl.sh baseline [extra run_baseline.py options]
  bash models/lightgbm/run_autodl.sh screen [extra run_imbalance_screen.py options]
  bash models/lightgbm/run_autodl.sh tune [extra run_optuna_tuning.py options]
  bash models/lightgbm/run_autodl.sh refine [extra run_optuna_tuning.py options]
  bash models/lightgbm/run_autodl.sh confirm [extra run_seed_confirmation.py options]
  bash models/lightgbm/run_autodl.sh lock-oof [extra run_fixed_oof_lock.py options]
  bash models/lightgbm/run_autodl.sh final-test --confirm-final-test I_UNDERSTAND_TEST_IS_ONE_TIME
  bash models/lightgbm/run_autodl.sh shap [extra run_final_shap.py options]

Environment variables:
  LGBM_THREADS   CPU threads used by LightGBM (default: 16)
  LGBM_RUN_DIR   persistent run directory (default: runs/lightgbm/baseline)
  LGBM_SCREEN_DIR persistent screening directory
                   (default: runs/lightgbm/imbalance_screen)
  LGBM_TUNE_DIR    persistent Optuna directory
                   (default: runs/lightgbm/optuna_rus)
  LGBM_N_TRIALS    target number of completed Optuna trials (default: 60)
  LGBM_REFINE_DIR  persistent local-refinement directory
                   (default: runs/lightgbm/optuna_refine)
  LGBM_REFINE_TRIALS target refinement trials (default: 40)
  LGBM_CONFIRM_DIR persistent multi-seed confirmation directory
                   (default: runs/lightgbm/seed_confirmation)
  LGBM_LOCK_OOF_DIR persistent fixed-tree OOF threshold-lock directory
                    (default: runs/lightgbm/fixed_oof_lock)
  LGBM_FINAL_TEST_DIR one-time final-test directory
                      (default: runs/lightgbm/final_test)
  LGBM_FINAL_SHAP_DIR post-test SHAP directory
                      (default: runs/lightgbm/final_test/shap_top_score)

All actions use persistent checkpoints. Optuna resumes from its SQLite study;
confirmation resumes from completed candidate/seed/fold JSON artifacts.
EOF
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
