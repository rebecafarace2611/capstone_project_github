# RFQC Model

This directory contains the complete Random Forest Quantile Classifier
workflow. Team members working on other models do not need to run these files.

## Contents

- `data.py`: RFQC-specific data validation and feature typing.
- `prepare_folds.py`: deterministic grouped cross-validation folds.
- `workflow.R`: core `randomForestSRC::imbalanced(method = "rfq")` functions.
- `run.R`: native R training, tuning, recovery, and final evaluation.
- `run.ps1`: Windows wrapper for `run.R`.
- `run_autodl.sh`: Linux/AutoDL staged execution wrapper.
- `install_packages.R`: project-local R dependency installation.
- `prepare_final_run.py`: locks the selected configuration before testing.
- `summarize_results.py`: combines baseline, CV, and final-test results.
- `reporting/`: RFQC figures and Word stage-report generation.

Reviewed aggregate RFQC results are stored under `outputs/rfqc/` in the private
repository. Row-level predictions, fold assignments, fitted models, logs, and
temporary server runs remain excluded from Git.
