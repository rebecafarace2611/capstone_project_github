# Insurance Fraud Modelling Project

This repository contains the shared data pipeline and model-specific workflows
for the capstone project.

## Repository Structure

```text
models/
  rfqc/          Complete RFQC implementation, runners, utilities, and reporting
scripts/
  leakage/       Leakage audit and grouped train/test split
  descriptive/   Descriptive summaries and correlation analysis
tests/           Workflow tests
data/            Confidential data; ignored by Git
outputs/         Reviewed aggregate analysis and model results
reports/         Report figures and supporting files
```

Future models should be added beside RFQC, for example:

```text
models/xgboost/
models/random_forest/
models/logistic_regression/
```

The `models/rfqc/` directory is self-contained:

- `data.py`: shared RFQC data validation.
- `prepare_folds.py`: grouped cross-validation folds.
- `workflow.R`: core `randomForestSRC` RFQ functions.
- `run.R`, `run.ps1`, `run_autodl.sh`: execution entry points.
- `prepare_final_run.py`, `summarize_results.py`: finalisation utilities.
- `reporting/`: RFQC figure and stage-report generation.

## Team Workflow

Use Python 3.10 or later. Place these authorised files in `data/`:

```text
ddbb_fraud.csv
variables_ddbb_fraud.xlsx
```

They are not stored on GitHub, including in the private repository, because
they are confidential. Install the Python dependencies:

```text
python -m pip install -r requirements.txt
```

Run the shared pipeline from the repository root:

```text
python scripts/leakage/run_leakage_analysis.py
python scripts/leakage/run_grouped_train_test_split.py
python scripts/descriptive/prepare_descriptive_data.py
```

The commands create the approved feature list, leakage-safe grouped train/test
files, and descriptive outputs. The leakage and grouped-split steps may each
take several minutes on a laptop because they process all 555,094 records.

## RFQC

RFQC training is retained for reproducibility but does not need to be rerun by
team members without the R environment and server resources.

The locked model used `ntree=3000`, `mtry=24`, `nodesize=20`, `nsplit=10`,
Gini splitting, and the training-prevalence q-star threshold. Local RFQC
artifacts are organised under:

```text
outputs/rfqc/folds/
outputs/rfqc/experiment_archive/
reports/rfqc_stage_report/
```

The model helpers are invoked as:

```text
python models/rfqc/prepare_folds.py
.\models\rfqc\run.ps1 smoke --threads 4
python models/rfqc/reporting/generate_report_figures.py
```

The Word report builder reads the school template from
`RFQC_REPORT_TEMPLATE`, with a fallback to
`~/Downloads/MScBA_CapstoneReport_StudentTemplate.docx`.

## GitHub Policy

The private repository includes source code, dependencies, tests, leakage and
descriptive outputs, aggregate RFQC results, tuning records, and report
figures. Final Word reports remain local.

Do not upload raw or processed datasets, the company variable dictionary,
train/test Parquet files, row-level predictions, fold assignments, fitted
models, local environments, logs, credentials, or temporary files.

Before pushing, review `git status --short --ignored` and confirm confidential
files are ignored rather than staged.
