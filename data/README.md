# Local Confidential Data

This directory is for authorized local use only. Its contents, except this
README, are excluded from Git.

Files required before running the team workflow:

- `ddbb_fraud.csv`
- `variables_ddbb_fraud.xlsx`

Files generated locally by the grouped split:

- `train_model_dataset.parquet`
- `test_model_dataset.parquet`

These files contain or describe proprietary insurance company data. Do not
upload them to GitHub, attach them to issues or pull requests, paste samples
into documentation, or share screenshots of their contents.

Derived datasets are confidential even when direct identifiers have been
removed. Aggregated summaries and variable dictionaries can also disclose
commercially sensitive information and must remain local unless the company
explicitly approves publication.

Before pushing changes, run `git status --ignored` and confirm all local data
files appear as ignored.
