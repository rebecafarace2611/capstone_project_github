# Leakage Analysis Report

## Decision context

- Prediction point: immediately after first claim notification and before fraud investigation.
- Target: `respuesta_dicot_c` (`0` non-fraud, `1` fraud).
- Approved model features: **144**.
- Excluded features: **3**.
- Final modelling files use **Parquet** format.

## Confirmed exclusions

- `situacionvts`: Claim status records investigation/adjudication outcomes and deterministically reveals the target.
- `scoreada`: Output of an undocumented ADA scoring system; its inputs and generation time cannot be audited.
- `respuesta_dicot1`: Text representation of the fraud target; it maps exactly to the binary target.

## Final conclusion

The feature audit approves **144** predictors. The original
random split contained 712 duplicate feature vectors across train and test. This issue
has been corrected by grouping rows on all approved features and assigning every group
entirely to one partition.

The final grouped split is approved for model development and evaluation:

| Check | Result |
|---|---:|
| Source rows | 555094 |
| Approved features | 144 |
| Train rows | 444074 |
| Test rows | 111020 |
| Train fraud cases | 1860 |
| Test fraud cases | 465 |
| Train fraud rate | 0.418849% |
| Test fraud rate | 0.418843% |
| Unique feature groups | 552403 |
| Cross-split feature-group overlap | 0 |
| Cross-split full-row overlap | 0 |
| Train duplicate rows retained within train | 3503 |
| Test duplicate rows retained within test | 876 |
| Feature groups with conflicting labels | 2 |
| Missing cells in final split | 0 |

Duplicate observations are retained within a single partition rather than deleted.
The two identical-feature groups with conflicting labels are also kept together in one
partition, preventing leakage while preserving the supplied outcomes.

## Statistical proxy screen

Every non-target field was screened using the training data available during the
initial leakage audit. Numeric fields received Pearson correlation and orientation-free
univariate ROC-AUC. Categorical fields received Cramer's V, class purity, and
deterministic-mapping checks. Strong association alone was not treated as leakage.

## Limitations

- No claim, policy, customer, or vehicle identifier is present, so entity-level overlap beyond identical approved feature vectors cannot be tested.
- No claim date or external-data vintage is present, so temporal alignment of area-level statistics cannot be verified directly.
- The supplied dataset already has no missing values, so the historical fitting scope of any earlier imputation cannot be reconstructed.

## Final files and reproducibility

- Train: `data/train_model_dataset.parquet`
- Test: `data/test_model_dataset.parquet`
- Split method: stratified five-fold group split, fold 0 used as the 20% test set.
- Grouping key: hash of all approved features.
- Random seed: `42`
- Python: `3.13.7`
- pandas: `2.3.3`
- NumPy: `2.3.3`
- SciPy: `1.16.3`
- scikit-learn: `1.9.0`
- SHA-256:
  - `ddbb_fraud.csv`: `568e578babbbca67200f7620636b72d41f5c776225748bbd32e8816074b6a916`
  - `train_model_dataset.parquet`: `823ec10fbdb9f9ab5cb05144ddda5219b8cdeb41354b02b6e7fd5738cc0da2e3`
  - `test_model_dataset.parquet`: `2e9d2e888505e03dccdd40c70ad9df007d18e760570291d8d1b0c2854f724907`
  - `variables_ddbb_fraud.xlsx`: `2249d01b304fd0211c30925f505b0a3d4950e1135d9841b0f0c60f90ed2a0f96`
