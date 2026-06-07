# ============================================================
# INSPECT INSURANCE DATASET
# Outputs:
# 1. Data dictionary
# 2. Variable types
# 3. Missing value report
# ============================================================


# Import libraries
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

file_path = (r"C:\Users\rebec\OneDrive\Documentos\UCD Third Trimester\Capstone Project\data\ddbb_fraud.csv")

# Read the dataset
df = pd.read_csv(file_path)

# Show basic shape of the dataset
#print("Dataset loaded successfully.")
#print(f"Number of rows: {df.shape[0]}")
#print(f"Number of columns: {df.shape[1]}")

# Display first rows
#print(df.head())


# ------------------------------------------------------------
# 2. BASIC DATASET OVERVIEW
# ------------------------------------------------------------

# Shows column names, data types, and non-null counts
#df.info()

# Shows summary statistics for numeric variables
#df.describe()

# Shows summary statistics for categorical/text variables
#df.describe(include="object")


# ------------------------------------------------------------
# 3. CREATE VARIABLE TYPES REPORT
# ------------------------------------------------------------

# This creates a table showing each variable and its Python data type
variable_types = pd.DataFrame({
    "variable_name": df.columns,
    "data_type": df.dtypes.astype(str).values
})

# Display report
#print(variable_types)


# ------------------------------------------------------------
# 4. CREATE MISSING VALUE REPORT
# ------------------------------------------------------------

missing_report = pd.DataFrame({
    "variable_name": df.columns,
    "missing_count": df.isnull().sum().values,
    "missing_percentage": (df.isnull().sum().values / len(df)) * 100
})

# Sort from most missing to least missing
missing_report = missing_report.sort_values(
    by="missing_percentage", 
    ascending=False
)

#print(missing_report)


# ------------------------------------------------------------
# 5. CREATE INITIAL DATA DICTIONARY
# ------------------------------------------------------------

data_dictionary = pd.DataFrame({
    "variable_name": df.columns,
    "data_type": df.dtypes.astype(str).values,
    "non_missing_count": df.notnull().sum().values,
    "missing_count": df.isnull().sum().values,
    "missing_percentage": (df.isnull().sum().values / len(df)) * 100,
    "unique_values": df.nunique().values
})

# Add a simple automatic classification of variable type
def classify_variable(series):
    """
    #This function gives a simple interpretation of each variable.
    #It is not perfect, but it is useful for the first inspection.
    #"""
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return "Numeric categorical / discrete"
        else:
            return "Numeric continuous"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "Date/time"
    else:
        return "Categorical / text"

data_dictionary["interpreted_type"] = [
    classify_variable(df[col]) for col in df.columns
]

#print(data_dictionary)


# ------------------------------------------------------------
# 6. DEFINE TARGET VARIABLE
# ------------------------------------------------------------

target_column = "respuesta_dicot_c"

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

print("Target variable found:", target_column)

print("\nTarget values before conversion:")
print(df[target_column].value_counts(dropna=False))

print("\nUnique target values before conversion:")
print(df[target_column].unique())


# ------------------------------------------------------------
# 7. CLEAN AND CONVERT TARGET VARIABLE
# ------------------------------------------------------------

# Remove extra spaces and make text consistent
df[target_column] = df[target_column].astype(str).str.strip().str.upper()

# Convert target labels into binary values
# 0 = not fraud
# 1 = fraud
df[target_column] = df[target_column].map({
    "NO FRAUDE": 0,
    "FRAUDE": 1
})

# Safety check: if any value became missing, it means there was an unexpected label
if df[target_column].isnull().sum() > 0:
    print("\nUnexpected target values found:")
    print(df.loc[df[target_column].isnull(), target_column].unique())
    raise ValueError("Target conversion failed. Check target labels.")

print("\nTarget distribution after conversion:")
print(df[target_column].value_counts())

print("\nTarget percentages after conversion:")
print(df[target_column].value_counts(normalize=True) * 100)


# ------------------------------------------------------------
# 8. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ------------------------------------------------------------
# 9. IDENTIFY NUMERIC AND CATEGORICAL VARIABLES
# ------------------------------------------------------------

numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nNumber of numeric columns:", len(numeric_columns))
print("Number of categorical columns:", len(categorical_columns))

print("\nCategorical columns:")
print(categorical_columns)


# ------------------------------------------------------------
# 10. HANDLE MISSING VALUES
# ------------------------------------------------------------

for col in numeric_columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

for col in categorical_columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna("Missing")

print("\nMissing values after preprocessing:")
print(X.isnull().sum().sum())


# ------------------------------------------------------------
# 11. ENCODE CATEGORICAL VARIABLES
# ------------------------------------------------------------

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("\nShape before encoding:", X.shape)
print("Shape after encoding:", X_encoded.shape)


# ------------------------------------------------------------
# 12. FEATURE TRANSFORMATIONS
# ------------------------------------------------------------

transformed_columns = []

for col in numeric_columns:

    if col in X_encoded.columns:

        non_negative = (X_encoded[col] >= 0).all()
        many_unique_values = X_encoded[col].nunique() > 20
        skewness = X_encoded[col].skew()

        if non_negative and many_unique_values and abs(skewness) > 2:
            new_col = col + "_log1p"
            X_encoded[new_col] = np.log1p(X_encoded[col])
            transformed_columns.append(new_col)

print("\nLog-transformed columns created:")
print(transformed_columns)


# ------------------------------------------------------------
# 13. CREATE CLEAN MODELLING DATASET
# ------------------------------------------------------------

clean_model_df = X_encoded.copy()
clean_model_df[target_column] = y.values

print("\nClean modelling dataset shape:")
print(clean_model_df.shape)

print("\nFinal missing values:")
print(clean_model_df.isnull().sum().sum())


# ------------------------------------------------------------
# 14. SAVE CLEAN MODELLING DATASET
# ------------------------------------------------------------

#output_file_clean = r"C:\Users\rebec\OneDrive\Documentos\UCD Third Trimester\Capstone Project\data\clean_modelling_dataset_step3.csv"

#clean_model_df.to_csv(output_file_clean, index=False)

#print("\nClean modelling dataset saved as:")
#print(output_file_clean)


# ------------------------------------------------------------
# 15. SAVE PREPROCESSING SUMMARY
# ------------------------------------------------------------

preprocessing_summary = pd.DataFrame({
    "item": [
        "Original rows",
        "Original columns",
        "Numeric variables before encoding",
        "Categorical variables before encoding",
        "Columns after encoding",
        "Log-transformed columns created",
        "Final rows",
        "Final columns",
        "Final missing values",
        "Fraud cases",
        "Non-fraud cases",
        "Fraud rate (%)"
    ],
    "value": [
        df.shape[0],
        df.shape[1],
        len(numeric_columns),
        len(categorical_columns),
        X_encoded.shape[1],
        len(transformed_columns),
        clean_model_df.shape[0],
        clean_model_df.shape[1],
        clean_model_df.isnull().sum().sum(),
        int(y.sum()),
        int((y == 0).sum()),
        round(y.mean() * 100, 4)
    ]
})

summary_file = r"C:\Users\rebec\OneDrive\Documentos\UCD Third Trimester\Capstone Project\data\step3_preprocessing_summary.csv"

preprocessing_summary.to_csv(summary_file, index=False)

print("\nPreprocessing summary saved as:")
print(summary_file)