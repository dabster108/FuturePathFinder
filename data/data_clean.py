import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# === 1. Load Dataset ===
input_path = "/Users/dikshanta/Documents/FuturePathFinder/data/assigned_careers_dataset.csv"
df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/datasets/assigned_careers_dataset.csv")

print(f"Initial shape: {df.shape}")

# === 2. Drop unnecessary ID column if present ===
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])
    print("Dropped 'Student_ID' column.")

# === 3. Remove completely empty rows ===
empty_rows = df.isnull().all(axis=1) | (df.applymap(lambda x: str(x).strip() == '').all(axis=1))
df = df[~empty_rows]
print(f"Removed {empty_rows.sum()} completely empty rows.")

# === 4. Replace empty strings with NaN for easier processing ===
df = df.replace(r'^\s*$', np.nan, regex=True)

# === 5. Strip whitespace from all string columns ===
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# === 6. Summary of missing values before imputation ===
missing_before = df.isnull().sum()
print("\nMissing values before imputation:")
print(missing_before[missing_before > 0])

# === 7. Handle Missing Values ===
# Numeric columns: fill missing with median (keep original values as is)
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled missing numeric values in '{col}' with median: {median_val}")

# Categorical (text) columns: fill missing with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Filled missing categorical values in '{col}' with mode: '{mode_val}'")

# === 8. Encode categorical columns (create new encoded columns, keep original columns intact) ===
# Here you can encode specific categorical columns for modeling
to_encode = ['Gender', 'Current_Job_Level', 'Entrepreneurship', 'Professional_Career']
label_encoders = {}
for col in to_encode:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoded '{col}' into '{col}_enc'.")

# === 9. Final check for missing values ===
missing_after = df.isnull().sum()
print("\nMissing values after imputation (should be 0):")
print(missing_after[missing_after > 0])

# === 10. Save cleaned + encoded data ===
output_path = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Cleaned & preprocessed data saved to:\n{output_path}")
print(f"Final shape: {df.shape}")
