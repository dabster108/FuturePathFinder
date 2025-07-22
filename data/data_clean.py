import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from math import pi

# === 1. Load Dataset ===
input_path = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/assigned_careers_dataset.csv"
df = pd.read_csv(input_path)

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
# Numeric columns: fill missing with median
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled missing numeric values in '{col}' with median: {median_val}")

# Categorical columns: fill missing with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Filled missing categorical values in '{col}' with mode: '{mode_val}'")

# === 8. Encode categorical columns (create new encoded columns) ===
to_encode = ['Gender', 'Current_Job_Level', 'Entrepreneurship', 'Professional_Career']
label_encoders = {}
for col in to_encode:
    if col in df.columns:
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

# Visualization folder
vis_folder = "/Users/dikshanta/Documents/FuturePathFinder/data/data_clean_visualizations"
os.makedirs(vis_folder, exist_ok=True)

# === Missingno: Missing Value Matrix ===
plt.figure(figsize=(12, 6))
msno.matrix(df, color=(0.25, 0.4, 0.8))
plt.title("Missing Value Matrix")
plt.tight_layout()
plt.savefig(f"{vis_folder}/missing_matrix_plot.png")
plt.close()

# === Missingno: Dendrogram of Missing Data Correlation ===
plt.figure(figsize=(10, 6))
msno.dendrogram(df)
plt.title("Missing Data Hierarchical Clustering")
plt.tight_layout()
plt.savefig(f"{vis_folder}/missing_dendrogram.png")
plt.close()

# === Curved Radial Plot (Polar Chart) for categorical distribution ===
def plot_radial_distribution(column):
    counts = df[column].value_counts(normalize=True)
    categories = counts.index.tolist()
    values = counts.values.tolist()
    
    values += values[:1]  # close the loop
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color='teal', linewidth=2)
    ax.fill(angles, values, color='skyblue', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    plt.title(f"Radial Distribution of {column}", y=1.08)
    plt.tight_layout()
    plt.savefig(f"{vis_folder}/radial_{column}.png")
    plt.close()

# Example usage for 'Gender' if present
if 'Gender' in df.columns:
    plot_radial_distribution('Gender')

# === Before/After Boxplot of Numeric Columns ===
imputed_num_cols = [col for col in num_cols if missing_before[col] > 0]
if imputed_num_cols:
    for col in imputed_num_cols:
        df_original = pd.read_csv(input_path)
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

        data_combined = pd.DataFrame({
            'Before': df_original[col],
            'After': df[col]
        })

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=data_combined, palette="coolwarm")
        plt.title(f"Boxplot of '{col}' Before vs After Imputation")
        plt.tight_layout()
        plt.savefig(f"{vis_folder}/boxplot_{col}_before_after.png")
        plt.close()

# === Violin Plot (if encoded Gender and target available) ===
if 'Gender_enc' in df.columns and 'University_GPA' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Gender_enc', y='University_GPA', data=df, palette='Set2')
    plt.title("University GPA Distribution by Gender")
    plt.tight_layout()
    plt.savefig(f"{vis_folder}/violinplot_gpa_gender.png")
    plt.close()

# === Countplot for a categorical column after cleaning ===
example_cat = "Gender" if "Gender" in df.columns else (cat_cols[0] if cat_cols else None)
if example_cat:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df[example_cat], palette="Set2")
    plt.title(f"Categorical Distribution: {example_cat} (After Cleaning)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{vis_folder}/countplot_{example_cat}_after_cleaning.png")
    plt.close()
