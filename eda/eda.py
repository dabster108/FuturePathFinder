import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set style
sns.set(style="whitegrid")

# Paths
DATA_PATH = "/Users/dikshanta/Documents/FuturePathFinder/data/assigned_careers_dataset.csv"
SAVE_DIR = "/Users/dikshanta/Documents/FuturePathFinder/eda_visualizations/"

# Create folder if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df_orig = df.copy()  # Keep original for plots needing dropped columns

# Encode Field_of_Study to Field_of_Study_enc if not already present
if 'Field_of_Study_enc' not in df.columns and 'Field_of_Study' in df.columns:
    le_field = LabelEncoder()
    df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'].astype(str).str.strip())

# Features and target
features = ['Field_of_Study_enc', 'University_GPA', 'Internships_Completed',
            'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
target = 'Professional_Career'

# Filter relevant columns only
cols = features + [target]
df = df[cols]

# 1. Class Distribution Bar Chart (horizontal, colored by class, not compact)
plt.figure(figsize=(10, 5))
ax = sns.countplot(
    y=target,
    data=df,
    order=df[target].value_counts().index,
    palette="tab10"
)
plt.title("Class Distribution of Target Variable")
plt.xlabel("Count")
plt.ylabel(target)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}01_class_distribution.png")
plt.close()

# 2. Countplot of Field_of_Study (distribution of fields of study)
plt.figure(figsize=(12,6))
sns.countplot(y='Field_of_Study', data=df_orig, order=df_orig['Field_of_Study'].value_counts().index, palette="coolwarm", legend=False)
plt.title("Distribution of Field of Study")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}02_field_of_study_distribution.png")
plt.close()


# 3. Correlation Heatmap (Numerical Features)
plt.figure(figsize=(8,6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}03_correlation_heatmap.png")
plt.close()

# 4. Boxplot of Field_of_Study_enc vs Professional_Career
plt.figure(figsize=(12,5))
sns.boxplot(x=target, y='Field_of_Study_enc', data=df)
plt.title(f"Field_of_Study_enc vs {target}")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}04_boxplot_Field_of_Study_enc.png")
plt.close()

# 5. Histogram of University GPA
plt.figure(figsize=(10,5))
sns.histplot(df['University_GPA'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of University GPA")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}05_histogram_University_GPA.png")
plt.close()

# 6. Histogram of Projects Completed
plt.figure(figsize=(10,5))
sns.histplot(df['Projects_Completed'], bins=30, kde=False, color='salmon')
plt.title("Distribution of Projects Completed")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}06_histogram_Projects_Completed.png")
plt.close()

# 7. Histogram of Certifications
plt.figure(figsize=(10,5))
sns.histplot(df['Certifications'], bins=30, kde=False, color='olive')
plt.title("Distribution of Certifications")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}07_histogram_Certifications.png")
plt.close()

# 8. Violin Plot: Soft Skills Score by Career
plt.figure(figsize=(12,6))
sns.violinplot(x=target, y='Soft_Skills_Score', data=df, palette="muted")
plt.title("Soft Skills Score Distribution by Career")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}08_violinplot_Soft_Skills_Score.png")
plt.close()

# 9. Scatter Plot: Internships Completed vs Projects Completed colored by Career (improved)
plt.figure(figsize=(10, 7))
unique_careers = df[target].unique()
palette = sns.color_palette("tab10", n_colors=len(unique_careers))
ax = sns.scatterplot(
    x='Internships_Completed',
    y='Projects_Completed',
    hue=target,
    data=df,
    palette=palette,
    alpha=0.8,
    s=60,
    edgecolor='w',
    linewidth=0.7
)
plt.title("Internships Completed vs Projects Completed by Career")
plt.xlabel("Internships Completed")
plt.ylabel("Projects Completed")
plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}09_scatter_Internships_vs_Projects.png", bbox_inches='tight')
plt.close()

# 10. Pairplot of Selected Features colored by Career
subset_features = ['University_GPA', 'Internships_Completed', 'Projects_Completed', 'Certifications', target]
sns.pairplot(df[subset_features], hue=target, diag_kind='kde')
plt.savefig(f"{SAVE_DIR}10_pairplot_selected_features.png")
plt.close()

# 11. KDE Plot: University GPA by Career
plt.figure(figsize=(12,6))
sns.kdeplot(data=df, x='University_GPA', hue=target, fill=True, common_norm=False, alpha=0.5)
plt.title("University GPA Distribution by Career")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}11_kdeplot_University_GPA_by_Career.png")
plt.close()

print(f"✅ 11 EDA plots saved in {SAVE_DIR}")
