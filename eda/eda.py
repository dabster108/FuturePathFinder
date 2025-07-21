import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

sns.set(style="whitegrid")

# File paths
DATA_PATH = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv"
SAVE_DIR = "/Users/dikshanta/Documents/FuturePathFinder/eda_visualizations/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df_orig = df.copy()

# Encode Field_of_Study if needed
if 'Field_of_Study_enc' not in df.columns and 'Field_of_Study' in df.columns:
    le_field = LabelEncoder()
    df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'].astype(str).str.strip())

# Feature and target selection
features = ['Field_of_Study_enc', 'University_GPA', 'Internships_Completed',
            'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
target = 'Professional_Career'
cols = features + [target]
df = df[cols]

# 1. Scatter Plot: University GPA vs Soft Skills Score
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x='University_GPA',
    y='Soft_Skills_Score',
    hue=target,
    data=df,
    palette="tab10",
    alpha=0.8,
    s=60,
    edgecolor='w',
    linewidth=0.7
)
plt.title("University GPA vs Soft Skills Score by Career")
plt.xlabel("University GPA")
plt.ylabel("Soft Skills Score")
plt.legend(title=target, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f"{SAVE_DIR}01_scatter_GPA_vs_SoftSkills.png", bbox_inches='tight')
plt.close()

# 2. Field of Study Distribution
plt.figure(figsize=(12,6))
sns.countplot(
    y='Field_of_Study',
    data=df_orig,
    order=df_orig['Field_of_Study'].value_counts().index,
    palette="coolwarm"
)
plt.title("Distribution of Field of Study")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}02_field_of_study_distribution.png")
plt.close()

# 3. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}03_correlation_heatmap.png")
plt.close()

# 4. Boxplot: Field_of_Study_enc vs Career
plt.figure(figsize=(12,5))
sns.boxplot(x=target, y='Field_of_Study_enc', data=df)
plt.title("Field_of_Study_enc vs Career")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}04_boxplot_Field_of_Study_enc.png")
plt.close()

# 5. Histogram: University GPA
plt.figure(figsize=(10,5))
sns.histplot(df['University_GPA'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of University GPA")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}05_histogram_University_GPA.png")
plt.close()

# 6. Histogram: Projects Completed
plt.figure(figsize=(10,5))
sns.histplot(df['Projects_Completed'], bins=30, kde=False, color='salmon')
plt.title("Distribution of Projects Completed")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}06_histogram_Projects_Completed.png")
plt.close()

# 7. Histogram: Certifications
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

# 9. Scatter Plot: Internships vs Projects Completed
plt.figure(figsize=(7, 5))
unique_careers = df[target].unique()
palette = sns.color_palette("tab10", n_colors=len(unique_careers))
sns.scatterplot(
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
plt.legend(title=target, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f"{SAVE_DIR}09_scatter_Internships_vs_Projects.png", bbox_inches='tight')
plt.close()

# 10. Pairplot
subset_features = ['University_GPA', 'Internships_Completed', 'Projects_Completed', 'Certifications', target]
sns.pairplot(df[subset_features], hue=target, diag_kind='kde')
plt.savefig(f"{SAVE_DIR}10_pairplot_selected_features.png")
plt.close()

# 11. KDE Plot: GPA by Career
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set2", n_colors=df[target].nunique())

sns.kdeplot(
    data=df,
    x='University_GPA',
    hue=target,
    fill=True,
    alpha=0.4,
    linewidth=2,
    palette=palette,
    common_norm=False
)

plt.axvline(df['University_GPA'].mean(), color='gray', linestyle='--', linewidth=1.5, label='Overall Mean GPA')

plt.title("University GPA Distribution Across Careers", fontsize=14, fontweight='bold')
plt.xlabel("University GPA", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title=target, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f"{SAVE_DIR}11_kdeplot_unique_University_GPA_by_Career.png", bbox_inches='tight')
plt.close()

print(f"✅ Unique KDE plot saved as '11_kdeplot_unique_University_GPA_by_Career.png' in {SAVE_DIR}")

