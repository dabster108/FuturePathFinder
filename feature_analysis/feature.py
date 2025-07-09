
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# --- Configuration ---
# Create a directory to save the visualizations
output_dir = 'feature_analysis/feature_visualizations'
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
# Load the cleaned dataset
try:
    df = pd.read_csv('datasets/cleaned_education_data.csv')
except FileNotFoundError:
    print("Error: 'datasets/cleaned_education_data.csv' not found.")
    print("Please ensure the data cleaning pipeline has been run.")
    exit()

# Define features and targets based on the dataset columns
numerical_features = ['High_School_GPA', 'University_GPA', 'SAT_Score', 'Soft_Skills_Score']
categorical_features = ['Field_of_Study', 'Interest', 'Current_Job_Level']
# Assuming 'Career_Domain' is our classification target. If not present, we can derive it or use another categorical column.
# For this script, we'll use 'Field_of_Study' as the classification target for demonstration if 'Career_Domain' is absent.
if 'Career_Domain' not in df.columns:
    print("Warning: 'Career_Domain' not found. Using 'Field_of_Study' as the classification target for demonstration.")
    df['Career_Domain'] = df['Field_of_Study']

regression_target = 'Starting_Salary'
classification_target = 'Career_Domain'

# --- Data Preparation for Modeling ---
# Create a copy for analysis to avoid modifying the original dataframe
analysis_df = df.copy()

# Encode categorical features for modeling
label_encoders = {}
for col in categorical_features + [classification_target]:
    le = LabelEncoder()
    analysis_df[col] = le.fit_transform(analysis_df[col])
    label_encoders[col] = le

# ### 1. Feature Importance Analysis
# We analyze feature importance using two methods:
# 1.  **Correlation Matrix**: For numerical features against the regression target.
# 2.  **ExtraTrees Models**: A powerful ensemble method to rank feature importance for both regression and classification tasks.

print("--- 1. Feature Importance Analysis ---")

# Correlation Matrix for Numerical Features
plt.figure(figsize=(10, 8))
correlation_matrix = analysis_df[numerical_features + [regression_target]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features and Starting Salary')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.show()
print("Saved correlation matrix plot.")

# Feature Importance with ExtraTreesRegressor
X = analysis_df[numerical_features + categorical_features]
y_reg = analysis_df[regression_target]

et_regressor = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_regressor.fit(X, y_reg)

importances_reg = pd.Series(et_regressor.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 8))
importances_reg.sort_values().plot(kind='barh')
plt.title('Feature Importance for Starting Salary (Regression)')
plt.savefig(os.path.join(output_dir, 'feature_importance_regression.png'))
plt.show()
print("Saved regression feature importance plot.")

# Feature Importance with ExtraTreesClassifier
y_clf = analysis_df[classification_target]

et_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_classifier.fit(X, y_clf)

importances_clf = pd.Series(et_classifier.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 8))
importances_clf.sort_values().plot(kind='barh')
plt.title(f'Feature Importance for {classification_target.replace("_", " ")} (Classification)')
plt.savefig(os.path.join(output_dir, 'feature_importance_classification.png'))
plt.show()
print("Saved classification feature importance plot.")


# ### 2. Target Variable Distribution
# Understanding the distribution of the target variable is crucial.
# - For regression, we check for normality.
# - For classification, we check for class balance.

print("\n--- 2. Target Variable Distribution ---")

# Regression Target Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df[regression_target], kde=True)
plt.title('Distribution of Starting Salary (Regression Target)')
plt.savefig(os.path.join(output_dir, 'regression_target_distribution.png'))
plt.show()
print("Saved regression target distribution plot.")

# Classification Target Distribution
plt.figure(figsize=(12, 8))
sns.countplot(y=df[classification_target], order=df[classification_target].value_counts().index)
plt.title(f'Class Balance for {classification_target.replace("_", " ")} (Classification Target)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classification_target_balance.png'))
plt.show()
print("Saved classification target balance plot.")


# ### 3. Feature Relationship Visualization
# Visualizing relationships between features helps uncover complex patterns.
# - **Pairplots** show pairwise relationships.
# - **Interaction Heatmaps** can reveal how two features jointly affect a target.

print("\n--- 3. Feature Relationship Visualization ---")

# Pairplot of Key Features
key_features = numerical_features + [classification_target]
plt.figure(figsize=(12, 12))
sns.pairplot(df[key_features], hue=classification_target, palette='viridis')
plt.suptitle('Pairwise Relationships of Key Features', y=1.02)
plt.savefig(os.path.join(output_dir, 'key_features_pairplot.png'))
plt.show()
print("Saved key features pairplot.")

# Interaction Heatmap (e.g., Field of Study and Interest vs. Starting Salary)
interaction_pivot = df.pivot_table(values='Starting_Salary', index='Field_of_Study', columns='Interest', aggfunc='mean')
plt.figure(figsize=(14, 10))
sns.heatmap(interaction_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('Mean Starting Salary by Field of Study and Interest')
plt.savefig(os.path.join(output_dir, 'salary_interaction_heatmap.png'))
plt.show()
print("Saved interaction heatmap.")


# ### 4. Dimensionality Reduction (Optional)
# For high-dimensional data, PCA and t-SNE help visualize feature space in 2D.

print("\n--- 4. Dimensionality Reduction (Optional) ---")

# Scale numerical features before applying PCA/t-SNE
X_scaled = StandardScaler().fit_transform(analysis_df[numerical_features])

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df[classification_target] = df[classification_target] # Use original labels for coloring

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue=classification_target, data=pca_df, palette='deep')
plt.title('PCA of Numerical Features')
plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
plt.show()
print("Saved PCA visualization.")

# t-SNE (can be slow on large datasets)
print("Running t-SNE (this might take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(data=X_tsne, columns=['t-SNE1', 't-SNE2'])
tsne_df[classification_target] = df[classification_target]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue=classification_target, data=tsne_df, palette='deep')
plt.title('t-SNE of Numerical Features')
plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
plt.show()
print("Saved t-SNE visualization.")


# ### 5. Business Rule Validation
# Here, we can enforce domain-specific rules. For example, we can filter out illogical combinations
# of 'Field_of_Study' and 'Interest'.

print("\n--- 5. Business Rule Validation ---")

# Example: Define a mapping of plausible interests for each field of study
# This is a simplified example. A real-world scenario would require expert knowledge.
valid_interest_map = {
    'Computer Science': ['Software Development', 'Data Science', 'Cybersecurity'],
    'Engineering': ['Robotics', 'Mechanical Engineering', 'Renewable Energy'],
    'Business': ['Marketing', 'Finance', 'Entrepreneurship'],
    'Healthcare': ['Medicine', 'Nursing', 'Public Health'],
    'Arts & Humanities': ['Creative Writing', 'Graphic Design', 'History']
}

# Filter out rows where the interest does not align with the field of study
initial_rows = len(df)
df_filtered = df[df.apply(lambda row: row['Interest'] in valid_interest_map.get(row['Field_of_Study'], [row['Interest']]), axis=1)]
removed_rows = initial_rows - len(df_filtered)

print(f"Initial dataset size: {initial_rows} rows")
print(f"Filtered dataset size after applying business rules: {len(df_filtered)} rows")
print(f"Removed {removed_rows} rows with potentially illogical field-interest pairs.")


# ### 6. Train-Test Split Preparation
# Finally, we prepare the data for modeling by performing a stratified train-test split.
# Stratification ensures that the class distribution in the target variable is preserved
# in both the training and testing sets, which is crucial for classification tasks.

print("\n--- 6. Train-Test Split Preparation ---")

# Use the filtered and encoded dataframe for splitting
X_final = analysis_df.drop([regression_target, classification_target], axis=1, errors='ignore')
y_final = analysis_df[classification_target]

# Perform stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("\nOriginal class distribution:")
print(df[classification_target].value_counts(normalize=True))
print("\nTraining set class distribution:")
print(y_train.value_counts(normalize=True))
print("\nTest set class distribution:")
print(y_test.value_counts(normalize=True))

print("\nPre-modeling analysis complete. Visualizations are saved in 'feature_analysis/feature_visualizations/'.")
