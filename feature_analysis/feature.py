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
output_dir = '/Users/dikshanta/Documents/FuturePathFinder/feature_analysis/feature_visualizations'
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
try:
    df = pd.read_csv('/Users/dikshanta/Documents/FuturePathFinder/datasets/cleaned_final_dataset.csv')
except FileNotFoundError:
    print("Error: 'cleaned_final_dataset.csv' not found.")
    exit()

# Define features and targets
numerical_features = ['High_School_GPA', 'University_GPA', 'SAT_Score', 'Soft_Skills_Score']
categorical_features = ['Field_of_Study', 'Interest', 'Current_Job_Level']

# Use 'Career' as the classification target
if 'Career' not in df.columns:
    print("Error: 'Career' column not found in dataset.")
    exit()

classification_target = 'Career'
regression_target = 'Starting_Salary'

# Prepare data
analysis_df = df.copy()

label_encoders = {}
for col in categorical_features + [classification_target]:
    le = LabelEncoder()
    analysis_df[col] = le.fit_transform(analysis_df[col])
    label_encoders[col] = le

# --- 1. Feature Importance ---

plt.figure(figsize=(10, 8))
correlation_matrix = analysis_df[numerical_features + [regression_target]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features and Starting Salary')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# ExtraTrees Regressor
X = analysis_df[numerical_features + categorical_features]
y_reg = analysis_df[regression_target]

et_regressor = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_regressor.fit(X, y_reg)

importances_reg = pd.Series(et_regressor.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 8))
importances_reg.sort_values().plot(kind='barh')
plt.title('Feature Importance for Starting Salary (Regression)')
plt.savefig(os.path.join(output_dir, 'feature_importance_regression.png'))
plt.close()

# ExtraTrees Classifier
y_clf = analysis_df[classification_target]

et_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_classifier.fit(X, y_clf)

importances_clf = pd.Series(et_classifier.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 8))
importances_clf.sort_values().plot(kind='barh')
plt.title(f'Feature Importance for {classification_target} (Classification)')
plt.savefig(os.path.join(output_dir, 'feature_importance_classification.png'))
plt.close()

# --- 2. Target Distribution ---

plt.figure(figsize=(10, 6))
sns.histplot(df[regression_target], kde=True)
plt.title('Distribution of Starting Salary')
plt.savefig(os.path.join(output_dir, 'regression_target_distribution.png'))
plt.close()

plt.figure(figsize=(12, 8))
sns.countplot(y=df[classification_target], order=df[classification_target].value_counts().index)
plt.title(f'Career Class Balance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classification_target_balance.png'))
plt.close()

# --- 3. Feature Relationships ---

key_features = numerical_features + [classification_target]
sns.pairplot(df[key_features], hue=classification_target, palette='viridis')
plt.suptitle('Pairwise Relationships of Key Features', y=1.02)
plt.savefig(os.path.join(output_dir, 'key_features_pairplot.png'))
plt.close()

interaction_pivot = df.pivot_table(values='Starting_Salary', index='Field_of_Study', columns='Interest', aggfunc='mean')
plt.figure(figsize=(14, 10))
sns.heatmap(interaction_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('Mean Starting Salary by Field of Study and Interest')
plt.savefig(os.path.join(output_dir, 'salary_interaction_heatmap.png'))
plt.close()

# --- 4. Dimensionality Reduction ---

X_scaled = StandardScaler().fit_transform(analysis_df[numerical_features])

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df[classification_target] = df[classification_target]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue=classification_target, data=pca_df, palette='deep')
plt.title('PCA of Numerical Features')
plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
plt.close()

# t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) - 1))
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(data=X_tsne, columns=['t-SNE1', 't-SNE2'])
tsne_df[classification_target] = df[classification_target]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue=classification_target, data=tsne_df, palette='deep')
plt.title('t-SNE of Numerical Features')
plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
plt.close()

# --- 5. Business Rule Validation ---

valid_interest_map = {
    'Computer Science': ['Software Development', 'Data Science', 'Cybersecurity'],
    'Engineering': ['Robotics', 'Mechanical Engineering', 'Renewable Energy'],
    'Business': ['Marketing', 'Finance', 'Entrepreneurship'],
    'Medicine': ['Public Health', 'Surgery', 'Pharmacology'],
    'Law': ['Corporate Law', 'Environmental Law'],
    'Arts': ['Creative Writing', 'Graphic Design']
}

initial_rows = len(df)
df_filtered = df[df.apply(lambda row: row['Interest'] in valid_interest_map.get(row['Field_of_Study'], [row['Interest']]), axis=1)]
removed_rows = initial_rows - len(df_filtered)

print(f"Initial dataset size: {initial_rows}")
print(f"Filtered dataset size: {len(df_filtered)}")
print(f"Removed: {removed_rows} illogical rows")

# --- 6. Train-Test Split ---

X_final = analysis_df.drop([regression_target, classification_target], axis=1, errors='ignore')
y_final = analysis_df[classification_target]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print("\nClass distribution:")
print(df[classification_target].value_counts(normalize=True))

print("\nFeature analysis complete. Visualizations saved in:")
print(f"📁 {output_dir}")
