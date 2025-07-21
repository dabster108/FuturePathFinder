import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# === 0. Output Directory for Plots ===
output_dir = "/Users/dikshanta/Documents/FuturePathFinder/feature/feature_visualizations/"
os.makedirs(output_dir, exist_ok=True)

# === 1. Load Cleaned Dataset ===
df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv")

# === 2. Drop Unnecessary ID Columns ===
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])

# === 3. Feature Engineering ===
# Feature Engineering refers to the process of transforming raw data into meaningful features
# (e.g., encoding, creating new columns, combining features, etc.)

label_encoders = {}
for column in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
    # Example: 'Professional_Career' → 0, 1, 2, ...

# You can also add engineered features here if needed (e.g., GPA buckets, ratios, interactions)

# === 4. Split into Features and Target Variable ===
X = df.drop(columns=["Professional_Career"])  # All input features
y = df["Professional_Career"]  # Target label

# === 5. Train Model for Feature Importance ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === 6. Feature Importance Table ===
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n===== Feature Importances =====\n")
print(feature_importance.to_string(index=False))

# === 7. Bar Plot: Feature Importance ===
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette="viridis")
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.show()

# === 8. Heatmap: Feature Correlation ===
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

# === 9. Count Plot: Target Distribution ===
plt.figure(figsize=(10, 6))
sns.countplot(x=y, palette="Set2")
plt.title('Distribution of Professional Careers')
plt.xlabel('Professional Career (Encoded)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "career_distribution.png"))
plt.show()

# === 10. Pairplot: Top 4 Features ===
top_features = feature_importance['Feature'].head(4).tolist()
pairplot = sns.pairplot(df[top_features + ['Professional_Career']], hue='Professional_Career', palette="husl")
pairplot.fig.suptitle("Pairplot of Top 4 Important Features", y=1.02)
pairplot.savefig(os.path.join(output_dir, "pairplot_top_features.png"))
plt.show()


# === 11. Unique Plot: Cumulative Feature Importance Curve ===
feature_importance['Cumulative'] = feature_importance['Importance'].cumsum()

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance,
    palette="magma"
)
plt.plot(
    feature_importance['Cumulative'],
    feature_importance['Feature'],
    color='cyan',
    linewidth=2,
    marker='o',
    label='Cumulative Importance'
)
plt.axhline(
    y=feature_importance[feature_importance['Cumulative'] >= 0.95].iloc[0]['Feature'],
    color='red',
    linestyle='--',
    label='95% Threshold'
)
plt.title("Cumulative Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cumulative_feature_importance.png"))
plt.show()
