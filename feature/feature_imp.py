import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# === 0. Output Directory for Plots ===
output_dir = "/Users/dikshanta/Documents/FuturePathFinder/feature/feature_visualizations/"
os.makedirs(output_dir, exist_ok=True)

# === 1. Load Dataset ===
df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv")

# === 2. Drop unnecessary ID column if present ===
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])

# === 3. Encode categorical features ===
label_encoders = {}
for column in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# === 4. Split features and target ===
X = df.drop(columns=["Professional_Career"])
y = df["Professional_Career"]

# === 5. Train Random Forest Classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === 6. Feature Importance (Terminal Output) ===
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n===== Feature Importances =====\n")
print(feature_importance.to_string(index=False))

# === 7. Plot: Feature Importance ===
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette="viridis")
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.show()

# === 8. Plot: Correlation Heatmap ===
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

# === 9. Plot: Countplot of Target Variable ===
plt.figure(figsize=(10, 6))
sns.countplot(x=y, palette="Set2")
plt.title('Distribution of Professional Careers')
plt.xlabel('Professional Career')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "career_distribution.png"))
plt.show()

# === 10. Plot: Pairplot of Top Features ===
top_features = feature_importance['Feature'].head(4).tolist()
pairplot = sns.pairplot(df[top_features + ['Professional_Career']], hue='Professional_Career', palette="husl")
pairplot.fig.suptitle("Pairplot of Top Features", y=1.02)
pairplot.savefig(os.path.join(output_dir, "pairplot_top_features.png"))
plt.show()
