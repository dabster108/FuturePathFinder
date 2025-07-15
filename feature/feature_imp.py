import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# === 1. Load Dataset ===
df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/assigned_careers_dataset.csv")  # Replace with your actual file path

# Drop unnecessary ID column if present
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])

# === 2. Encode categorical features ===
label_encoders = {}
for column in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# === 3. Split features and target ===
X = df.drop(columns=["Professional_Career"])
y = df["Professional_Career"]

# === 4. Train Random Forest Classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# === 5. Feature Importance (Terminal Output) ===
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n===== Feature Importances =====\n")
print(feature_importance.to_string(index=False))

# === 6. Plot: Feature Importance ===
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette="viridis")
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# === 7. Plot: Correlation Heatmap ===
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
