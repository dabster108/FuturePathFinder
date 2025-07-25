import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

output_dir = "/Users/dikshanta/Documents/FuturePathFinder/feature/feature_visualizations/"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv")

if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define target and drop target + encoded versions from features
target_column = "Professional_Career"
columns_to_drop = [target_column]

for col in df.columns:
    if col != target_column and col.lower().startswith(target_column.lower()):
        columns_to_drop.append(col)

X = df.drop(columns=columns_to_drop)
y = df[target_column]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature importance
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n")
print(feature_importance.to_string(index=False))

print("\nTop Features by Importance:")
for idx, row in feature_importance.iterrows():
    print(f"{idx + 1}) {row['Feature']} - Importance: {row['Importance']:.4f}")

# Plots: Feature importance, correlation heatmap, etc.
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette="viridis")
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Feature Correlation Heatmap (Excludes Target)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x=y, palette="Set2")
plt.title('Distribution of Professional Careers')
plt.xlabel('Professional Career (Encoded)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "career_distribution.png"))
plt.show()

top_features = feature_importance['Feature'].head(4).tolist()
pairplot = sns.pairplot(df[top_features + [target_column]], hue=target_column, palette="husl")
pairplot.fig.suptitle("Pairplot of Top 4 Important Features", y=1.02)
pairplot.savefig(os.path.join(output_dir, "pairplot_top_features.png"))
plt.show()

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
