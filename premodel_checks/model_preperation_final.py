import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Load cleaned data
df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/datasets/cleaned_education_data.csv")

# Convert continuous salary to categorical ranges
bins = [0, 30000, 50000, 70000, 90000, np.inf]
labels = ['<30k', '30-50k', '50-70k', '70-90k', '90k+']
df['Salary_Range'] = pd.cut(df['Starting_Salary'], bins=bins, labels=labels)

# Define features and new target
features = [
    'High_School_GPA',
    'University_GPA',
    'Soft_Skills_Score',
    'Internships_Completed',
    'Field_of_Study',
    'Interest'
]
target = 'Salary_Range'

X = df[features]
y = df[target]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing pipeline
numeric_features = ['High_School_GPA', 'University_GPA', 'Soft_Skills_Score', 'Internships_Completed']
categorical_features = ['Field_of_Study', 'Interest']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Baseline model
dummy = DummyClassifier(strategy="stratified")
dummy.fit(X_train_processed, y_train)
baseline_acc = dummy.score(X_test_processed, y_test)
print(f"Baseline Accuracy: {baseline_acc:.1%}")

# Train Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_processed, y_train)

# Evaluation
print("\nModel Performance:")
print(f"Train Accuracy: {model.score(X_train_processed, y_train):.1%}")
print(f"Test Accuracy: {model.score(X_test_processed, y_test):.1%}")

print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test_processed)))

# Feature importance
feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
importances = pd.Series(model.feature_importances_, index=feature_names)
print("\nTop 10 Features:")
print(importances.sort_values(ascending=False).head(10))

# Final validation
print("\nData Validation:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print("\nClass distribution:")
print(y_train.value_counts(normalize=True))