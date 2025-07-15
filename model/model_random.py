import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# Load data
df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/assigned_careers_dataset.csv")

# Features and target
features = ['Field_of_Study_enc', 'University_GPA', 'Internships_Completed',
            'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
target = 'Professional_Career'

# Encode categorical features
le_field = LabelEncoder()
df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'].astype(str).str.strip())

# Prepare X and y
X = df[features]
y = df[target]

le_target = LabelEncoder()
y_enc = le_target.fit_transform(y.astype(str).str.strip())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("=== Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.3f}")
print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.3f}")
print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.3f}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Helper for validated input
def get_valid_input(prompt, valid_values=None, cast_func=str):
    while True:
        val = input(prompt).strip()
        if cast_func == str:
            val_norm = val.lower()
            valid_norm = [v.lower() for v in valid_values] if valid_values is not None else None
            if valid_norm is not None and val_norm not in valid_norm:
                print(f"Invalid input. Allowed values: {valid_values}")
                continue
            return val  # keep original case
        else:
            try:
                val_cast = cast_func(val)
                if valid_values is not None and val_cast not in valid_values:
                    print(f"Invalid input. Allowed values: {valid_values}")
                else:
                    return val_cast
            except ValueError:
                print(f"Invalid input. Please enter a value of type {cast_func.__name__}.")

# Career prediction from user input
print("\n=== Career Prediction from Input ===")

# Show allowed values for Field_of_Study
print("Available Fields of Study:", list(le_field.classes_))

# Get inputs
field_input = get_valid_input("Field of Study: ", valid_values=le_field.classes_, cast_func=str)
# Normalize capitalization for encoding
field_input_norm = next(v for v in le_field.classes_ if v.lower() == field_input.lower())
field_enc = le_field.transform([field_input_norm])[0]

uni_gpa = get_valid_input("University GPA (e.g. 3.5): ", cast_func=float)
internships = get_valid_input("Internships Completed (integer): ", cast_func=int)
projects = get_valid_input("Projects Completed (integer): ", cast_func=int)
certifications = get_valid_input("Certifications (integer): ", cast_func=int)
soft_skills = get_valid_input("Soft Skills Score (1-10): ", cast_func=int)

# Prepare input DataFrame with proper columns
input_df = pd.DataFrame([{
    'Field_of_Study_enc': field_enc,
    'University_GPA': uni_gpa,
    'Internships_Completed': internships,
    'Projects_Completed': projects,
    'Certifications': certifications,
    'Soft_Skills_Score': soft_skills
}])

# Predict and decode
pred_enc = model.predict(input_df)[0]
pred_career = le_target.inverse_transform([pred_enc])[0]

print(f"\nRecommended Career: {pred_career}")
