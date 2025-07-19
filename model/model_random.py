# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, label_binarize
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     classification_report, confusion_matrix, roc_curve,
#     auc, precision_recall_curve, average_precision_score
# )

# df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv")
# df['Field_of_Study'] = df['Field_of_Study'].astype(str).str.strip()
# le_field = LabelEncoder()
# df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'])
# le_target = LabelEncoder()
# df['Professional_Career_enc'] = le_target.fit_transform(df['Professional_Career'].astype(str).str.strip())
# features = ['Field_of_Study_enc', 'University_GPA', 'Internships_Completed',
#             'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
# target = 'Professional_Career_enc'
# X = df[features]
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)
# print("\n=== Model Evaluation ===")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
# print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.3f}")
# print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.3f}")
# print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.3f}")
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred, target_names=le_target.classes_))
# plt.figure(figsize=(12, 10))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=False, cmap="Blues",
#             xticklabels=le_target.classes_,
#             yticklabels=le_target.classes_)
# plt.title("Confusion Matrix (Heatmap)")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
# y_test_bin = label_binarize(y_test, classes=np.unique(y))
# fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve (Macro-Average)")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.show()
# precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
# pr_auc = average_precision_score(y_test_bin, y_proba, average="macro")
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, color='blue', lw=2, label=f"PR Curve (AUC = {pr_auc:.2f})")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve (Macro-Average)")
# plt.legend(loc="lower left")
# plt.tight_layout()
# plt.show()
# importances = model.feature_importances_
# indices = np.argsort(importances)
# plt.figure(figsize=(8, 6))
# plt.barh(range(len(indices)), importances[indices], color='teal')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.title("Feature Importances (Random Forest)")
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(10, 5))
# df['Professional_Career'].value_counts().plot(kind='bar', color='coral')
# plt.title("Class Distribution: Professional_Career")
# plt.xticks(rotation=90)
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(8, 6))
# sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.tight_layout()
# plt.show()
# for feature in features:
#     plt.figure(figsize=(12, 5))
#     sns.boxplot(x='Professional_Career', y=feature, data=df)
#     plt.title(f"{feature} vs Professional_Career")
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.show()
# print("\n=== Career Prediction from Input ===")
# unique_fields = sorted(df['Field_of_Study'].dropna().unique())
# print("Available Fields of Study:", unique_fields)
# def get_valid_input(prompt, valid_values=None, cast_func=str):
#     while True:
#         val = input(prompt).strip()
#         if cast_func == str:
#             val_norm = val.lower()
#             valid_norm = [v.lower() for v in valid_values] if valid_values is not None else None
#             if valid_norm is not None and val_norm not in valid_norm:
#                 print(f"Invalid input. Allowed values: {valid_values}")
#                 continue
#             return next(v for v in valid_values if v.lower() == val_norm)
#         else:
#             try:
#                 val_cast = cast_func(val)
#                 return val_cast
#             except ValueError:
#                 print(f"Invalid input. Please enter a value of type {cast_func.__name__}.")
# field_input = get_valid_input("Enter your Field of Study: ", valid_values=unique_fields, cast_func=str)
# field_enc = le_field.transform([field_input])[0]
# uni_gpa = get_valid_input("University GPA (e.g. 3.5): ", cast_func=float)
# internships = get_valid_input("Internships Completed (integer): ", cast_func=int)
# projects = get_valid_input("Projects Completed (integer): ", cast_func=int)
# certifications = get_valid_input("Certifications (integer): ", cast_func=int)
# soft_skills = get_valid_input("Soft Skills Score (1-10): ", cast_func=int)
# input_df = pd.DataFrame([{
#     'Field_of_Study_enc': field_enc,
#     'University_GPA': uni_gpa,
#     'Internships_Completed': internships,
#     'Projects_Completed': projects,
#     'Certifications': certifications,
#     'Soft_Skills_Score': soft_skills
# }])
# pred_enc = model.predict(input_df)[0]
# pred_label = le_target.inverse_transform([pred_enc])[0]
# print(f"\n Recommended Career Based on Your Inputs: {pred_label}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve, average_precision_score
)
import joblib


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Field_of_Study'] = df['Field_of_Study'].astype(str).str.strip()
    df['Professional_Career'] = df['Professional_Career'].astype(str).str.strip()

    # Label encode categorical features and target
    le_field = LabelEncoder()
    df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'])

    le_target = LabelEncoder()
    df['Professional_Career_enc'] = le_target.fit_transform(df['Professional_Career'])

    return df, le_field, le_target


def scale_features(X_train, X_test, numeric_features):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train_scaled, X_test_scaled, scaler


def perform_cross_validation(model, X_train, y_train, cv_splits=5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return cv_scores


def evaluate_and_plot(model, X_test, y_test, le_target, features, df):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.3f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.3f}")
    print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.3f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    # Confusion matrix heatmap
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    plt.title("Confusion Matrix (Heatmap)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # ROC Curve (macro-average)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Macro-Average)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
    pr_auc = average_precision_score(y_test_bin, y_proba, average="macro")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Macro-Average)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], color='teal')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

    # Class Distribution
    plt.figure(figsize=(10, 5))
    df['Professional_Career'].value_counts().plot(kind='bar', color='coral')
    plt.title("Class Distribution: Professional_Career")
    plt.xticks(rotation=90)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Feature Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Boxplots for each feature by class
    for feature in features:
        plt.figure(figsize=(12, 5))
        sns.boxplot(x='Professional_Career', y=feature, data=df)
        plt.title(f"{feature} vs Professional_Career")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


def interactive_prediction(model, le_field, le_target):
    print("\n=== Career Prediction from Input ===")
    unique_fields = sorted(le_field.classes_)
    print("Available Fields of Study:", unique_fields)

    def get_valid_input(prompt, valid_values=None, cast_func=str):
        while True:
            val = input(prompt).strip()
            if cast_func == str:
                val_norm = val.lower()
                valid_norm = [v.lower() for v in valid_values] if valid_values is not None else None
                if valid_norm is not None and val_norm not in valid_norm:
                    print(f"Invalid input. Allowed values: {valid_values}")
                    continue
                return next(v for v in valid_values if v.lower() == val_norm)
            else:
                try:
                    val_cast = cast_func(val)
                    return val_cast
                except ValueError:
                    print(f"Invalid input. Please enter a value of type {cast_func.__name__}.")

    field_input = get_valid_input("Enter your Field of Study: ", valid_values=unique_fields, cast_func=str)
    field_enc = le_field.transform([field_input])[0]

    uni_gpa = get_valid_input("University GPA (e.g. 3.5): ", cast_func=float)
    internships = get_valid_input("Internships Completed (integer): ", cast_func=int)
    projects = get_valid_input("Projects Completed (integer): ", cast_func=int)
    certifications = get_valid_input("Certifications (integer): ", cast_func=int)
    soft_skills = get_valid_input("Soft Skills Score (1-10): ", cast_func=int)

    # Prepare input dataframe
    input_df = pd.DataFrame([{
        'Field_of_Study_enc': field_enc,
        'University_GPA': uni_gpa,
        'Internships_Completed': internships,
        'Projects_Completed': projects,
        'Certifications': certifications,
        'Soft_Skills_Score': soft_skills
    }])

    pred_enc = model.predict(input_df)[0]
    pred_label = le_target.inverse_transform([pred_enc])[0]
    print(f"\nRecommended Career Based on Your Inputs: {pred_label}")


def main():
    data_path = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv"

    df, le_field, le_target = load_and_prepare_data(data_path)

    features = ['Field_of_Study_enc', 'University_GPA', 'Internships_Completed',
                'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
    target = 'Professional_Career_enc'

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    # Numeric columns to scale
    numeric_features = ['University_GPA', 'Internships_Completed',
                        'Projects_Completed', 'Certifications', 'Soft_Skills_Score']

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numeric_features)

    # Initialize model
    model = RandomForestClassifier(random_state=42)

    # Cross-validation
    print("Performing 5-fold Stratified Cross-validation...")
    perform_cross_validation(model, X_train_scaled, y_train, cv_splits=5)

    # Train final model on full training data
    model.fit(X_train_scaled, y_train)

    # Save model, encoders, and scaler for FastAPI
    joblib.dump({
        "model": model,
        "le_field": le_field,
        "le_target": le_target,
        "scaler": scaler,
        "features": features,
        "numeric_features": numeric_features
    }, "/Users/dikshanta/Documents/FuturePathFinder/model/career_model.pkl")

    # Evaluate on test set with detailed metrics and plots
    evaluate_and_plot(model, X_test_scaled, y_test, le_target, features, df)

    # Interactive prediction
    interactive_prediction(model, le_field, le_target)


if __name__ == "__main__":
    main()
