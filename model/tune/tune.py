import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Field_of_Study'] = df['Field_of_Study'].astype(str).str.strip()
    df['Professional_Career'] = df['Professional_Career'].astype(str).str.strip()

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

def save_plot(plt, filename, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, filename)
    plt.savefig(full_path, bbox_inches='tight')
    print(f"Saved plot: {full_path}")

def main():
    data_path = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv"
    model_save_path = "/Users/dikshanta/Documents/FuturePathFinder/model/career_model_tuned.pkl"
    tune_vis_folder = "/Users/dikshanta/Documents/FuturePathFinder/model/tune"

    df, le_field, le_target = load_and_prepare_data(data_path)

    features = ['Field_of_Study_enc', 'University_GPA', 'Internships_Completed',
                'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
    target = 'Professional_Career_enc'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    numeric_features = ['University_GPA', 'Internships_Completed',
                        'Projects_Completed', 'Certifications', 'Soft_Skills_Score']

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numeric_features)

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    print("Starting Grid Search for hyperparameter tuning...")
    grid_search.fit(X_train_scaled, y_train)

    print("\nBest Hyperparameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    joblib.dump({
        'model': best_model,
        'le_field': le_field,
        'le_target': le_target,
        'scaler': scaler,
        'features': features,
        'numeric_features': numeric_features
    }, model_save_path)

    print(f"\nBest tuned model saved to {model_save_path}")

    # ========== Visualization Section ===========
    results = pd.DataFrame(grid_search.cv_results_)

    # Convert None to str 'None' for plotting
    results['param_max_depth'] = results['param_max_depth'].apply(lambda x: 'None' if x is None else str(x))

    # Plot mean test score vs max_depth for different n_estimators
    plt.figure(figsize=(10, 6))
    for n_est in sorted(results['param_n_estimators'].unique()):
        subset = results[results['param_n_estimators'] == n_est]
        plt.plot(subset['param_max_depth'], subset['mean_test_score'], marker='o', label=f"n_estimators={n_est}")
    plt.xlabel('max_depth')
    plt.ylabel('Mean CV Accuracy')
    plt.title('Grid Search: max_depth vs Accuracy for different n_estimators')
    plt.legend()
    plt.grid(True)
    save_plot(plt, 'max_depth_vs_accuracy_by_n_estimators.png', tune_vis_folder)
    plt.close()

    # Plot mean test score vs min_samples_split for different min_samples_leaf
    plt.figure(figsize=(10, 6))
    for leaf in sorted(results['param_min_samples_leaf'].unique()):
        subset = results[results['param_min_samples_leaf'] == leaf]
        plt.plot(subset['param_min_samples_split'], subset['mean_test_score'], marker='o', label=f"min_samples_leaf={leaf}")
    plt.xlabel('min_samples_split')
    plt.ylabel('Mean CV Accuracy')
    plt.title('Grid Search: min_samples_split vs Accuracy for different min_samples_leaf')
    plt.legend()
    plt.grid(True)
    save_plot(plt, 'min_samples_split_vs_accuracy_by_min_samples_leaf.png', tune_vis_folder)
    plt.close()

    # Histogram of best parameters frequency (optional)
    best_params_df = pd.DataFrame([grid_search.best_params_])
    print("\nBest parameters summary:")
    print(best_params_df)

if __name__ == "__main__":
    main()
