import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import plot_tree
import joblib

#DATA LOADING & ENCODING
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Field_of_Study'] = df['Field_of_Study'].astype(str).str.strip()
    df['Professional_Career'] = df['Professional_Career'].astype(str).str.strip()
    le_field = LabelEncoder()
    df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'])
    le_target = LabelEncoder()
    df['Professional_Career_enc'] = le_target.fit_transform(df['Professional_Career'])
    return df, le_field, le_target

#  FEATURE SCALING
def scale_features(X_train, X_test, numeric_features):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    return X_train_scaled, X_test_scaled, scaler

# CROSS-VALIDATION PHASE
def perform_cross_validation(model, X_train, y_train, cv_splits=5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return cv_scores

def save_plot(plt, filename, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, filename)
    plt.savefig(full_path, bbox_inches='tight')
    print(f"Saved {filename} successfully.")

# TESTING PHASE
def evaluate_and_plot(model, X_test, y_test, le_target, features, df):
    model_vis_folder = os.path.join(os.path.dirname(__file__), 'model_visualizations')
    os.makedirs(model_vis_folder, exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.3f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.3f}")
    print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.3f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    cm = confusion_matrix(y_test, y_pred)

    # --- UPDATED: Confusion Matrix (Numbers Only) ---
    plt.figure(figsize=(12, 10))
    # Create an axes object
    ax = plt.gca() 
    # Use imshow to create a grid, but with a transparent colormap or 'None'
    # Using 'white' colormap and no interpolation to just show the grid for text placement
    ax.imshow(np.zeros_like(cm), cmap='Greys', interpolation='nearest', vmin=0, vmax=1) 
    
    # Manually place text for each cell
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', fontsize=10, color='black')

    ax.set_xticks(np.arange(len(le_target.classes_)))
    ax.set_yticks(np.arange(len(le_target.classes_)))
    ax.set_xticklabels(le_target.classes_, rotation=90)
    ax.set_yticklabels(le_target.classes_)
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    save_plot(plt, 'confusion_matrix_text_only.png', model_vis_folder)
    plt.show()
    # --- END UPDATED SECTION ---

    # Original Confusion Matrix (Heatmap) - Kept as requested to not change others
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    plt.title("Confusion Matrix (Heatmap)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_plot(plt, 'confusion_matrix_heatmap.png', model_vis_folder)
    plt.show()

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
    save_plot(plt, 'roc_curve_macro.png', model_vis_folder)
    plt.show()
    
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
    pr_auc = average_precision_score(y_test_bin, y_proba, average="macro")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Macro-Average)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    save_plot(plt, 'precision_recall_curve_macro.png', model_vis_folder)
    plt.show()
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], color='teal')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    save_plot(plt, 'feature_importances.png', model_vis_folder)
    plt.show()
    plt.figure(figsize=(10, 5))
    df['Professional_Career'].value_counts().plot(kind='bar', color='coral')
    plt.title("Class Distribution: Professional_Career")
    plt.xticks(rotation=90)
    plt.ylabel("Count")
    plt.tight_layout()
    save_plot(plt, 'class_distribution.png', model_vis_folder)
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    save_plot(plt, 'feature_correlation_heatmap.png', model_vis_folder)
    plt.show()
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features))
    for i, feature in enumerate(features):
        ax = axes[i] if n_features > 1 else axes
        sns.boxplot(x='Professional_Career', y=feature, data=df, ax=ax)
        ax.set_title(f"{feature} vs Professional_Career")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    save_plot(plt, 'combined_boxplot.png', model_vis_folder)
    plt.show()
    try:
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(df[features])
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=df['Professional_Career_enc'], cmap='tab10', alpha=0.7)
        plt.title('t-SNE Feature Space')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter, ticks=range(len(le_target.classes_)), label='Career')
        plt.tight_layout()
        save_plot(plt, 'tsne_feature_space.png', model_vis_folder)
        plt.show()
    except Exception as e:
        print(f"t-SNE visualization failed: {e}")

#INTERACTIVE INPUT PREDICTION

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

def plot_training_curves(cv_scores, output_folder):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', label='Cross-Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Cross-Validation Folds')
    plt.legend()
    save_plot(plt, 'accuracy_training_curve.png', output_folder)
    plt.show()

def plot_decision_boundaries(model, X, y, features, le_target, output_folder):
    if X.shape[1] != 2:
        print("Decision boundaries can only be plotted for 2D feature spaces.")
        return
    plt.figure(figsize=(8, 6))
    DecisionBoundaryDisplay.from_estimator(model, X, response_method='predict', alpha=0.5, cmap='coolwarm')
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Decision Boundaries')
    plt.colorbar(scatter, ticks=range(len(le_target.classes_)), label='Career')
    save_plot(plt, 'decision_boundaries.png', output_folder)
    plt.show()

def plot_bar_pie_charts(df, target_column, output_folder):
    import matplotlib.ticker as mtick
    sns.set(style='whitegrid')
    value_counts = df[target_column].value_counts().sort_index()
    total = value_counts.sum()
    plt.figure(figsize=(12, 6))
    bars = plt.bar(value_counts.index.astype(str), value_counts.values, color='skyblue', edgecolor='black')
    plt.title('Bar Chart of Class Distribution', fontsize=14)
    plt.xlabel(target_column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + total * 0.01,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(plt, 'bar_chart_class_distribution.png', output_folder)
    plt.show()
    plt.figure(figsize=(10, 10))
    labels = [f"{label} ({count / total * 100:.1f}%)" for label, count in value_counts.items()]
    wedges, texts = plt.pie(
        value_counts,
        colors=plt.cm.Paired.colors[:len(labels)],
        startangle=140,
        radius=1.2
    )
    plt.legend(wedges, labels, title=target_column, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
    plt.title('Pie Chart of Class Distribution', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    save_plot(plt, 'pie_chart_class_distribution.png', output_folder)
    plt.show()

def visualize_one_decision_tree(model, feature_names, class_names, output_folder, tree_index=0):
    plt.figure(figsize=(16, 10))
    estimator = model.estimators_[tree_index]
    plot_tree(estimator,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=8,
              max_depth=3)
    plt.title(f"Decision Tree #{tree_index + 1} from Random Forest")
    plt.tight_layout()
    save_plot(plt, f'decision_tree_{tree_index + 1}.png', output_folder)
    plt.show()

#MODEL TRAINING AND TESTING PIPELINE

def main():
    data_path = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv"
    df, le_field, le_target = load_and_prepare_data(data_path)
    features = ['Field_of_Study_enc', 'University_GPA', 'Internships_Completed',
                'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
    target = 'Professional_Career_enc'
    X = df[features]
    y = df[target]
     # Feature-Target Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
     # Scaling
    numeric_features = ['University_GPA', 'Internships_Completed',
                        'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numeric_features)
      # Model Initialization
    model = RandomForestClassifier(random_state=42)
    print("Performing 5-fold Stratified Cross-validation...")
    cv_scores = perform_cross_validation(model, X_train_scaled, y_train, cv_splits=5)
    model_vis_folder = os.path.join(os.path.dirname(__file__), 'model_visualizations')
    plot_training_curves(cv_scores, model_vis_folder)
    model.fit(X_train_scaled, y_train)
    joblib.dump({
        "model": model,
        "le_field": le_field,
        "le_target": le_target,
        "scaler": scaler,
        "features": features,
        "numeric_features": numeric_features
    }, "/Users/dikshanta/Documents/FuturePathFinder/model/career_model.pkl")
    evaluate_and_plot(model, X_test_scaled, y_test, le_target, features, df)
    visualize_one_decision_tree(model, features, le_target.classes_, model_vis_folder)
    if len(features) == 2:
        plot_decision_boundaries(model, X_test_scaled, y_test, features, le_target, model_vis_folder)
    plot_bar_pie_charts(df, 'Professional_Career', model_vis_folder)
    interactive_prediction(model, le_field, le_target)
    
if __name__ == "__main__":
    main()






