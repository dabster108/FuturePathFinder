import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                             accuracy_score, classification_report, confusion_matrix)
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

DATA_PATH = '/Users/dikshanta/Documents/FuturePathFinder/datasets/cleaned_final_dataset.csv'
MODEL_DIR = '/Users/dikshanta/Documents/FuturePathFinder/model/saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    'High_School_GPA', 'University_GPA', 'Internships_Completed',
    'Projects_Completed', 'Certifications', 'University_Ranking', 'Soft_Skills_Score',
    'Age', 'SAT_Score', 'Networking_Score', 'Job_Offers'
]
CATEGORICAL_FEATURES = ['Field_of_Study', 'Interest', 'Gender', 'Current_Job_Level']
CLASSIFICATION_TARGET = 'Career'
REGRESSION_TARGET = 'Starting_Salary'

def get_user_input():
    print("\nFUTURE PATH FINDER - CAREER PREDICTION 🎓")
    print("Tell us about yourself to get personalized career recommendations and salary predictions.\n")
    
    user_input = {}

    print("👤 PERSONAL INFORMATION:")
    while True:
        try:
            user_input['Age'] = int(input("Age: "))
            if 16 <= user_input['Age'] <= 80:
                break
            print("Please enter a realistic age.")
        except ValueError:
            print("Please enter a valid number.")
    
    user_input['Gender'] = input("Gender (Male, Female, Other): ")


    print("\n📚 ACADEMIC INFORMATION:")
    while True:
        try:
            user_input['High_School_GPA'] = float(input("High School GPA (0.0-4.0): "))
            if 0 <= user_input['High_School_GPA'] <= 4.0:
                break
            print("GPA must be between 0.0 and 4.0")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            user_input['SAT_Score'] = int(input("SAT Score (e.g., 400-1600): "))
            if 400 <= user_input['SAT_Score'] <= 1600:
                break
            print("SAT score must be between 400 and 1600.")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            user_input['University_GPA'] = float(input("University GPA (0.0-4.0): "))
            if 0 <= user_input['University_GPA'] <= 4.0:
                break
            print("GPA must be between 0.0 and 4.0")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            user_input['University_Ranking'] = int(input("University Ranking (e.g., 1-500): "))
            if user_input['University_Ranking'] > 0:
                break
            print("Ranking must be a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    print("\n💼 CAREER PREPARATION:")
    for key, label in [
        ('Internships_Completed', "Number of internships completed"),
        ('Projects_Completed', "Number of projects completed"),
        ('Certifications', "Number of certifications"),
        ('Job_Offers', "Number of job offers received")
    ]:
        while True:
            try:
                val = int(input(f"{label}: "))
                if val >= 0:
                    user_input[key] = val
                    break
                print("Value must be 0 or higher")
            except ValueError:
                print("Please enter a valid number.")

    print("\n➕ ADDITIONAL INFORMATION:")
    user_input['Field_of_Study'] = input("Field of Study (e.g., Computer Science, Business): ")
    user_input['Interest'] = input("Interest (e.g., Programming, Finance): ")
    user_input['Current_Job_Level'] = input("Current or Desired Job Level (e.g., Entry, Mid, Senior): ")


    while True:
        try:
            user_input['Soft_Skills_Score'] = int(input("Soft Skills Score (1-10): "))
            if 1 <= user_input['Soft_Skills_Score'] <= 10:
                break
            print("Score must be between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")
            
    while True:
        try:
            user_input['Networking_Score'] = int(input("Networking Skills Score (1-10): "))
            if 1 <= user_input['Networking_Score'] <= 10:
                break
            print("Score must be between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")

    return user_input

def find_closest_match(target, options):
    target = target.lower()
    best_match = None
    best_score = 0
    for option in options:
        score = sum(c in option.lower() for c in target)
        if score > best_score:
            best_score = score
            best_match = option
    return best_match if best_match else list(options)[0]

def get_field_aligned_careers(df, field, interests):
    field_careers = df[df['Field_of_Study'].str.lower() == field.lower()]['Career'].value_counts()
    if len(field_careers) == 0:
        field_careers = df[df['Field_of_Study'].str.lower().str.contains(field.lower())]['Career'].value_counts()
    if len(field_careers) == 0 and ' ' in field:
        first_word = field.split(' ')[0].lower()
        field_careers = df[df['Field_of_Study'].str.lower().str.contains(first_word)]['Career'].value_counts()
    total = field_careers.sum() if field_careers.sum() > 0 else 1
    return sorted([(career, int(round((count / total) * 100))) for career, count in field_careers.items()], key=lambda x: x[1], reverse=True)

def get_similar_alumni_profiles(df, field, interests):
    matches = df[df['Field_of_Study'].str.lower() == field.lower()]
    if matches.empty:
        matches = df[df['Field_of_Study'].str.lower().str.contains(field.lower())]
    if matches.empty:
        closest = find_closest_match(field, df['Field_of_Study'].unique())
        matches = df[df['Field_of_Study'] == closest]
    if len(matches) > 5 and interests:
        filtered = matches[matches['Interest'].str.lower().str.contains(interests.lower())]
        if not filtered.empty:
            matches = filtered
    return matches[['Field_of_Study', 'Career', 'Starting_Salary']].head(5)

def train_and_evaluate(df):
    print("Starting model training and evaluation...")

    le_career = LabelEncoder()
    le_field = LabelEncoder()
    le_interest = LabelEncoder()
    le_gender = LabelEncoder()
    le_job_level = LabelEncoder()

    df['Career_enc'] = le_career.fit_transform(df[CLASSIFICATION_TARGET])
    df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'])
    df['Interest_enc'] = le_interest.fit_transform(df['Interest'])
    df['Gender_enc'] = le_gender.fit_transform(df['Gender'])
    df['Current_Job_Level_enc'] = le_job_level.fit_transform(df['Current_Job_Level'])

    encoders = {
        'career': le_career,
        'field': le_field,
        'interest': le_interest,
        'gender': le_gender,
        'job_level': le_job_level
    }

    X_numeric = df[NUMERIC_FEATURES]
    X_categorical = df[['Field_of_Study_enc', 'Interest_enc', 'Gender_enc', 'Current_Job_Level_enc']]
    
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y_reg = df[REGRESSION_TARGET]
    y_clf = df['Career_enc']

    # Split data into training and testing sets
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    # --- Train and Evaluate Classifier ---
    print("\n--- Training Career Classifier ---")
    scaler_clf = StandardScaler()
    X_train_clf_scaled = scaler_clf.fit_transform(X_train)
    X_test_clf_scaled = scaler_clf.transform(X_test)

    clf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [2, 4],
        'class_weight': ['balanced']
    }
    rf_classifier = RandomForestClassifier(random_state=42)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_clf = GridSearchCV(rf_classifier, clf_param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
    grid_clf.fit(X_train_clf_scaled, y_train_clf)
    best_classifier = grid_clf.best_estimator_

    y_pred_clf_train = best_classifier.predict(X_train_clf_scaled)
    y_pred_clf_test = best_classifier.predict(X_test_clf_scaled)

    print("Classifier Best Params:", grid_clf.best_params_)
    print(f"Classifier Accuracy: {accuracy_score(y_test_clf, y_pred_clf_test) * 100:.1f}%")

    # --- Train and Evaluate Regressor ---
    print("\n--- Training Salary Regressor ---")
    
    # Add true career to training data for regressor
    X_train_reg = X_train.copy()
    X_train_reg['Career_enc'] = y_train_clf

    # Add predicted career to test data for regressor
    X_test_reg = X_test.copy()
    X_test_reg['Career_enc'] = y_pred_clf_test

    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)

    reg_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_leaf': [2, 4],
        'min_samples_split': [5, 10]
    }
    rf_regressor = RandomForestRegressor(random_state=42)
    grid_reg = GridSearchCV(rf_regressor, reg_param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_reg.fit(X_train_reg_scaled, y_train_reg)
    best_regressor = grid_reg.best_estimator_

    y_pred_reg = best_regressor.predict(X_test_reg_scaled)
    print("Regressor Best Params:", grid_reg.best_params_)
    print(f"Regressor R2 Score: {r2_score(y_test_reg, y_pred_reg):.3f}")
    print(f"Regressor MAE: ${mean_absolute_error(y_test_reg, y_pred_reg):.2f}")

    # --- Post-training analysis ---
    print("\nClassification Report:")
    print(classification_report(y_test_clf, y_pred_clf_test, target_names=le_career.classes_))

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test_clf, y_pred_clf_test, labels=le_career.transform(le_career.classes_))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_career.classes_, yticklabels=le_career.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    print(f"\nConfusion matrix saved to {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")

    model_assets = {
        'regressor': best_regressor,
        'classifier': best_classifier,
        'encoders': encoders,
        'scaler_clf': scaler_clf,
        'scaler_reg': scaler_reg,
        'clf_columns': X.columns.tolist(),
        'reg_columns': X_train_reg.columns.tolist()
    }
    return model_assets

def predict_user_input(model_assets, user_data):
    input_dict = {feature: user_data.get(feature, 0) for feature in NUMERIC_FEATURES}
    
    encoders = model_assets['encoders']
    le_field = encoders['field']
    le_interest = encoders['interest']
    le_gender = encoders['gender']
    le_job_level = encoders['job_level']
    le_career = encoders['career']

    field_mapping = dict(zip(le_field.classes_, le_field.transform(le_field.classes_)))
    interest_mapping = dict(zip(le_interest.classes_, le_interest.transform(le_interest.classes_)))
    gender_mapping = dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))
    job_level_mapping = dict(zip(le_job_level.classes_, le_job_level.transform(le_job_level.classes_)))

    closest_field = find_closest_match(user_data['Field_of_Study'], field_mapping.keys())
    if closest_field != user_data['Field_of_Study']:
        print(f"Note: Field '{user_data['Field_of_Study']}' matched to '{closest_field}'")
    input_dict['Field_of_Study_enc'] = field_mapping[closest_field]

    closest_interest = find_closest_match(user_data['Interest'], interest_mapping.keys())
    if closest_interest != user_data['Interest']:
        print(f"Note: Interest '{user_data['Interest']}' matched to '{closest_interest}'")
    input_dict['Interest_enc'] = interest_mapping[closest_interest]

    closest_gender = find_closest_match(user_data.get('Gender', ''), gender_mapping.keys())
    input_dict['Gender_enc'] = gender_mapping[closest_gender]

    closest_job_level = find_closest_match(user_data.get('Current_Job_Level', ''), job_level_mapping.keys())
    input_dict['Current_Job_Level_enc'] = job_level_mapping[closest_job_level]

    # Predict Career
    classifier = model_assets['classifier']
    scaler_clf = model_assets['scaler_clf']
    clf_columns = model_assets['clf_columns']
    
    input_clf_df = pd.DataFrame([input_dict])
    input_clf_df = input_clf_df.reindex(columns=clf_columns, fill_value=0)
    input_clf_scaled = scaler_clf.transform(input_clf_df)
    
    probas = classifier.predict_proba(input_clf_scaled)[0]
    top_careers_indices = probas.argsort()[-5:][::-1]
    top_careers = [(le_career.classes_[i], f"{100*probas[i]:.0f}% match") for i in top_careers_indices]
    
    predicted_career_enc = classifier.predict(input_clf_scaled)[0]
    input_dict['Career_enc'] = predicted_career_enc

    # Predict Salary
    regressor = model_assets['regressor']
    scaler_reg = model_assets['scaler_reg']
    reg_columns = model_assets['reg_columns']

    input_reg_df = pd.DataFrame([input_dict])
    input_reg_df = input_reg_df.reindex(columns=reg_columns, fill_value=0)
    input_reg_scaled = scaler_reg.transform(input_reg_df)

    salary_pred = regressor.predict(input_reg_scaled)[0]
    salary_std = np.std([tree.predict(input_reg_scaled)[0] for tree in regressor.estimators_])

    return salary_pred, salary_std, top_careers

def main():
    df = pd.read_csv(DATA_PATH)
    
    print("Training new models...")
    model_assets = train_and_evaluate(df)

    user_data = get_user_input()
    
    salary_pred, salary_std, top_careers = predict_user_input(model_assets, user_data)

    aligned = get_field_aligned_careers(df, user_data['Field_of_Study'], user_data['Interest'])
    alumni = get_similar_alumni_profiles(df, user_data['Field_of_Study'], user_data['Interest'])

    print("\n" + "="*60)
    print("YOUR PERSONALIZED CAREER RECOMMENDATIONS")
    print("="*60)
    print(f"\nPROFILE SUMMARY:")
    print(f"Academic: {user_data['High_School_GPA']} HS GPA, {user_data['University_GPA']} Uni GPA")
    print(f"Experience: {user_data['Internships_Completed']} internships, "
          f"{user_data['Projects_Completed']} projects, {user_data['Certifications']} certifications")
    print(f"Field of Study: {user_data['Field_of_Study']}")
    print(f"Interests: {user_data['Interest']}")

    print(f"\nSALARY PREDICTION:")
    print(f"Predicted Starting Salary: ${salary_pred:,.0f} (±${salary_std:,.0f})")

    print(f"\nCAREER RECOMMENDATIONS:")
    for i, (career, match) in enumerate(top_careers, 1):
        print(f"{i}. {career} ({match})")

    if aligned:
        print("\nCAREERS ALIGNED WITH YOUR FIELD AND INTERESTS:")
        for i, (career, score) in enumerate(aligned[:3], 1):
            print(f"{i}. {career} ({score}% field alignment)")

    if not alumni.empty:
        print("\nSIMILAR ALUMNI PROFILES:")
        for _, row in alumni.iterrows():
            print(f"• {row['Field_of_Study']} → {row['Career']} (${row['Starting_Salary']:,.0f})")

if __name__ == '__main__':
    main()
