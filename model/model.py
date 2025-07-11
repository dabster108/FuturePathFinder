import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Load dataset
DATA_PATH = '/Users/dikshanta/Documents/FuturePathFinder/datasets/final_csv_file.csv'

# Features for the models
NUMERIC_FEATURES = ['High_School_GPA', 'University_GPA', 'SAT_Score', 'Internships_Completed',
                    'Projects_Completed', 'Certifications']

CATEGORICAL_FEATURES = ['Field_of_Study', 'Interest']  # Fixed: 'Interest' not 'Interests'

CLASSIFICATION_TARGET = 'Career'
REGRESSION_TARGET = 'Starting_Salary'

def get_user_input():
    """Get user input in a conversational way"""
    print("\n🎓 FUTURE PATH FINDER - CAREER PREDICTION 🎓")
    print("Tell us about yourself to get personalized career recommendations and salary predictions.\n")
    
    user_input = {}
    
    # Academic information
    print("📚 ACADEMIC INFORMATION:")
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
            user_input['University_GPA'] = float(input("University GPA (0.0-4.0): "))
            if 0 <= user_input['University_GPA'] <= 4.0:
                break
            print("GPA must be between 0.0 and 4.0")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            sat = input("SAT Score (400-1600) or press Enter to skip: ")
            if sat == "":
                user_input['SAT_Score'] = 1000  # Default value
                break
            sat_score = float(sat)
            if 400 <= sat_score <= 1600:
                user_input['SAT_Score'] = sat_score
                break
            print("SAT score must be between 400 and 1600")
        except ValueError:
            print("Please enter a valid number.")
    
    # Career preparation
    print("\n💼 CAREER PREPARATION:")
    while True:
        try:
            user_input['Internships_Completed'] = int(input("Number of internships completed: "))
            if user_input['Internships_Completed'] >= 0:
                break
            print("Number must be 0 or higher")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            user_input['Projects_Completed'] = int(input("Number of projects completed: "))
            if user_input['Projects_Completed'] >= 0:
                break
            print("Number must be 0 or higher")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            user_input['Certifications'] = int(input("Number of certifications: "))
            if user_input['Certifications'] >= 0:
                break
            print("Number must be 0 or higher")
        except ValueError:
            print("Please enter a valid number.")
    
    # Field of study and interests (now used in the model)
    print("\n➕ ADDITIONAL INFORMATION:")
    
    # Get field of study
    user_input['Field_of_Study'] = input("Field of Study (e.g., Computer Science, Business, Marketing): ")
    
    # Get interests - Fixed: 'Interest' not 'Interests'
    user_input['Interest'] = input("Interest (e.g., Programming, Finance, Marketing): ")
    
    # Add default values for the removed features that might still be in the dataset
    user_input['Soft_Skills_Score'] = 7  # Default value
    user_input['Networking_Score'] = 7  # Default value
    user_input['University_Ranking'] = 50  # Default value
    
    return user_input

def main():
    df = pd.read_csv(DATA_PATH)

    # Encode categorical target for classification
    le_career = LabelEncoder()
    df['Career_enc'] = le_career.fit_transform(df[CLASSIFICATION_TARGET])
    
    # Encode categorical features
    le_field = LabelEncoder()
    le_interest = LabelEncoder()  # Fixed: 'Interest' not 'Interests'
    
    df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'])
    df['Interest_enc'] = le_interest.fit_transform(df['Interest'])  # Fixed: 'Interest' not 'Interests'
    
    # Store the mapping for later use
    field_mapping = dict(zip(le_field.classes_, le_field.transform(le_field.classes_)))
    interest_mapping = dict(zip(le_interest.classes_, le_interest.transform(le_interest.classes_)))  # Fixed: 'Interest' not 'Interests'

    # Prepare features and targets
    X_numeric = df[NUMERIC_FEATURES]
    X_categorical = df[['Field_of_Study_enc', 'Interest_enc']]  # Fixed: 'Interest' not 'Interests'
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y_reg = df[REGRESSION_TARGET]
    y_clf = df['Career_enc']

    # Train-test split
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42)

    # Train Random Forest Regressor for salary prediction
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train_reg)

    # Train Random Forest Classifier for career prediction
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train_clf)

    # Evaluate Regression
    y_pred_reg = rf_regressor.predict(X_test)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print("Random Forest Regressor Performance on Test Set:")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${np.sqrt(mse):.2f}")
    print(f"R2 Score: {r2:.3f}")
    print()

    # Evaluate Classification
    y_pred_clf = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    
    print("Random Forest Classifier Performance on Test Set:")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print("Classification Report:")
    print(classification_report(y_test_clf, y_pred_clf, target_names=le_career.classes_))
    print()

    # --- Get user input for prediction ---
    user_data = get_user_input()
    
    # Process user input
    input_dict = {}
    for feature in NUMERIC_FEATURES:
        input_dict[feature] = user_data[feature]
    
    # Handle categorical features with encoding
    # If field of study is not in our mapping, use the most similar one
    field = user_data['Field_of_Study']
    if field in field_mapping:
        input_dict['Field_of_Study_enc'] = field_mapping[field]
    else:
        # Find closest match or use a default
        closest_field = find_closest_match(field, field_mapping.keys())
        print(f"\nNote: Field of study '{field}' was matched to '{closest_field}' in our database.")
        input_dict['Field_of_Study_enc'] = field_mapping[closest_field]
    
    # If interest is not in our mapping, use the most similar one
    interest = user_data['Interest']  # Fixed: 'Interest' not 'Interests'
    if interest in interest_mapping:
        input_dict['Interest_enc'] = interest_mapping[interest]  # Fixed: 'Interest' not 'Interests'
    else:
        # Find closest match or use a default
        closest_interest = find_closest_match(interest, interest_mapping.keys())
        print(f"Note: Interest '{interest}' was matched to '{closest_interest}' in our database.")
        input_dict['Interest_enc'] = interest_mapping[closest_interest]  # Fixed: 'Interest' not 'Interests'

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predict salary with Random Forest Regressor
    salary_pred = rf_regressor.predict(input_df)[0]

    # Estimate prediction uncertainty
    all_tree_preds = np.array([tree.predict(input_df)[0] for tree in rf_regressor.estimators_])
    salary_std = all_tree_preds.std()

    # Predict career with Random Forest Classifier
    # Get probabilities for all classes
    career_probs = rf_classifier.predict_proba(input_df)[0]
    
    # Create a list of (career, probability) tuples and sort by probability
    career_prob_pairs = [(le_career.classes_[i], prob) for i, prob in enumerate(career_probs)]
    career_prob_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 5 career recommendations with probability
    top_careers = career_prob_pairs[:5]
    career_recommendations = [(career, f"{prob*100:.0f}% match") for career, prob in top_careers]

    # Find careers that best match the field of study and interests
    field_aligned_careers = get_field_aligned_careers(df, user_data['Field_of_Study'], user_data['Interest'])  # Fixed: 'Interest' not 'Interests'

    # --- Output ---
    print("\n" + "="*60)
    print("🚀 YOUR PERSONALIZED CAREER RECOMMENDATIONS")
    print("="*60)
    
    print(f"\n👤 PROFILE SUMMARY:")
    print(f"• Academic: {user_data['High_School_GPA']:.1f} HS GPA, {user_data['University_GPA']:.1f} Uni GPA")
    print(f"• Experience: {user_data['Internships_Completed']} internships, {user_data['Projects_Completed']} projects, {user_data['Certifications']} certifications")
    print(f"• Field of Study: {user_data['Field_of_Study']}")
    print(f"• Interests: {user_data['Interest']}")  # Fixed: 'Interest' not 'Interests'

    print("\n✅ SALARY PREDICTION:")
    print(f"💰 Predicted Starting Salary: ${salary_pred:,.0f} (±${salary_std:,.0f})")
    print(f"Model Accuracy: {r2*100:.1f}% (R² Score)")

    print("\n✅ CAREER RECOMMENDATIONS:")
    print("🎯 Top Career Matches:")
    for i, (career, match) in enumerate(career_recommendations, 1):
        print(f"{i}. {career} ({match})")
    print(f"Model Accuracy: {accuracy*100:.1f}%")
    
    if field_aligned_careers:
        print("\n🔍 CAREERS ALIGNED WITH YOUR FIELD AND INTERESTS:")
        for i, (career, alignment) in enumerate(field_aligned_careers[:3], 1):
            print(f"{i}. {career} ({alignment}% field alignment)")

    # Get similar alumni profiles based on field of study and interests
    similar_alumni = get_similar_alumni_profiles(df, user_data['Field_of_Study'], user_data['Interest'])  # Fixed: 'Interest' not 'Interests'
    
    print("\n👨‍🎓 SIMILAR ALUMNI PROFILES:")
    for idx, row in similar_alumni.iterrows():
        print(f"• {row['Field_of_Study']} → {row['Career']} (${row['Starting_Salary']:,.0f})")

def find_closest_match(target, options):
    """Find the closest string match using simple character overlap"""
    target = target.lower()
    best_match = None
    best_score = 0
    
    for option in options:
        option_lower = option.lower()
        # Count matching characters
        score = sum(c in option_lower for c in target)
        if score > best_score:
            best_score = score
            best_match = option
    
    # If no good match found, return a default
    if best_score == 0:
        return list(options)[0]  # Return the first option as default
    
    return best_match

def get_field_aligned_careers(df, field, interests):
    """Get careers that align with the given field of study and interests"""
    # Find careers that are commonly associated with this field
    field_careers = df[df['Field_of_Study'].str.lower() == field.lower()]['Career'].value_counts()
    
    # If no exact match, try a more fuzzy match
    if len(field_careers) == 0:
        field_careers = df[df['Field_of_Study'].str.lower().str.contains(field.lower())]['Career'].value_counts()
    
    # If still no match, try with just the first word of the field
    if len(field_careers) == 0 and ' ' in field:
        first_word = field.split(' ')[0].lower()
        field_careers = df[df['Field_of_Study'].str.lower().str.contains(first_word)]['Career'].value_counts()
    
    # Convert to percentage
    total = field_careers.sum() if field_careers.sum() > 0 else 1
    field_careers_percent = (field_careers / total * 100).round().astype(int)
    
    # Convert to list of tuples
    result = [(career, percent) for career, percent in field_careers_percent.items()]
    
    # Sort by percentage (descending)
    result.sort(key=lambda x: x[1], reverse=True)
    
    return result

def get_similar_alumni_profiles(df, field, interests):
    """Get alumni profiles similar to the user's field and interests"""
    # Try to find exact field matches
    field_matches = df[df['Field_of_Study'].str.lower() == field.lower()]
    
    # If no exact matches, try contains
    if len(field_matches) == 0:
        field_matches = df[df['Field_of_Study'].str.lower().str.contains(field.lower())]
    
    # If still no matches, find the closest match
    if len(field_matches) == 0:
        closest_field = find_closest_match(field, df['Field_of_Study'].unique())
        field_matches = df[df['Field_of_Study'] == closest_field]
    
    # If we have more than 5 matches, try to filter by interests
    if len(field_matches) > 5 and interests:
        interest_matches = field_matches[field_matches['Interest'].str.lower().str.contains(interests.lower())]  # Fixed: 'Interest' not 'Interests'
        if len(interest_matches) > 0:
            field_matches = interest_matches
    
    # Return at most 5 records
    return field_matches[['Field_of_Study', 'Career', 'Starting_Salary']].head(5)

if __name__ == '__main__':
    main()
