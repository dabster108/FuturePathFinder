# FuturePathFinder: Model Documentation

## Models Overview

The FuturePathFinder application uses **supervised learning** models to provide two key predictions:
1. Career recommendation (classification)
2. Starting salary prediction (regression)

## Model Types and Implementation

### Supervised Learning Approach

Both models implemented in this project are supervised learning algorithms:

1. **Random Forest Classifier** for career prediction
   - Predicts the most suitable career paths based on user's academic and personal attributes
   - Multi-class classification problem
   - Outputs probability scores for different career options

2. **Random Forest Regressor** for salary prediction
   - Predicts expected starting salary based on the same input features
   - Regression problem
   - Includes uncertainty estimation through ensemble variance

### Features Used

The models use a combination of:

**Numeric Features:**
- High School GPA (0.0-4.0)
- University GPA (0.0-4.0)
- SAT Score (400-1600)
- Internships Completed
- Projects Completed
- Certifications

**Categorical Features:**
- Field of Study
- Interest

### Data Processing

- Categorical features are encoded using Label Encoding
- The dataset is split into training (80%) and testing (20%) sets
- No explicit feature scaling is performed as Random Forest models are invariant to monotonic transformations

## Model Performance

### Regression Model (Salary Prediction)
- **Metrics tracked:**
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score

### Classification Model (Career Prediction)
- **Metrics tracked:**
  - Accuracy
  - Classification Report (Precision, Recall, F1-score)

## Prediction Methodology

1. **User Input Collection**
   - Interactive collection of user academic and career preparation details
   - Handling of categorical inputs with fuzzy matching for fields/interests not in the training data

2. **Prediction Process**
   - Encoding of categorical inputs to match training data format
   - Salary prediction with uncertainty estimation (using standard deviation across trees)
   - Career recommendations based on probability scores from the classifier
   - Additional context-aware recommendations based on field and interest alignment

3. **Results Presentation**
   - Top 5 career recommendations with match percentages
   - Salary prediction with uncertainty range
   - Field-aligned career suggestions
   - Similar alumni profiles for reference

## Recommendation Enhancements

Beyond the core models, the system includes supplementary recommendation features:

1. **Field-Aligned Career Matching**
   - Identifies careers commonly associated with the user's field of study
   - Calculates percentage alignment based on historical data

2. **Similar Alumni Profiles**
   - Finds profiles in the dataset that match the user's field and interests
   - Provides real-world examples of career paths and salaries

3. **Fuzzy Matching Algorithm**
   - Handles user inputs that don't exactly match the training data categories
   - Finds the closest match using character overlap similarity

## Future Improvements

Potential enhancements to consider:
- Implementing more sophisticated models (gradient boosting, neural networks)
- Adding feature importance analysis for better explainability
- Incorporating more features related to soft skills and extracurricular activities
- Implementing a more robust uncertainty quantification method
- Adding time-series analysis to track salary trends over time




PA + Field of Study to predict salary?

And use other features (e.g., Interest, Projects, Internships, Certs) for career?



RandomForestRegressor for salary prediction

RandomForestClassifier and KNN for career prediction

Accuracy evaluations

Cross-validation support

Clean structure   