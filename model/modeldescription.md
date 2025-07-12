Model Summary: Salary Prediction and Career Recommendation
This document explains the models used for predicting Starting Salary and recommending Career Fields based on a student's academic and skill profile. Two different machine learning algorithms were used, each selected based on the nature of the target variable and expected model behavior.

1. Objective
The primary objectives of this system are:

To predict the likely starting salary of a student using regression.

To recommend potential career fields that align with the student's background and interests using classification.

2. Input Features Used
The following input features are used across both models:

Feature Name	Description
High_School_GPA	GPA obtained in high school (0.0 – 4.0 scale)
University_GPA	GPA obtained at the university level
SAT_Score	Standardized test score (400 – 1600)
University_Ranking	Global ranking of the student's university
Internships_Completed	Number of internships completed
Projects_Completed	Number of academic or personal projects
Certifications	Number of completed certifications
Soft_Skills_Score	Score (1–10) indicating soft skills proficiency
Networking_Score	Score (1–10) indicating networking ability
Course (Field_of_Study)	The student's academic field (e.g., Computer Science)
Career_Interest	The student’s stated career interest (e.g., AI)

3. Models Used
A. Salary Prediction – Random Forest Regressor
Goal: Predict a numerical value for the student's expected starting salary.

Algorithm: Random Forest Regressor

Why Random Forest:

Handles non-linear relationships between features and salary.

Naturally captures feature interactions.

Provides robust predictions even with correlated inputs.

Offers built-in feature importance scoring for interpretability.

Target Variable:

Starting_Salary (a continuous numeric value)

Training Configuration:

Train-Test Split: 80% training, 20% testing

Evaluation Metrics: R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE)

B. Career Recommendation – K-Nearest Neighbors (KNN)
Goal: Recommend suitable career fields for the student based on their profile.

Algorithm: K-Nearest Neighbors (KNN) Classifier

Why KNN:

Simple and interpretable algorithm that works well for small-to-medium datasets.

Learns by comparing new input data to historical examples.

Provides a natural way to identify similar alumni for career insight.

Useful in providing personalized recommendations by observing trends among similar students.

Target Variable:

Career (a categorical label indicating the chosen or likely career path)

Training Configuration:

Train-Test Split: 80% training, 20% testing

Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

4. Why Different Algorithms Were Used
The prediction tasks differ in nature:

Salary Prediction requires a regression model because the output is continuous.

Random Forest is effective at capturing non-linear trends without extensive preprocessing.

Career Recommendation is a classification task, aiming to match the user to the most likely career.

KNN is ideal here because it relies on similarity, which aligns with the recommendation nature of the task.

Using the same algorithm for both tasks would not be optimal, as each problem has different structural and mathematical requirements.

5. Sample Output
Given Input:

python
Copy
Edit
{
    'High_School_GPA': 3.8,
    'University_GPA': 3.9,
    'SAT_Score': 1450,
    'Internships_Completed': 2,
    'Projects_Completed': 5,
    'Certifications': 2,
    'Soft_Skills_Score': 8,
    'Networking_Score': 6,
    'Course': 'Computer Science',
    'Career_Interest': 'AI',
    'University_Ranking': 150
}
Model Output:

Random Forest Prediction: Starting Salary: $72,500 (± $5,200)

KNN Recommendation:

Recommended Career Fields:

Data Science (87% match)

Machine Learning (76% match)

Similar Alumni:

CS Grad → ML Engineer @ Google ($85,000)

CS Grad → Data Analyst @ Meta ($76,000)





--- differences if asked to write -- 
Algorithm Used: Random Forest Regressor
Random Forest is an ensemble learning method that combines multiple decision trees to provide more stable and accurate predictions.

✅ Why Random Forest?
Handles both linear and non-linear relationships well.

Automatically captures interactions between features.

Reduces variance via ensemble averaging (low overfitting).

Robust to missing or noisy data.

Provides feature importance, aiding in explainability.

❌ Alternatives and Why Not Used:
Algorithm	Reason Not Chosen
Linear Regression	Assumes linearity, not flexible for complex real-world educational data
SVR (Support Vector Regression)	Sensitive to feature scaling, slow on large datasets
KNN Regressor	Struggles with high-dimensional data and does not generalize; inefficient
XGBoost/LightGBM	Powerful, but more complex tuning required; RF is simpler for early modeling

2. Career Recommendation (Classification)
Algorithm Used: K-Nearest Neighbors (KNN) Classifier
KNN is a lazy-learning, non-parametric algorithm that classifies based on proximity to training instances.

✅ Why KNN?
Provides recommendations based on similar students, which aligns with user expectations.

Easy to interpret and explain: “You are similar to X, who became a Y.”

Returns example-based predictions, not just a class label.

Great for prototype-based reasoning (matching to alumni profiles).

❌ Alternatives and Why Not Used:
Algorithm	Reason Not Chosen
Random Forest Classifier	Good accuracy, but doesn't return similar examples; lacks personalization
Logistic Regression	Too simplistic; linear boundaries not enough for career pathways
SVM Classifier	High complexity, less interpretable, slower in large datasets
Naive Bayes	Assumes feature independence, unrealistic for education/career data