import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/datasets/cleaned_education_data.csv")

features = [
    'High_School_GPA',
    'University_GPA',
    'Soft_Skills_Score',
    'Internships_Completed',
    'Field_of_Study',
    'Interest'
]
target = 'Starting_Salary'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=df['Field_of_Study']
)

numeric_features = ['High_School_GPA', 'University_GPA', 'Soft_Skills_Score', 'Internships_Completed']
numeric_transformer = StandardScaler()

categorical_features = ['Field_of_Study', 'Interest']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

def regression_baseline(y_train, y_test):
    mean_pred = np.mean(y_train)
    mae = mean_absolute_error(y_test, [mean_pred]*len(y_test))
    print(f"Baseline MAE (predicting mean): ${mae:,.2f}")
    
    plt.figure(figsize=(10,6))
    sns.kdeplot(y_test, label='True Values')
    plt.axvline(mean_pred, color='red', linestyle='--', label='Baseline Prediction')
    plt.title("Baseline Model: Mean Prediction vs Actual")
    plt.legend()
    plt.show()

def classification_baseline(X_train, y_train, X_test, y_test):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    acc = dummy.score(X_test, y_test)
    print(f"Baseline Accuracy (predicting most frequent class): {acc:.2%}")
    
    plt.figure(figsize=(10,6))
    y_train.value_counts().plot(kind='bar')
    plt.title("Class Distribution for Baseline Reference")
    plt.show()

if y.nunique() > 10:
    regression_baseline(y_train, y_test)
else:
    classification_baseline(X_train_processed, y_train, X_test_processed, y_test)

print("\n=== Final Checks ===")
print(f"Train size: {len(X_train)} samples")
print(f"Test size: {len(X_test)} samples")
print("\nFeature matrix shape:", X_train_processed.shape)
print("Target distribution:")
print(y_train.value_counts(normalize=True))