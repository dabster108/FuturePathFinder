import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv")
df['Field_of_Study'] = df['Field_of_Study'].astype(str).str.strip()

le_field = LabelEncoder()
df['Field_of_Study_enc'] = le_field.fit_transform(df['Field_of_Study'])
le_target = LabelEncoder()
df['Professional_Career_enc'] = le_target.fit_transform(df['Professional_Career'].astype(str).str.strip())

features = ['Field_of_Study_enc', 'University_GPA', 'Internships_Completed',
            'Projects_Completed', 'Certifications', 'Soft_Skills_Score']
target = 'Professional_Career_enc'

X = df[features]
y = df[target]

accuracies, precisions, recalls, f1s = [], [], [], []

print("=== Training Random Forest Classifier 10 Times with 80/20 Splits ===\n")

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=i
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

    print(f"Run {i + 1}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}\n")

avg_acc = np.mean(accuracies)
avg_prec = np.mean(precisions)
avg_rec = np.mean(recalls)
avg_f1 = np.mean(f1s)

print("=== Average Performance Across 10 Runs ===")
print(f"Avg Accuracy : {avg_acc:.4f}")
print(f"Avg Precision: {avg_prec:.4f}")
print(f"Avg Recall   : {avg_rec:.4f}")
print(f"Avg F1 Score : {avg_f1:.4f}")

print("\n=== Overall Score ===")
print(f"Overall Accuracy Score (Mean of 10 Runs): {avg_acc:.4f}")
