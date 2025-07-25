import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io # Keep io for potential future use or if user reverts to string data

def generate_confusion_matrix_heatmap(filepath):
    """
    Generates and displays a confusion matrix heatmap for 'Professional_Career'
    using all original career labels.

    Parameters:
    - filepath (str): The path to the CSV data file.
    """
    # Load the data from the provided filepath
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return
    except Exception as e:
        print(f"Error loading data from '{filepath}': {e}")
        return

    # --- Step 1: Prepare data for modeling (using original Professional_Career) ---
    # Ensure 'Professional_Career' column is clean
    df['Professional_Career'] = df['Professional_Career'].astype(str).str.strip()

    # Features to use for prediction (you can adjust these)
    features = [
        'Age', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
        'University_GPA', 'Internships_Completed', 'Projects_Completed',
        'Certifications', 'Soft_Skills_Score', 'Networking_Score',
        'Job_Offers', 'Starting_Salary', 'Career_Satisfaction',
        'Years_to_Promotion', 'Work_Life_Balance',
    ]
    
    # Ensure all required features exist in the DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in the dataset: {missing_features}")
        print("Please check your feature list or the CSV file.")
        return

    # Encode the target variable (original careers)
    le_target = LabelEncoder()
    df['Professional_Career_enc'] = le_target.fit_transform(df['Professional_Career'])

    X = df[features]
    y = df['Professional_Career_enc']

    # Handle missing values in features by dropping rows
    X = X.dropna()
    y = y[X.index] # Ensure y matches X after dropping NaNs

    # Split data
    if len(X) < 2:
        print("Not enough data to perform train-test split and generate confusion matrix.")
        print("Please provide more diverse sample data with at least two samples per class.")
        return

    # Determine if stratification is possible
    stratify_param = None
    if len(np.unique(y)) > 1: # Only try to stratify if there's more than one class
        # Check if any class has fewer than 2 samples
        min_samples_per_class = y.value_counts().min()
        if min_samples_per_class >= 2:
            stratify_param = y
        else:
            print(f"Warning: Cannot stratify as the least populated class has {min_samples_per_class} sample(s).")
            print("Stratification will be skipped for train_test_split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify_param
    )
    
    print("\n--- Test Set Class Distribution (Actual) ---")
    # Use le_target.inverse_transform to show actual career names
    print(pd.Series(le_target.inverse_transform(y_test)).value_counts().sort_index())

    # --- Step 2: Train a simple classifier ---
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Predicted Class Distribution (on Test Set) ---")
    # Use le_target.inverse_transform to show actual career names
    print(pd.Series(le_target.inverse_transform(y_pred)).value_counts().sort_index())

    # --- Step 3: Generate and display the confusion matrix heatmap ---
    cm = confusion_matrix(y_test, y_pred)
    class_names = le_target.classes_ # Get the names of ALL original classes

    print("\n--- Raw Confusion Matrix ---")
    print(cm)
    print(f"Class labels (order): {class_names}")

    # Create the ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot the confusion matrix with the desired style
    # Adjust figsize and font sizes for a large number of classes
    fig, ax = plt.subplots(figsize=(25, 25)) # Significantly larger figure size
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d',
             text_kw={"fontsize": 6}) # Smaller font size for annotations

    ax.set_title("Confusion Matrix (All Original Careers)", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.xticks(rotation=90, ha='right', fontsize=8) # Rotate and shrink x-axis labels
    plt.yticks(rotation=0, fontsize=8) # Shrink y-axis labels
    plt.tight_layout()
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Specify the path to your actual data file
    data_file_path = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv"
    generate_confusion_matrix_heatmap(data_file_path)
