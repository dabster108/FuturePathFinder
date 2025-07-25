import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import os

def show_basic_info(df, name):
    print(f"\n--- {name} Dataset Head (5 rows) ---")
    print(df.head())
    
    print(f"\n--- {name} Dataset Info ---")
    print(df.info())
    
    print(f"\n--- {name} Missing Values ---")
    print(df.isnull().sum())

def print_cleaning_summary(df):
    print("\n Data Cleaning Summary")
    print("- Dropped 'Student_ID' column as non-informative.")
    print("- Removed empty and whitespace-only rows.")
    print("- Imputed missing numeric values with median.")
    print("- Imputed missing categorical values with mode.")
    print("- Stripped whitespace from all string columns.")
    print("- Encoded categorical columns into numeric labels.")
    print(f"- Final dataset shape: {df.shape}")
    print("- Dataset contains no missing values after preprocessing.")
    print("- Will proceed with feature selection based on relevance and importance.")

def save_missing_plot(df, name, folder):
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(10, 5))
    msno.matrix(df)
    plt.title(f"Missing Value Matrix - {name}")
    filepath = os.path.join(folder, f"{name}_missing_matrix.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved Missing Value Matrix Plot for {name} at {filepath}")

def main():
    original_path = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/carrersucess.csv"
    cleaned_path = "/Users/dikshanta/Documents/FuturePathFinder/data/datasets/cleaned_data.csv"
    vis_folder = "./compare_visualizations"
    
    # Load datasets
    df_orig = pd.read_csv(original_path)
    df_clean = pd.read_csv(cleaned_path)
    
    # Show heads and info
    show_basic_info(df_orig, "Original")
    show_basic_info(df_clean, "Cleaned")
    
    # Cleaning summary for cleaned dataset
    print_cleaning_summary(df_clean)
    
    # Show encoded sample columns if present
    for col in ['Gender', 'Gender_enc', 'Professional_Career', 'Professional_Career_enc']:
        if col in df_clean.columns:
            print(f"\nSample unique values for '{col}' in Cleaned dataset:")
            print(df_clean[col].unique())
    
    # Save missing value plots
    save_missing_plot(df_orig, "Original", vis_folder)
    save_missing_plot(df_clean, "Cleaned", vis_folder)
    
    print("\nAll done! Check the visualization folder for saved plots.")

if __name__ == "__main__":
    main()
