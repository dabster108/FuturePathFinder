import pandas as pd 
import numpy as np

def clean_education_dataset(input_path, output_path):
    df = pd.read_csv("/Users/dikshanta/Documents/Introduction-to-LLM-models/AI_Project/datasets/aligned_education_career_success.csv")
    
    print("Initial Data Overview:")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())
    
    num_cols = ['High_School_GPA', 'SAT_Score', 'University_GPA']
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = ['Field_of_Study', 'Gender', 'Current_Job_Level']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    type_corrections = {
        'Age': 'int8',
        'SAT_Score': 'int16',
        'Starting_Salary': 'float32',
        'Career_Satisfaction': 'int8'
    }
    df = df.astype(type_corrections)
    
    print(f"\nDuplicates Found: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    
    gender_map = {'male':'Male', 'female':'Female', 'other':'Other', 'non-binary':'Other'}
    df['Gender'] = df['Gender'].str.lower().map(gender_map).fillna('Other')
    
    df['Field_of_Study'] = df['Field_of_Study'].str.title()
    
    df['SAT_Score'] = df['SAT_Score'].clip(400, 1600)
    
    gpa_cols = ['High_School_GPA', 'University_GPA']
    df[gpa_cols] = df[gpa_cols].clip(0, 4.0)
    
    field_to_interests = {
        'Computer Science': ['AI', 'Cybersecurity', 'Data Science', 'Software Development', 'Machine Learning'],
    }
    
    df['Interest_Aligned'] = df.apply(
        lambda row: row['Interest'] in field_to_interests.get(row['Field_of_Study'], []),
        axis=1
    )
    
    print("\nCleaning Summary:")
    print(f"Final Shape: {df.shape}")
    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())
    print("\nData Types After Cleaning:")
    print(df.dtypes)
    
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    return df

if __name__ == "__main__":
    input_file = "/Users/dikshanta/Documents/Introduction-to-LLM-models/AI_Project/datasets/aligned_education_career_success.csv"
    output_file = "/Users/dikshanta/Documents/Introduction-to-LLM-models/AI_Project/datasets/cleaned_education_data.csv"
    
    cleaned_data = clean_education_dataset(input_file, output_file)
    print("\nSample of cleaned data:")
    print(cleaned_data.head(3))