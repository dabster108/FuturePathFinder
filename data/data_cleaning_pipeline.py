import pandas as pd
import numpy as np

def clean_final(input_path, output_path):
    df = pd.read_csv(input_path)
    
    print("Initial Data Overview:")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())
    
    # Fill missing numerical values with median
    num_cols = ['High_School_GPA', 'SAT_Score', 'University_GPA']
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fill missing categorical values with mode
    cat_cols = ['Field_of_Study', 'Gender', 'Current_Job_Level', 'Interest']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Type corrections
    type_corrections = {
        'Age': 'int8',
        'SAT_Score': 'int16',
        'Starting_Salary': 'float32',
        'Career_Satisfaction': 'int8'
    }
    for col, dtype in type_corrections.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    
    print(f"\nDuplicates Found: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    
    # Normalize Gender entries if present
    if 'Gender' in df.columns:
        gender_map = {'male':'Male', 'female':'Female', 'other':'Other', 'non-binary':'Other'}
        df['Gender'] = df['Gender'].str.lower().map(gender_map).fillna('Other')
    
    # Title-case for Field_of_Study
    if 'Field_of_Study' in df.columns:
        df['Field_of_Study'] = df['Field_of_Study'].str.title()
    
    # Clip SAT and GPA scores
    if 'SAT_Score' in df.columns:
        df['SAT_Score'] = df['SAT_Score'].clip(400, 1600)
    gpa_cols = ['High_School_GPA', 'University_GPA']
    for col in gpa_cols:
        if col in df.columns:
            df[col] = df[col].clip(0, 4.0)
    
    # Mapping from field of study to possible career interests
    field_to_careers = {
        'Computer Science': ['AI', 'Cybersecurity', 'Data Science', 'Software Development', 'Machine Learning'],
        'Arts': ['Creative Writing', 'Graphic Design', 'Journalism', 'Marketing'],
        'Law': ['Corporate Law', 'Environmental Law', 'Legal Advisor'],
        'Medicine': ['Public Health', 'Medical Researcher', 'Physician'],
        'Engineering': ['Robotics', 'Mechanical Engineer', 'Civil Engineer'],
        'Business': ['Finance', 'Marketing', 'Sales']
    }
    
    # Add Interest_Aligned column (boolean)
    if 'Interest' in df.columns and 'Field_of_Study' in df.columns:
        df['Interest_Aligned'] = df.apply(
            lambda row: row['Interest'] in field_to_careers.get(row['Field_of_Study'], []),
            axis=1
        )
    
        # Add Career column: best-fit career based on field and interest
        def determine_career(row):
            study = row['Field_of_Study']
            interest = row['Interest']
            possible_careers = field_to_careers.get(study, [])
            
            if interest in possible_careers:
                return interest
            elif possible_careers:
                return possible_careers[0]  # default career from field
            else:
                return interest  # fallback
        
        df['Career'] = df.apply(determine_career, axis=1)
    
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
    input_file = "/Users/dikshanta/Documents/FuturePathFinder/datasets/final_csv_file.csv"
    output_file = "/Users/dikshanta/Documents/FuturePathFinder/datasets/cleaned_final_dataset.csv"
    
    cleaned_data = clean_final(input_file, output_file)
    print("\nSample of cleaned data:")
    print(cleaned_data[['Field_of_Study', 'Interest', 'Interest_Aligned', 'Career']].head(5))
