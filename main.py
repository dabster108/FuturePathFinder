import pandas as pd

def add_career_column(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Mapping from field of study to possible careers
    field_to_careers = {
        'Computer Science': ['Machine Learning', 'Data Science', 'Software Engineer', 'Cybersecurity'],
        'Arts': ['Creative Writing', 'Graphic Design', 'Journalism', 'Marketing'],
        'Law': ['Corporate Law', 'Environmental Law', 'Legal Advisor'],
        'Medicine': ['Public Health', 'Medical Researcher', 'Physician'],
        'Engineering': ['Robotics', 'Mechanical Engineer', 'Civil Engineer'],
        'Business': ['Finance', 'Marketing', 'Sales']
    }
    
    def determine_career(row):
        study = row['Field_of_Study']
        interest = row['Interest']
        possible_careers = field_to_careers.get(study, [])
        
        if interest in possible_careers:
            return interest
        elif possible_careers:
            return possible_careers[0]
        else:
            return interest  # fallback if no mapping available
    
    df['Career'] = df.apply(determine_career, axis=1)
    
    df.to_csv(output_path, index=False)
    print(f"Saved updated dataset with 'Career' column to: {output_path}")
    print(df[['Field_of_Study', 'Interest', 'Career']].head(10))

if __name__ == "__main__":
    input_file = "/Users/dikshanta/Documents/FuturePathFinder/datasets/cleaned_education_data.csv"
    output_file = "/Users/dikshanta/Documents/FuturePathFinder/datasets/final_csv_file.csv"
    
    add_career_column(input_file, output_file)
