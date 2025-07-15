import pandas as pd

# Load data
df = pd.read_csv("/Users/dikshanta/Documents/FuturePathFinder/data/education_career_success (2).csv")

# Assign career role based on logic with detailed roles
def assign_career(row):
    field = row['Field_of_Study'].strip().lower()
    projects = row['Projects_Completed']
    certs = row['Certifications']
    internships = row['Internships_Completed']
    soft = row['Soft_Skills_Score']
    entrepreneur = str(row['Entrepreneurship']).strip().lower()

    # ---- Tech/Computer Science (8 roles)
    if 'computer' in field or 'cs' in field or 'software' in field:
        if certs >= 5 and projects >= 6 and soft >= 8:
            return 'Full Stack Developer'
        elif certs >= 4 and projects >= 4 and internships >= 3:
            return 'DevOps Engineer'
        elif projects >= 5 and soft >= 7:
            return 'Frontend Developer'
        elif internships >= 4 and projects >= 2:
            return 'Backend Developer'
        elif certs >= 3 and soft >= 6:
            return 'Software Engineer'
        elif projects >= 3 and internships >= 1:
            return 'QA Engineer'
        elif soft >= 7 and projects < 2:
            return 'Technical Writer'
        else:
            return 'IT Support Specialist'

    # ---- Data / Mathematics (7 roles)
    elif 'math' in field or 'statistics' in field or 'data' in field:
        if certs >= 5 and projects >= 6 and soft >= 7:
            return 'Data Scientist'
        elif certs >= 3 and internships >= 3:
            return 'Data Analyst'
        elif projects >= 5 and soft >= 5:
            return 'Machine Learning Engineer'
        elif internships >= 4 and projects >= 3:
            return 'Business Intelligence Analyst'
        elif certs >= 2 and soft >= 6:
            return 'Research Scientist'
        elif projects >= 2 and internships >= 1:
            return 'Statistician'
        else:
            return 'Math Tutor'

    # ---- Engineering (7 roles)
    elif 'engineer' in field or 'engineering' in field:
        if certs >= 5 and projects >= 5 and internships >= 4:
            return 'Machine Learning Engineer'
        elif certs >= 4 and projects >= 4:
            return 'Electrical Engineer'
        elif certs >= 3 and internships >= 3:
            return 'Mechanical Engineer'
        elif projects >= 3 and soft >= 6:
            return 'Civil Engineer'
        elif internships >= 2 and soft >= 5:
            return 'Quality Engineer'
        elif projects >= 1 and certs >= 1:
            return 'Engineering Technician'
        else:
            return 'Engineer'

    # ---- Medicine (7 roles)
    elif 'medicine' in field or 'medical' in field or 'health' in field:
        if certs >= 5 and projects >= 5:
            return 'Surgeon'
        elif certs >= 4 and internships >= 3:
            return 'Nurse'
        elif projects >= 4 and soft >= 6:
            return 'Physical Therapist'
        elif certs >= 3 and projects >= 3:
            return 'Medical Researcher'
        elif internships >= 2 and soft >= 5:
            return 'Physician'
        elif projects >= 1:
            return 'Medical Technician'
        else:
            return 'Healthcare Assistant'

    # ---- Law (7 roles)
    elif 'law' in field:
        if certs >= 5 and projects >= 4:
            return 'Legal Consultant'
        elif certs >= 4 and internships >= 3:
            return 'Judge Assistant'
        elif certs >= 3:
            return 'Lawyer'
        elif internships >= 2:
            return 'Paralegal'
        elif projects >= 1:
            return 'Legal Researcher'
        else:
            return 'Legal Clerk'

    # ---- Business (7 roles)
    elif 'business' in field or 'management' in field:
        if entrepreneur == 'yes':
            return 'Entrepreneur'
        elif certs >= 5 and projects >= 4:
            return 'Marketing Manager'
        elif certs >= 3 and internships >= 3:
            return 'Business Analyst'
        elif projects >= 4 and soft >= 7:
            return 'Product Manager'
        elif internships >= 2 and soft >= 5:
            return 'Sales Manager'
        elif projects >= 1:
            return 'Financial Analyst'
        else:
            return 'Business Associate'

    # ---- Arts (7 roles)
    elif 'art' in field or 'design' in field:
        if projects >= 6 and soft >= 8:
            return 'Musician'
        elif projects >= 5 and soft >= 7:
            return 'Graphic Designer'
        elif soft >= 8:
            return 'Content Creator'
        elif projects >= 3 and soft >= 6:
            return 'Visual Artist'
        elif internships >= 2:
            return 'Photographer'
        elif soft >= 5:
            return 'Art Teacher'
        else:
            return 'Creative Assistant'

    # ---- Default fallback
    return 'Generalist'


# Apply to DataFrame
df['Professional_Career'] = df.apply(assign_career, axis=1)

# Save to new CSV
df.to_csv("assigned_careers_dataset.csv", index=False)
print("✅ Careers assigned and file saved as 'assigned_careers_dataset.csv'")
