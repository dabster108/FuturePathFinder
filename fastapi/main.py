from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os

# --- Model and Encoder Loading ---
# Ensure these paths are correct relative to where main.py is run
# os.path.dirname(__file__) gets the directory of main.py
# os.path.abspath ensures it's an absolute path
# os.path.join builds paths correctly for different OS
# '../model/career_model.pkl' assumes 'model' folder is one level up from 'fastapi'
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/career_model.pkl'))

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    le_field = model_bundle['le_field']
    le_target = model_bundle['le_target']
    scaler = model_bundle['scaler']
    features = model_bundle['features']
    numeric_features = model_bundle['numeric_features']
    print(f"INFO: Model loaded successfully from {MODEL_PATH}")
    print(f"INFO: Features expected by model: {features}")
    print(f"INFO: Numeric features to scale: {numeric_features}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Please ensure the 'model' directory and 'career_model.pkl' exist.")
    exit(1) # Exit if model is critical and not found
except KeyError as e:
    print(f"ERROR: Missing key in model_bundle: {e}. Ensure career_model.pkl contains 'model', 'le_field', 'le_target', 'scaler', 'features', 'numeric_features'.")
    exit(1)
except Exception as e:
    print(f"ERROR: Failed to load model or its components: {e}")
    exit(1)

# Career descriptions (realistic, extend as needed)
# Moved this out of the global scope if not dynamically loaded, for clarity
CAREER_DESCRIPTIONS = {
    "Full Stack Developer": "Designs and builds both the front-end and back-end of web applications, ensuring seamless integration and user experience.",
    "DevOps Engineer": "Bridges development and operations by automating deployment, monitoring, and infrastructure management.",
    "Frontend Developer": "Specializes in creating visually appealing and interactive user interfaces for web applications.",
    "Backend Developer": "Focuses on server-side logic, databases, and APIs to power web and mobile applications.",
    "Software Engineer": "Applies engineering principles to design, develop, test, and maintain software systems.",
    "QA Engineer": "Ensures the quality and reliability of software through rigorous testing and validation.",
    "Technical Writer": "Creates clear and concise documentation, manuals, and guides for technical products.",
    "IT Support Specialist": "Provides technical assistance and troubleshooting for computer systems and networks.",
    "Data Scientist": "Extracts insights from complex data using statistical analysis, machine learning, and visualization.",
    "Data Analyst": "Analyzes data to help organizations make informed business decisions.",
    "Machine Learning Engineer": "Designs and implements machine learning models and systems for predictive analytics.",
    "Business Intelligence Analyst": "Transforms data into actionable insights to support strategic business decisions.",
    "Research Scientist": "Conducts scientific research to advance knowledge in a specific field.",
    "Statistician": "Applies statistical methods to collect, analyze, and interpret data.",
    "Math Tutor": "Helps students understand and excel in mathematics.",
    "Electrical Engineer": "Designs, develops, and tests electrical systems and equipment.",
    "Mechanical Engineer": "Designs and builds mechanical devices and systems.",
    "Civil Engineer": "Plans, designs, and oversees construction of infrastructure projects.",
    "Quality Engineer": "Ensures products meet quality standards and regulatory requirements.",
    "Engineering Technician": "Assists engineers in the development and testing of products and systems.",
    "Engineer": "Applies scientific and mathematical principles to solve technical problems.",
    "Surgeon": "Performs surgical operations to treat injuries, diseases, and deformities.",
    "Nurse": "Provides patient care, support, and education in healthcare settings.",
    "Physical Therapist": "Helps patients recover mobility and manage pain through physical exercises.",
    "Medical Researcher": "Con`ducts research to improve medical knowledge and patient care.",
    "Physician": "Diagnoses and treats illnesses, injuries, and other health conditions.",
    "Medical Technician": "Operates medical equipment and assists in patient diagnosis and treatment.",
    "Healthcare Assistant": "Supports healthcare professionals in providing patient care.",
    "Legal Consultant": "Advises clients on legal matters and compliance.",
    "Judge Assistant": "Assists judges in legal research and case preparation.",
    "Lawyer": "Represents clients in legal proceedings and provides legal advice.",
    "Paralegal": "Supports lawyers by conducting research and preparing documents.",
    "Legal Researcher": "Conducts research on legal issues and case law.",
    "Legal Clerk": "Performs administrative and clerical tasks in legal settings.",
    "Entrepreneur": "Starts and manages new business ventures, taking on financial risks.",
    "Marketing Manager": "Develops and implements marketing strategies to promote products and services.",
    "Business Analyst": "Analyzes business processes and recommends improvements.",
    "Product Manager": "Oversees the development and lifecycle of products from concept to launch.",
    "Sales Manager": "Leads sales teams and develops strategies to achieve sales targets.",
    "Financial Analyst": "Evaluates financial data to guide investment and business decisions.",
    "Business Associate": "Supports business operations and project management tasks.",
    "Musician": "Performs, composes, or produces music in various genres.",
    "Graphic Designer": "Creates visual content for print and digital media.",
    "Content Creator": "Produces engaging content for online platforms and audiences.",
    "Visual Artist": "Expresses ideas through visual media such as painting, drawing, or sculpture.",
    "Photographer": "Captures images for artistic, commercial, or journalistic purposes.",
    "Art Teacher": "Educates students in various forms of art and creative expression.",
    "Creative Assistant": "Supports creative projects and teams in various tasks.",
    "Generalist": "Possesses a broad range of skills and can adapt to various roles as needed."
}

# Career recommendation explanations (mock)
# Ensure this is created *after* le_target is loaded
CAREER_EXPLANATIONS = {
    name: f"Recommended because your profile matches key skills for {name}. This is based on your field of study, GPA, internships, projects, certifications, and soft skills." for name in le_target.classes_
}


app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "templates")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

class RecommendRequest(BaseModel):
    field_of_study: str
    university_gpa: float
    internships_completed: int
    projects_completed: int
    certifications: int
    soft_skills_score: int

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/fields")
async def get_fields():
    # Return fields only if le_field was loaded successfully
    if 'le_field' in globals():
        return {"fields": list(le_field.classes_)}
    else:
        return JSONResponse(status_code=500, content={"error": "Field encoder not loaded. Model initialization failed."})

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    try:
        # Check if encoders/scalers are loaded
        if not all(k in globals() for k in ['le_field', 'le_target', 'scaler', 'model', 'features', 'numeric_features']):
            raise RuntimeError("Model components not loaded. Server likely started with errors.")

        # Encode field of study
        try:
            field_enc = le_field.transform([req.field_of_study])[0]
        except ValueError:
            # Handle unknown field of study gracefully
            print(f"WARNING: Unknown field of study: {req.field_of_study}. Attempting to proceed or use a default.")
            # Option 1: Use a default/placeholder value (e.g., 0, or average)
            # field_enc = 0
            # Option 2: Raise a more specific error for the client
            return JSONResponse(status_code=422, content={"error": f"Invalid 'Field of Study': '{req.field_of_study}'. Please select from the provided options."})

        # Prepare input in the correct order as expected by 'features' list
        input_dict = {
            'Field_of_Study_enc': field_enc,
            'University_GPA': req.university_gpa,
            'Internships_Completed': req.internships_completed,
            'Projects_Completed': req.projects_completed,
            'Certifications': req.certifications,
            'Soft_Skills_Score': req.soft_skills_score
        }

        # Ensure all expected features are present in input_dict
        # This is a good sanity check if 'features' list gets out of sync with input_dict
        if not all(f in input_dict for f in features):
            missing_features = [f for f in features if f not in input_dict]
            raise ValueError(f"Input data missing expected features: {missing_features}. Expected: {features}")

        # Prepare input for scaler, maintaining feature order
        input_arr = np.array([[input_dict[f] for f in features]])

        # Scale only numeric features (columns matching numeric_features list)
        input_arr_scaled = input_arr.copy()
        for feature_name in numeric_features:
            if feature_name in features: # Ensure numeric feature is actually in the model's features
                idx = features.index(feature_name)
                # Reshape to 1, -1 for single feature scaling if the scaler expects 2D array for a single feature
                # However, for multiple numeric features, scaler.transform(X) expects X to have same number of features as it was trained on
                # The original approach of scaling only specific columns in a pre-arranged array is more robust here.
                # Let's re-verify: if scaler was fit on (GPA, Internships, Projects, Certs, SoftSkills), then extract these, scale, and put back.
                
                # Create a temporary array of *only* numeric features for scaling
                numeric_data_for_scaling = np.array([input_dict[f] for f in numeric_features]).reshape(1, -1)
                scaled_numeric_data = scaler.transform(numeric_data_for_scaling)
                
                # Place scaled numeric data back into the correct positions in input_arr_scaled
                for i, num_feature_name in enumerate(numeric_features):
                    input_arr_scaled[0, features.index(num_feature_name)] = scaled_numeric_data[0, i]
            else:
                print(f"WARNING: Numeric feature '{feature_name}' not found in model's 'features' list.")


        # Predict
        probs = model.predict_proba(input_arr_scaled)[0]
        top_idx = np.argsort(probs)[::-1][:3] # Get top 3 indices
        
        results = []
        for idx in top_idx:
            name = le_target.inverse_transform([idx])[0]
            # Ensure confidence is a float for JSON serialization
            results.append({
                "career": name,
                "confidence": float(probs[idx]) * 100, # Convert to percentage
                "description": CAREER_DESCRIPTIONS.get(name, "No description available for this career."),
                "explanation": CAREER_EXPLANATIONS.get(name, "Explanation not available for this career.")
            })

        print(f"INFO: Successfully generated recommendations: {results}")
        return {"recommendations": results}

    except ValueError as ve:
        print(f"CLIENT ERROR in /recommend: {ve}")
        return JSONResponse(status_code=422, content={"error": f"Invalid input: {str(ve)}"})
    except Exception as e:
        print(f"SERVER ERROR in /recommend: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred on the server. Please try again. Details: {str(e)}"})