from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import sys # For sys.exit()

# --- Model and Encoder Loading ---
# Construct the MODEL_PATH dynamically and robustly
# Assuming main.py is in 'fastapi' and model is in 'model' sibling folder
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, '../model/career_model.pkl') # Path is: /Users/dikshanta/Documents/FuturePathFinder/model/career_model.pkl

model_bundle = None
try:
    print(f"INFO: Attempting to load model from: {MODEL_PATH}")
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    le_field = model_bundle['le_field']
    le_target = model_bundle['le_target']
    scaler = model_bundle['scaler']
    features = model_bundle['features'] # List of feature names in the order the model expects
    numeric_features = model_bundle['numeric_features'] # List of numeric feature names for scaling

    # Basic sanity checks after loading
    if not all(k in model_bundle for k in ['model', 'le_field', 'le_target', 'scaler', 'features', 'numeric_features']):
        raise KeyError("One or more required components (model, encoders, scaler, features, numeric_features) are missing from the .pkl bundle.")

    print(f"INFO: Model loaded successfully from {MODEL_PATH}")
    print(f"INFO: Model expects features (order matters): {features}")
    print(f"INFO: Numeric features to be scaled: {numeric_features}")

except FileNotFoundError:
    print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}. Please ensure the 'model' directory and 'career_model.pkl' exist in the correct location relative to main.py.", file=sys.stderr)
    sys.exit(1) # Exit immediately as the app cannot function without the model
except KeyError as e:
    print(f"CRITICAL ERROR: Missing expected component in model_bundle: {e}. The 'career_model.pkl' might be corrupted or incorrectly saved.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: An unexpected error occurred while loading the model or its components: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Career descriptions (realistic, extend as needed)
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
    "Research Scientist": "Con`ducts research to improve medical knowledge and patient care.",
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
    "Legal Researcher": "Con`ducts research on legal issues and case law.",
    "Legal Clerk": "Performs administrative and clerical tasks in legal settings.",
    "Entrepreneur": "Starts and manages new business ventures, taking on financial risks.",
    "Marketing Manager": "Develops and implements marketing strategies to promote products and services.",
    "Business Analyst": "Analyzes business processes and and recommends improvements.",
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
# This is populated *after* le_target is loaded, ensuring it's ready.
CAREER_EXPLANATIONS = {}
if model_bundle: # Only if model_bundle was successfully loaded
    CAREER_EXPLANATIONS = {
        name: f"Recommended because your profile aligns well with the required skills and background for a {name}. This considers your field of study, GPA, internships, projects, certifications, and soft skills score."
        for name in le_target.classes_
    }

app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "templates")), name="static")
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Pydantic model for request body, with validation
class RecommendRequest(BaseModel):
    field_of_study: str = Field(..., min_length=1, max_length=100, description="The user's field of study.")
    university_gpa: float = Field(..., ge=0.0, le=4.0, description="University GPA, typically on a 4.0 scale.")
    internships_completed: int = Field(..., ge=0, le=20, description="Number of internships completed.")
    projects_completed: int = Field(..., ge=0, le=50, description="Number of personal or academic projects completed.")
    certifications: int = Field(..., ge=0, le=20, description="Number of professional certifications obtained.")
    soft_skills_score: int = Field(..., ge=1, le=10, description="A score representing soft skills (e.g., communication, teamwork).")


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/fields")
async def get_fields():
    if model_bundle: # Check if model_bundle (and thus le_field) was loaded successfully
        return {"fields": list(le_field.classes_)}
    else:
        # If model_bundle is None, it means a critical error occurred during startup
        return JSONResponse(status_code=500, content={"error": "Server error: Field of study data not loaded. Please check server logs for critical errors."})

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    print(f"INFO: Received recommendation request: {req.dict()}")

    # Ensure model components are loaded before attempting prediction
    if not model_bundle:
        print("ERROR: Model components are not loaded. Rejecting request.")
        return JSONResponse(status_code=500, content={"error": "Prediction service not ready: Model components failed to load on startup."})

    try:
        # 1. Encode 'Field_of_Study'
        # Check if the field of study exists in the encoder's classes
        if req.field_of_study not in le_field.classes_:
            print(f"WARNING: Unknown field of study received: '{req.field_of_study}'.")
            return JSONResponse(status_code=422, content={"error": f"Invalid 'Field of Study': '{req.field_of_study}'. Please select from the available options."})
        field_enc = le_field.transform([req.field_of_study])[0]

        # 2. Prepare input data dictionary based on expected 'features' order
        # This ensures the input array for the model is always in the correct column order.
        input_data_dict = {
            'Field_of_Study_enc': field_enc,
            'University_GPA': req.university_gpa,
            'Internships_Completed': req.internships_completed,
            'Projects_Completed': req.projects_completed,
            'Certifications': req.certifications,
            'Soft_Skills_Score': req.soft_skills_score
        }

        # Create the raw numpy array in the order of 'features'
        # This is CRITICAL for the model's input
        input_arr_raw = np.array([[input_data_dict[f] for f in features]], dtype=np.float64)

        # 3. Scale numeric features
        input_arr_scaled = input_arr_raw.copy() # Start with a copy to modify
        
        # Identify the columns corresponding to numeric_features in the 'features' list
        numeric_indices_in_features = [features.index(f) for f in numeric_features if f in features]

        if numeric_indices_in_features:
            # Extract only the numeric columns to scale
            numeric_data_to_scale = input_arr_raw[:, numeric_indices_in_features]
            
            # Perform scaling
            scaled_numeric_data = scaler.transform(numeric_data_to_scale)
            
            # Place the scaled data back into the correct positions in input_arr_scaled
            for i, original_idx in enumerate(numeric_indices_in_features):
                input_arr_scaled[0, original_idx] = scaled_numeric_data[0, i]
        else:
            print("WARNING: No numeric features identified for scaling, or numeric_features list is empty/misconfigured.")

        print(f"INFO: Input array for model (after scaling): {input_arr_scaled}")

        # 4. Predict probabilities
        probs = model.predict_proba(input_arr_scaled)[0]

        # 5. Get only the TOP 1 recommendation
        top_idx = np.argsort(probs)[::-1][:1] # Changed to [:1]
        
        results = []
        for idx in top_idx:
            career_name = le_target.inverse_transform([idx])[0]
            confidence_score = float(probs[idx]) * 100 # Convert to percentage

            results.append({
                "career": career_name,
                "confidence": round(confidence_score, 2), # Round confidence for cleaner display
                "description": CAREER_DESCRIPTIONS.get(career_name, "No description available for this career."),
                "explanation": CAREER_EXPLANATIONS.get(career_name, "Explanation not available for this career.")
            })

        print(f"INFO: Successfully generated recommendations: {results}")
        return {"recommendations": results}

    except ValueError as ve:
        # Catches validation errors from Pydantic or our custom checks (e.g., unknown field of study)
        print(f"CLIENT ERROR (422 Unprocessable Entity): {ve}", file=sys.stderr)
        return JSONResponse(status_code=422, content={"error": f"Invalid input: {str(ve)}"})
    except Exception as e:
        # Catch-all for any other unexpected errors during prediction
        print(f"SERVER ERROR (500 Internal Server Error) in /recommend endpoint: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback to server console
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred on the server. Please try again. Technical details: {type(e).__name__} - {str(e)}"})