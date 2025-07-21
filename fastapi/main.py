from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load model and encoders
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/career_model.pkl'))
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
le_field = model_bundle['le_field']
le_target = model_bundle['le_target']
scaler = model_bundle['scaler']
features = model_bundle['features']
numeric_features = model_bundle['numeric_features']

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
    "Medical Researcher": "Conducts research to improve medical knowledge and patient care.",
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
CAREER_EXPLANATIONS = {
    name: f"Recommended because your profile matches key skills for {name}." for name in le_target.classes_
}

app = FastAPI()

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
def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/fields")
def get_fields():
    return {"fields": list(le_field.classes_)}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    # Encode field of study
    try:
        field_enc = le_field.transform([req.field_of_study])[0]
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid field of study."})
    # Prepare input
    input_arr = np.array([[field_enc, req.university_gpa, req.internships_completed,
                           req.projects_completed, req.certifications, req.soft_skills_score]])
    # Scale numeric features
    input_df = {features[i]: input_arr[0][i] for i in range(len(features))}
    for i, feat in enumerate(features):
        if feat in numeric_features:
            input_df[feat] = float(input_df[feat])
    # Prepare for scaler
    arr_for_scaler = np.array([[input_df[f] for f in features]])
    arr_for_scaler[:, 1:] = scaler.transform(arr_for_scaler[:, 1:])
    # Predict
    probs = model.predict_proba(arr_for_scaler)[0]
    top_idx = np.argsort(probs)[::-1][:5]
    results = []
    for idx in top_idx:
        name = le_target.inverse_transform([idx])[0]
        results.append({
            "career": name,
            "confidence": float(probs[idx]) * 100,
            "description": CAREER_DESCRIPTIONS.get(name, "No description."),
            "explanation": CAREER_EXPLANATIONS.get(name, "No explanation.")
        })
    return {"recommendations": results}