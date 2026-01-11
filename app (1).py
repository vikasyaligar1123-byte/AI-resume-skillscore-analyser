import gradio as gr
import pdfplumber
import docx
import re
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Load model ONCE (global)
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Skill vocabulary (baseline)
# -------------------------------
SKILLS = [
    "python", "java", "c", "c++", "sql",
    "machine learning", "deep learning", "nlp",
    "data analysis", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch",
    "html", "css", "javascript", "react",
    "node", "git", "docker"
]

# -------------------------------
# Resume text extraction
# -------------------------------
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + " "
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------
# Cleaning (for embeddings only)
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------------------
# Skill extraction (DO NOT over-clean)
# -------------------------------
def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS if skill in text]

# -------------------------------
# Semantic similarity
# -------------------------------
def semantic_similarity(resume_text, jd_text):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    return float(util.cos_sim(emb_resume, emb_jd)[0][0])

# -------------------------------
# Score computation
# -------------------------------
def compute_score(similarity, resume_skills, jd_skills):
    if not jd_skills:
        return round(similarity * 100, 2)
    overlap = len(set(resume_skills) & set(jd_skills)) / len(jd_skills)
    return round((0.7 * similarity + 0.3 * overlap) * 100, 2)

# -------------------------------
# Explanation
# -------------------------------
def explain(resume_skills, jd_skills):
    if not jd_skills:
        return {
            "Matched Skills": resume_skills,
            "Missing Skills": ["No skills detected in Job Description"]
        }
    return {
        "Matched Skills": list(set(resume_skills) & set(jd_skills)),
        "Missing Skills": list(set(jd_skills) - set(resume_skills))
    }

# -------------------------------
# Main pipeline (UI calls this)
# -------------------------------
def analyze_resume(file, job_description):
    resume_raw = extract_text(file)
    jd_raw = job_description

    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)

    resume_skills = extract_skills(resume_raw)
    jd_skills = extract_skills(jd_raw)

    similarity = semantic_similarity(resume_clean, jd_clean)
    score = compute_score(similarity, resume_skills, jd_skills)
    explanation = explain(resume_skills, jd_skills)

    return score, explanation

# -------------------------------
# Gradio UI
# -------------------------------
interface = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.File(label="Upload Resume (PDF or DOCX)"),
        gr.Textbox(label="Job Description", lines=10)
    ],
    outputs=[
        gr.Number(label="Match Score (%)"),
        gr.JSON(label="Skill Analysis")
    ],
    title="AI Resume Analyzer",
    description="Semantic resume–job matching with skill gap analysis"
)

# -------------------------------
# Entry point (REQUIRED)
# -------------------------------
interface.launch()
