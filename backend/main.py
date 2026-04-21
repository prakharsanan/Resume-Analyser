from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import fitz
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("../data/skills.json", "r") as f:
    SKILLS_DB = json.load(f)

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.lower()

def extract_skills(resume_text: str) -> dict:
    found_skills = {}
    for category, skills in SKILLS_DB.items():
        matched = []
        for skill in skills:
            if skill in resume_text:
                matched.append(skill)
        if matched:
            found_skills[category] = matched
    return found_skills

def match_job(resume_text: str, job_description: str) -> dict:
    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    resume_words = set(resume_text.split())
    job_words = set(job_description.lower().split())
    missing = job_words - resume_words
    important_missing = [w for w in missing if len(w) > 4][:10]

    return {
        "match_score": round(float(score) * 100, 2),
        "missing_keywords": important_missing
    }

def get_ai_suggestions(resume_text: str, job_description: str, skills: dict) -> str:
    prompt = f"""
You are an expert resume reviewer and career coach.

Here is a candidate's resume (extracted text):
{resume_text[:2000]}

Here is the job description they are applying for:
{job_description}

Skills found on their resume:
{json.dumps(skills, indent=2)}

Please provide:
1. Three specific strengths of this resume for this role
2. Three specific improvements they should make
3. Two skills they should add or highlight better
4. An ATS score out of 100 with brief reasoning

Be specific and actionable. Keep each point to 1-2 sentences.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a professional resume coach. Give honest, specific, actionable advice."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=600,
        temperature=0.7
    )

    return response.choices[0].message.content

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(default="python machine learning data science sql tensorflow")
):
    file_bytes = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)
    skills = extract_skills(resume_text)
    match = match_job(resume_text, job_description)
    suggestions = get_ai_suggestions(resume_text, job_description, skills)

    return {
        "extracted_skills": skills,
        "total_skills_found": sum(len(v) for v in skills.values()),
        "job_match": match,
        "ai_suggestions": suggestions
    }