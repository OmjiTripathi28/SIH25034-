import os
import re
import fitz
from docx import Document
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# CONFIG
# ======================
UPLOAD_FOLDER = "uploads"
CSV_FILE = "internship.csv"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
DEFAULT_TOP_N = 5

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ======================
# LOAD CSV
# ======================
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError("internship.csv not found!")

df = pd.read_csv(CSV_FILE).fillna("")
internship_records = df.to_dict(orient="records")

# Build combined text (title + company + location)
def build_text(r):
    return f"{r['internship_title']} {r['company_name']} {r['location']}"

texts = [build_text(r) for r in internship_records]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)

# ======================
# FILE HANDLING
# ======================
def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text") or ""
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_generic(path):
    _, ext = os.path.splitext(path.lower())
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    return ""

# ======================
# SCORING
# ======================
def recommend_from_text(text, n=DEFAULT_TOP_N, location=None):
    if not text.strip():
        return []

    query_vec = vectorizer.transform([text])
    sims = cosine_similarity(query_vec, X).flatten()

    # Boost location if provided
    if location:
        sims = sims + 0.1 * np.array(
            [1 if location.lower() in (r["location"].lower()) else 0 for r in internship_records]
        )

    # Normalize
    if sims.max() > 0:
        sims = sims / sims.max()

    top_idx = np.argsort(-sims)[:n]
    results = []
    for i in top_idx:
        rec = internship_records[i]
        results.append({
            "title": rec["internship_title"],
            "company": rec["company_name"],
            "location": rec["location"],
            "start_date": rec["start_date"],
            "duration": rec["duration"],
            "stipend": rec["stipend"],
            "score": round(float(sims[i]), 4)
        })
    return results

# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Internship recommender running"})

@app.route("/recommend", methods=["POST"])
def recommend_manual():
    data = request.get_json(force=True)

    # Extract fields
    free_text = data.get("free_text", "").strip()
    location = data.get("location", "").strip()
    skills = data.get("skills", [])

    # Fallback if free_text is empty
    if not free_text and skills:
        if isinstance(skills, list):
            free_text = " ".join(skills)
        elif isinstance(skills, str):
            free_text = skills
    if not free_text and location:
        free_text = location

    n = int(data.get("n", DEFAULT_TOP_N))
    recs = recommend_from_text(free_text, n=n, location=location)
    return jsonify({"internships": recs})

@app.route("/recommend_resume", methods=["POST"])
def recommend_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    text = extract_text_generic(filepath)
    if not text.strip():
        return jsonify({"error": "Could not extract text"}), 400

    location = request.form.get("location", "")
    n = int(request.form.get("n", DEFAULT_TOP_N))

    recs = recommend_from_text(text, n=n, location=location)
    return jsonify({"internships": recs})

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
