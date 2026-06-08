import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import PyPDF2

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# ================= DOWNLOAD NLTK =================
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */

.stApp {
    background-image:
    linear-gradient(
        rgba(15, 23, 42, 0.82),
        rgba(15, 23, 42, 0.82)
    ),

    url("https://images.unsplash.com/photo-1516321318423-f06f85e504b3");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ===== MAIN CONTAINER ===== */

.main {
    background: transparent;
}

/* ===== TITLES ===== */

.title {
    font-size: 48px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-top: 10px;
    text-shadow: 2px 2px 8px black;
}

.subtitle {
    font-size: 22px;
    color: #e2e8f0;
    text-align: center;
    margin-bottom: 30px;
    text-shadow: 1px 1px 5px black;
}

/* ===== HEADER BOX ===== */

.header-box {
    background: rgba(30, 41, 59, 0.75);
    backdrop-filter: blur(8px);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
    margin-bottom: 25px;
}

/* ===== CARDS ===== */

[data-testid="stVerticalBlock"] > div:has(.element-container) {
    border-radius: 15px;
}

/* ===== SIDEBAR ===== */

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.92);
}

/* ===== TEXT ===== */

h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* ===== INPUT BOX ===== */

.stTextInput input {
    background-color: rgba(255,255,255,0.1);
    color: white;
    border-radius: 10px;
}

/* ===== FILE UPLOADER ===== */

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
}

/* ===== BUTTON ===== */

.stButton button {
    background: linear-gradient(to right, #2563eb, #7c3aed);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

/* ===== FOOTER ===== */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div style="
background: linear-gradient(to right, #0f172a, #1e3a8a);
padding: 25px;
border-radius: 20px;
text-align:center;
">
<h1 style="color:white;">📄 AI Resume Analyzer & Job Fit Predictor</h1>
<p style="color:#cbd5e1;">
Upload your resume and get AI-powered analysis
</p>
</div>
""", unsafe_allow_html=True)



# ================= LOAD DATA =================
# @st.cache_data
# def load_dataset():
#     return pd.read_csv("dataset9/training_data.csv")

# df = load_dataset()

import zipfile
import os

if not os.path.exists("dataset9") and os.path.exists("dataset9.zip"):   #hear ihad changed
    with zipfile.ZipFile("dataset9.zip", "r") as zip_ref:
        zip_ref.extractall(".")

dataset_path = "dataset9"

# ================= CLEAN TEXT =================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'http\\S+', '', text)
    text = re.sub(r'\\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ================= PREPROCESS =================
df['Clean_Resume'] = df['Resume Text'].astype(str).apply(clean_text)

label_encoder = LabelEncoder()

df['Job_Label'] = label_encoder.fit_transform(df['Job Role'])

# ================= TFIDF =================
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['Clean_Resume'])

y = df['Job_Label']

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ================= MODEL =================
@st.cache_resource
def train_model():

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model

model = train_model()

# ================= ACCURACY =================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ================= SIDEBAR =================
st.sidebar.title("📊 Dashboard")

st.sidebar.success(
    f"Model Accuracy: {round(accuracy * 100, 2)}%"
)

st.sidebar.info("AI Resume Screening System")

# ================= FILE UPLOADER =================
uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=['pdf']
)

# ================= PDF TEXT EXTRACTION =================
def extract_text_from_pdf(pdf_file):

    text = ""

    try:

        reader = PyPDF2.PdfReader(pdf_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    except:
        st.error("Error reading PDF file.")

    return text

# ================= SKILLS =================
skills_list = pd.read_csv("dataset9/skills_list.csv")

all_skills = skills_list.iloc[:, 0].dropna().tolist()

# ================= ANALYSIS FUNCTION =================
def analyze_resume(resume_text):

    strengths = []
    weaknesses = []
    suggestions = []

    text = resume_text.lower()

    found_skills = []

    for skill in all_skills:

        if str(skill).lower() in text:
            found_skills.append(skill)

    # Strengths
    if len(found_skills) >= 5:
        strengths.append("Strong technical skill set")

    if 'project' in text:
        strengths.append("Projects included")

    if 'internship' in text:
        strengths.append("Internship experience available")

    # Weaknesses
    if 'github' not in text:
        weaknesses.append("GitHub profile missing")

    if len(found_skills) < 3:
        weaknesses.append("Limited technical skills")

    # Suggestions
    suggestions.append("Customize resume according to job role")
    suggestions.append("Add measurable achievements")
    suggestions.append("Keep resume ATS friendly")

    if 'github' not in text:
        suggestions.append("Add GitHub links")

    # Resume Score
    score = min(50 + len(found_skills) * 5, 100)

    return strengths, weaknesses, suggestions, found_skills, score

# ================= MAIN APP =================
if uploaded_file is not None:

    resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text.strip() == "":
        st.error("No text found in PDF.")
        st.stop()

    clean_resume = clean_text(resume_text)

    # Prediction
    resume_vector = vectorizer.transform([clean_resume])

    prediction = model.predict(resume_vector)

    predicted_role = label_encoder.inverse_transform(prediction)[0]

    # Analysis
    strengths, weaknesses, suggestions, found_skills, score = analyze_resume(resume_text)

    # ATS SCORE
    ats_score = min(score + 10, 100)

    # JOB MATCH
    job_match = min(score + 5, 100)

    # ================= OUTPUT =================
    st.markdown("---")

    col1, col2 = st.columns(2)

    # LEFT
    with col1:

        st.subheader("🎯 Predicted Job Role")

        st.success(predicted_role)

        st.subheader("📈 Resume Score")

        st.progress(score / 100)

        st.info(f"Resume Score: {score}/100")

        st.subheader("🛠 Extracted Skills")

        if found_skills:
            for skill in found_skills:
                st.write(f"✅ {skill}")
        else:
            st.warning("No major skills detected")

    # RIGHT
    with col2:

        st.subheader("✅ Strengths")

        for s in strengths:
            st.success(s)

        st.subheader("❌ Weaknesses")

        for w in weaknesses:
            st.error(w)

        st.subheader("📌 Suggestions")

        for sug in suggestions:
            st.warning(sug)

    # ================= ROLE CARD =================
    st.markdown("---")

    st.markdown(f"""
    <div style="
        background: linear-gradient(to right, #2563eb, #7c3aed);
        padding:25px;
        border-radius:20px;
        text-align:center;
        color:white;
    ">
        <h1>🎯 {predicted_role}</h1>
        <h4>Best Matched Career Role</h4>
    </div>
    """, unsafe_allow_html=True)

    # ================= JOB MATCH =================
    st.subheader("📊 Job Match Percentage")

    st.progress(job_match / 100)

    st.success(f"Job Match: {job_match}%")

    # ================= ATS =================
    st.subheader("🤖 ATS Compatibility Score")

    st.progress(ats_score / 100)

    st.success(f"ATS Score: {ats_score}/100")

    # ================= RESUME TEXT =================
    with st.expander("📄 View Resume Text"):
        st.write(resume_text)

# ================= CHATBOT =================
st.markdown("---")

st.subheader("🤖 Resume Career Assistant")

user_question = st.text_input("Ask your question")

def chatbot_response(question):

    question = question.lower()

    if 'resume' in question:
        return "Add projects, GitHub links, certifications, and measurable achievements."

    elif 'skills' in question:
        return "Add Python, Machine Learning, SQL, NLP, TensorFlow, and Streamlit."

    elif 'project' in question:
        return "Build AI Resume Analyzer, Chatbot, Face Detection, and NLP projects."

    elif 'ats' in question:
        return "Use proper headings and keyword-rich content."

    else:
        return "Please ask career or resume related questions."

if user_question:

    st.success(chatbot_response(user_question))

# ================= FOOTER =================
st.markdown("---")

st.markdown("""
<center>
<h4 style='color:white;'>
AI Resume Analyzer using NLP & Machine Learning
</h4>
<p style='color:gray;'>
Developed using Streamlit, Scikit-learn, NLTK & Python
</p>
</center>
""", unsafe_allow_html=True)
