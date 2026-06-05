# =============================
# 
# Resume Screening + ATS Scoring 
# =============================

import streamlit as st
import pickle
import re
import nltk
import docx2txt
import PyPDF2
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Resume Screening App", layout="wide")

# ---------------- NLTK SETUP ----------------
@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

load_nltk()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_models():
    clf = pickle.load(open("clf.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    return clf, tfidf

clf, tfidf = load_models()

# ---------------- CLEAN TEXT ----------------
def clean_resume(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9@.+\n ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# ---------------- FILE TEXT EXTRACTION ----------------
def extract_text(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return file.read().decode("utf-8", errors="ignore")

# ---------------- STRUCTURED INFO EXTRACTION ----------------
def extract_required_info(text: str) -> str:
    email = re.search(r"[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"\+?\d[\d\s-]{8,}\d", text)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = lines[0] if lines else "Not Found"

    def grab_section(keywords, limit):
        for key in keywords:
            match = re.search(
                rf"{key}\s*[:\-]?([\s\S]*?)(?=\n[A-Z][A-Za-z ]{{2,}}:|$)",
                text,
                re.I
            )
            if match:
                content = match.group(1)
                content = re.sub(r"\n{2,}", "\n", content)
                clean = [c.strip() for c in content.split("\n") if len(c.strip()) > 2]
                return "\n".join(clean[:limit])
        return "Not Found"

    return f"""
👤 NAME
{name}

📧 EMAIL
{email.group() if email else 'Not Found'}

📱 PHONE
{phone.group() if phone else 'Not Found'}

🛠 SKILLS
{grab_section(['skills','technical skills'], 4)}

📌 PROJECTS
{grab_section(['projects'], 3)}

💼 EXPERIENCE
{grab_section(['experience','internship'], 3)}

🎓 EDUCATION
{grab_section(['education','qualification'], 2)}
""".strip()

# ---------------- ATS SCORE ----------------
def calculate_ats_score(raw_text: str, vector) -> int:
    text = raw_text.lower()
    wc = len(text.split())

    skills = [
        "python","java","react","javascript","sql","html","css",
        "machine learning","data science","django","flask","aws","docker"
    ]

    skill_hits = sum(1 for s in skills if s in text)
    skill_score = min(skill_hits * 4, 30)

    section_score = sum(1 for s in ["skills","project","experience","education"] if s in text) * 6

    length_score = 20 if 180 <= wc <= 900 else 8

    keyword_density = min(np.sum(vector.toarray()) * 8, 20)

    final_score = skill_score + section_score + length_score + keyword_density
    return int(np.clip(final_score, 35, 95))

# ---------------- ROLE MAP ----------------
ROLE_MAP = {
    0: "Advocate", 1: "Arts", 2: "Automation Testing",
    3: "Blockchain", 4: "Business Analyst", 5: "Civil Engineer",
    6: "Data Scientist", 7: "Database", 8: "DevOps Engineer",
    9: "DotNet Developer", 10: "ETL Developer",
    11: "Electrical Engineer", 12: "HR", 13: "Hadoop",
    14: "Health and Fitness", 15: "Java Developer",
    16: "Mechanical Engineer", 17: "Network Security Engineer",
    18: "Operations Manager", 19: "PMO",
    20: "Python Developer", 21: "SAP Developer",
    22: "Sales", 23: "Testing", 24: "Web Designer"
}

# ---------------- MAIN APP ----------------
def main():
    st.title("📄 Smart Resume Screening Application")

    files = st.file_uploader(
        "Upload Resume(s)",
        type=["pdf","docx","txt"],
        accept_multiple_files=True
    )

    if not files:
        st.info("Upload resumes to start screening")
        return

    results = []

    for file in files:
        raw_text = extract_text(file)
        clean_text = clean_resume(raw_text)

        vector = tfidf.transform([clean_text])
        pred = clf.predict(vector)[0]
        ats = calculate_ats_score(raw_text, vector)

        results.append({
            "Resume": file.name,
            "Predicted Role": ROLE_MAP[pred],
            "ATS Score": ats,
            "Details": extract_required_info(raw_text)
        })

    df = pd.DataFrame(results).sort_values("ATS Score", ascending=False)

    st.subheader("📊 Candidate Ranking")
    st.dataframe(df[["Resume","Predicted Role","ATS Score"]], use_container_width=True)

    st.subheader("🔍 Resume Details")
    for r in results:
        with st.expander(f"{r['Resume']} — {r['ATS Score']}%"):
            st.progress(r['ATS Score'] / 100)
            st.text_area("Extracted Info", r["Details"], height=260)


if __name__ == "__main__":
    main()










    
    
# import streamlit as st
# import pickle
# import re
# import nltk
# import docx2txt
# import PyPDF2
# import pandas as pd
# import numpy as np

# # --- 1. PAGE CONFIG & THEME ---
# st.set_page_config(page_title="AI Resume Intelligence", page_icon="🎯", layout="wide")

# # Custom CSS for a Premium Dark Look
# st.markdown("""
#     <style>
#     /* Background and global font */
#     .stApp {
#         background-color: #050505;
#         color: #E0E0E0;
#     }
    
#     /* Neon Title Effect */
#     .main-title {
#         font-size: 45px;
#         font-weight: 800;
#         background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 10px;
#     }
    
#     /* Modern Glassmorphism Card */
#     .res-card {
#         background: rgba(255, 255, 255, 0.03);
#         padding: 25px;
#         border-radius: 15px;
#         border: 1px solid rgba(255, 255, 255, 0.1);
#         margin-bottom: 25px;
#         transition: transform 0.3s ease;
#     }
#     .res-card:hover {
#         border: 1px solid #00d2ff;
#         transform: translateY(-5px);
#     }
    
#     /* Metric styling */
#     [data-testid="stMetricValue"] {
#         color: #00d2ff !important;
#         font-family: 'Courier New', monospace;
#     }
    
#     /* Sidebar styling */
#     section[data-testid="stSidebar"] {
#         background-color: #0a0a0a;
#         border-right: 1px solid #222;
#     }
    
#     /* Buttons */
#     .stButton>button {
#         width: 100%;
#         border-radius: 8px;
#         background: linear-gradient(45deg, #00d2ff, #3a7bd5);
#         color: white;
#         border: none;
#         font-weight: bold;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # --- 2. LOGIC & PROCESSING ---
# @st.cache_resource
# def load_nltk():
#     nltk.download('punkt')
#     nltk.download('punkt_tab')
#     nltk.download('stopwords')

# load_nltk()

# try:
#     clf = pickle.load(open('clf.pkl', 'rb'))
#     tfidf = pickle.load(open('tfidf.pkl', 'rb'))
# except FileNotFoundError:
#     st.error("Error: 'clf.pkl' or 'tfidf.pkl' not found.")

# def cleanResume(txt):
#     txt = re.sub(r"http\S+\s", " ", txt)
#     txt = re.sub(r"RT|cc", " ", txt)
#     txt = re.sub(r"#\S+\s", " ", txt)
#     txt = re.sub(r"@\S+", " ", txt)
#     txt = re.sub(r"[%s]" % re.escape("""!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), " ", txt)
#     txt = re.sub(r"[^\x00-\x7f]", " ", txt)
#     txt = re.sub(r"\s+", " ", txt).strip()
#     return txt

# def extract_text(uploaded_file):
#     file_name = uploaded_file.name.lower()
#     if file_name.endswith(".pdf"):
#         pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
#         return text
#     elif file_name.endswith(".docx"):
#         return docx2txt.process(uploaded_file)
#     else:
#         try: return uploaded_file.read().decode("utf-8")
#         except: return uploaded_file.read().decode("latin-1")

# def calculate_real_ats(resume_text, vector_input):
#     weights = vector_input.toarray().flatten()
#     relevant_terms_count = np.count_nonzero(weights)
#     total_weight = np.sum(weights)
#     word_count = len(resume_text.split())
#     score = 30 
#     keyword_score = (total_weight * 8) + (relevant_terms_count * 0.4)
#     penalty = 15 if word_count < 200 else (10 if word_count > 1000 else 0)
#     final_score = min(max(int(score + keyword_score - penalty), 35), 96)
#     return final_score

# # --- 3. MAIN UI ---
# def main():
#     with st.sidebar:
#         st.image("https://cdn-icons-png.flaticon.com/512/942/942799.png", width=80)
#         st.title("Navigation")
#         st.write("Welcome to the next-gen recruitment tool.")
#         st.info("Upload resumes to see the magic ✨")
#         st.markdown("---")
#         if st.button("Clear Cache"):
#             st.cache_resource.clear()

#     st.markdown("<h1 class='main-title'>Smart Resume Screening ✅</h1>", unsafe_allow_html=True)
#     st.write("Analyze and rank candidates using Advanced ML & Keyword Analysis.")

#     # File uploader with better container
#     with st.container():
#         uploaded_files = st.file_uploader("", type=["pdf", "docx", "txt"], accept_multiple_files=True)

#     if uploaded_files:
#         all_data = []

#         for uploaded_file in uploaded_files:
#             resume_text = extract_text(uploaded_file)
#             cleaned_resume = cleanResume(resume_text)
#             vector_input = tfidf.transform([cleaned_resume])
#             prediction_id = clf.predict(vector_input)[0]

#             category_mapping = {
#                 15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
#                 20: "Python Developer", 24: "Web Designing", 12: "HR",
#                 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
#                 18: "Operation Manager", 6: "Data Science", 22: "Sales",
#                 16: "Mechanical Engineer", 1: "Arts", 7: "Database",
#                 11: "Electrical Engineer", 14: "Health and Fitness",
#                 19: "PMO", 4: "Business Analyst", 9: "Dotnet Developer",
#                 2: "Automation Testing", 17: "Network Security Engineer",
#                 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
#             }
#             category_name = category_mapping.get(prediction_id, "Unknown Role")
#             ats_score = calculate_real_ats(resume_text, vector_input)

#             all_data.append({
#                 "file_name": uploaded_file.name,
#                 "text": resume_text,
#                 "category": category_name,
#                 "score": ats_score
#             })

#         # --- RANKING TABLE ---
#         st.markdown("### 🏆 Candidate Leaderboard")
#         df_summary = pd.DataFrame([
#             {"Candidate": d['file_name'], "Role": d['category'], "ATS Score": d['score']} 
#             for d in all_data
#         ]).sort_values(by="ATS Score", ascending=False)
        
#         # Displaying stylized dataframe
#         st.dataframe(df_summary.style.format({"ATS Score": "{:.0f}%"})
#                      .background_gradient(cmap='Blues', subset=['ATS Score']), 
#                      use_container_width=True)

#         st.markdown("---")
#         st.markdown("### 🔍 In-depth Analysis")

#         # --- CARDS ---
#         for i, data in enumerate(all_data):
#             st.markdown(f"""
#                 <div class="res-card">
#                     <span style="color: #00d2ff; font-size: 0.9rem;">FILE: {data['file_name']}</span>
#                     <h2 style="margin-top: 10px;">{data['category']}</h2>
#                 </div>
#             """, unsafe_allow_html=True)

#             col1, col2, col3 = st.columns([1, 2, 2])
            
#             with col1:
#                 st.metric("ATS MATCH", f"{data['score']}%")
            
#             with col2:
#                 st.write("**Match Confidence**")
#                 st.progress(data['score'] / 100)
            
#             with col3:
#                 if data['score'] < 60:
#                     st.error("Low match: Requires more technical keywords.")
#                 elif data['score'] > 85:
#                     st.success("Strong match: Candidate exceeds baseline criteria.")
#                 else:
#                     st.warning("Average match: Recommend manual review.")

#             with st.expander("📝 View Resume Text"):
#                 st.text_area("", data['text'], height=200, key=f"text_{i}")

# if __name__ == '__main__':
#     main()