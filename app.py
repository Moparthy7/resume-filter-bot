import os
import docx
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read text from .docx file
def read_docx(file_path):
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Streamlit UI
st.title("📄 Resume Matcher Chatbot")

st.write("Upload multiple DOCX resumes and paste a job description. We'll rank top candidates for you!")

# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes (.docx)", type="docx", accept_multiple_files=True)

# Paste job description
job_description = st.text_area("Paste Job Description Here")

# Match button
if st.button("Find Top Candidates"):
    if not uploaded_files or not job_description.strip():
        st.warning("Please upload resumes and enter a job description.")
    else:
        resume_texts = []
        resume_names = []
        for file in uploaded_files:
            text = read_docx(file)
            resume_texts.append(text)
            resume_names.append(file.name)

        # TF-IDF matching
        documents = resume_texts + [job_description]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)
        jd_vector = tfidf_matrix[-1]
        resume_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(jd_vector, resume_vectors)[0]

        # Rank resumes
        scored_resumes = list(zip(resume_names, similarities))
        ranked = sorted(scored_resumes, key=lambda x: x[1], reverse=True)

        # Display top 3
        st.subheader("🏆 Top 3 Candidates:")
        for i, (name, score) in enumerate(ranked[:3], 1):
            st.write(f"**{i}. {name}** — Score: `{score:.2f}`")
