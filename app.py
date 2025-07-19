import streamlit as st
import os
import docx
import PyPDF2
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import sleep

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_number_from_prompt(prompt):
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    numbers = re.findall(r'\b\d+\b', prompt)
    if numbers:
        return int(numbers[0])
    
    words = re.findall(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', prompt, re.IGNORECASE)
    if words:
        return word_to_num[words[0].lower()]

    return 3  # default if nothing found

def extract_job_description_from_prompt(prompt):
    match = re.search(r'(?:role|position)\s*:\s*(.+)', prompt, re.IGNORECASE)
    if match:
        return match.group(1)
    return prompt

def score_resumes(resume_texts, job_description):
    tfidf = TfidfVectorizer(stop_words='english')
    docs = resume_texts + [job_description]
    tfidf_matrix = tfidf.fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return cosine_sim.flatten()

# Streamlit UI
st.set_page_config(page_title="Resume Ranking Bot", page_icon="📄")
st.title("📄 Resume Ranking Bot")

uploaded_files = st.file_uploader("Upload resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
prompt = st.text_area("Prompt: (e.g., Find the top 3 candidates for this job role: [Paste Job Description])")

if st.button("🔍 Find Top Candidates"):
    if not uploaded_files:
        st.warning("Please upload some resumes first.")
    elif not prompt:
        st.warning("Please enter your prompt including the job description.")
    else:
        with st.spinner("Analyzing resumes..."):
            sleep(1)

            num_candidates = extract_number_from_prompt(prompt)
            job_description = extract_job_description_from_prompt(prompt)

            resume_texts = []
            filenames = []

            for file in uploaded_files:
                try:
                    if file.name.endswith(".pdf"):
                        text = read_pdf(file)
                    elif file.name.endswith(".docx"):
                        text = read_docx(file)
                    else:
                        st.warning(f"{file.name} is not supported.")
                        continue
                    resume_texts.append(text)
                    filenames.append(file.name)
                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}")

            if resume_texts:
                scores = score_resumes(resume_texts, job_description)
                ranked_resumes = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)
                top_resumes = ranked_resumes[:num_candidates]

                st.success(f"Top {num_candidates} Candidates:")

                results = []
                for i, (filename, score) in enumerate(top_resumes, 1):
                    st.write(f"**{i}. {filename}** - Similarity Score: {score:.2f}")
                    results.append({"Rank": i, "Filename": filename, "Score": round(score, 2)})

                df = pd.DataFrame(results)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", csv, "top_candidates.csv", "text/csv")
            else:
                st.error("No readable resumes found.")
