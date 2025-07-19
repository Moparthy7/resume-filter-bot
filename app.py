import streamlit as st
import os
import docx
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from .docx file
def read_docx(file):
    doc = docx.Document(file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

# Function to extract text from .pdf file
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to convert number words to integers
number_words = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10
}

def extract_number_from_prompt(prompt):
    numbers = re.findall(r'\d+', prompt)
    if numbers:
        return int(numbers[0])
    for word in prompt.lower().split():
        if word in number_words:
            return number_words[word]
    return 3  # Default value if nothing is found

# Streamlit UI
st.title("Resume Filter Chatbot")

uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", accept_multiple_files=True)
job_description = st.text_area("Paste the Job Description")
prompt = st.text_input("What do you want the bot to do?", "Find top 3 candidates")

if uploaded_files and job_description and prompt:
    num_candidates = extract_number_from_prompt(prompt)

    resumes = []
    for file in uploaded_files:
        if file.name.endswith('.docx'):
            text = read_docx(file)
        elif file.name.endswith('.pdf'):
            text = read_pdf(file)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue
        resumes.append((file.name, text))

    corpus = [job_description] + [res[1] for res in resumes]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    results = list(zip([res[0] for res in resumes], scores))
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader(f"Top {num_candidates} Candidate(s):")
    for i in range(min(num_candidates, len(results))):
        st.write(f"{i+1}. {results[i][0]} - Match Score: {results[i][1]:.2f}")
