import os
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read text from .docx file
def read_docx(file_path):
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Folder containing resumes
resume_folder = "resumes"
resume_texts = []
resume_names = []

# Read each resume
for file_name in os.listdir(resume_folder):
    if file_name.endswith(".docx"):
        full_path = os.path.join(resume_folder, file_name)
        text = read_docx(full_path)
        resume_texts.append(text)
        resume_names.append(file_name)

# Job description input
job_description = input("Paste the job description:\n")

# Combine all resume texts + JD
documents = resume_texts + [job_description]

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# Compare JD (last index) to each resume
jd_vector = tfidf_matrix[-1]
resume_vectors = tfidf_matrix[:-1]

similarities = cosine_similarity(jd_vector, resume_vectors)[0]

# Create list of (resume_name, score)
scored_resumes = list(zip(resume_names, similarities))

# Sort by score (descending)
ranked_resumes = sorted(scored_resumes, key=lambda x: x[1], reverse=True)

# Print top 3
print("\n📋 Top 3 Candidates:")
for i, (name, score) in enumerate(ranked_resumes[:3], start=1):
    print(f"{i}. {name} - Similarity Score: {score:.2f}")
