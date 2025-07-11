import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_docx(file_path):
    import docx
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Load the resume text
resume_text = read_docx("sample_resume.docx")

# Define a sample job description
job_description = """
We are looking for a Software Engineer who is proficient in Python, has experience with machine learning,
and is comfortable working in a dynamic team environment. The candidate should be able to develop robust code,
analyze data, and collaborate cross-functionally.
"""

# Prepare our documents for comparison: resume and job description
documents = [resume_text, job_description]

# Convert texts into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute cosine similarity between the two documents
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

print(f"Similarity Score between resume and job description: {similarity:.2f}")
