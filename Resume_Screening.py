import streamlit as st
import pdfplumber
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

resume_data = pd.read_csv('UpdatedResumeDataSet.csv')

tfidf = TfidfVectorizer(max_features=5000)
x = tfidf.fit_transform(resume_data['Resume']).toarray()
y = resume_data.iloc[:, 0]

rf = RandomForestClassifier(n_estimators=300, max_depth=20)
rf.fit(x, y)

# Streamlit UI
st.title("📄 AI-Powered Resume Job Role Predictor")
st.write("Upload your resume (PDF/DOCX), and we'll predict your job role!")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])


def extract_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()


if uploaded_file:
    text = extract_text_from_file(uploaded_file)

    if text:
        text_vectorized = tfidf.transform([text]).toarray()

        predicted_role = rf.predict(text_vectorized)[0]

        st.subheader("Predicted Job Role:")
        st.success(predicted_role)
    else:
        st.error("Could not extract text. Please try another resume.")
