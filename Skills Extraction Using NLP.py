#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfplumber
import docx
import nltk
import joblib
import re
import spacy
from fuzzywuzzy import fuzz
import fitz 


# In[2]:


resume_data = pd.read_csv('UpdatedResumeDataSet.csv')
# pd.set_option("display.max_rows", None)
resume_data


# In[3]:


tfidf = TfidfVectorizer(max_features=5000)
x = tfidf.fit_transform(resume_data['Resume']).toarray()
y = resume_data.iloc[:, 0] 


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[5]:


rf = RandomForestClassifier(n_estimators=300, max_depth=20)
rf.fit(x_train, y_train)


# In[6]:


rf.predict(x_test)


# In[7]:


TrainingAccuracy = rf.score(x_train,y_train)*100
TestingAccuraccy = rf.score(x_test,y_test)*100
print(f"Training accuracy : {TrainingAccuracy:.2f}")
print(f"Testing accuracy : {TestingAccuraccy:.2f}")


# In[16]:


# Need to modify more 
nlp = spacy.load("en_core_web_sm")

common_skills = {skill.lower() for skill in {
    "Python", "Java", "C++", "C", "JavaScript", "HTML", "CSS", "TypeScript", "Swift", "Kotlin", "Go", "Ruby", "PHP",
    "R", "MATLAB", "Perl", "Rust", "Dart", "Scala", "Shell Scripting",
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js", "Laravel",
    "Bootstrap", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "NLTK", "Pandas", "NumPy",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Firebase", "Cassandra", "Oracle", "Redis", "MariaDB",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform", "CI/CD", "Jenkins", "Git", "GitHub",
    "Cybersecurity", "Penetration Testing", "Ubuntu", "Ethical Hacking", "Firewalls", "Cryptography", "IDS", "Network Security",
    "Machine Learning", "Deep Learning", "Numpy", "Pandas", "Matplotlib", "Computer Vision", "NLP", "Big Data", "Hadoop", "Spark", "Data Analytics",
    "Power BI", "Tableau", "Data Visualization", "Reinforcement Learning",
    "Advanced DSA", "DSA", "Data Structures and Algorithm", "DevOps", "ML", "DL", "Image Processing", "JIRA", "Postman",
    "Excel", "Leadership", "Problem-Solving", "Communication", "Time Management", "Adaptability", "Teamwork",
    "Presentation Skills", "Critical Thinking", "Decision Making", "Public Speaking", "Project Management"
}}

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n" if page.extract_text() else ""
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_sections(text):
    sections = {
        "summary": None,
        "education": None,
        "work_experience": None,
        "projects": None,
        "skills": None,
        "certifications": None,
        "publications": None,
        "competencies": None,
    }

    section_patterns = {
        "summary": r"(summary|profile|about me)[:\n]",
        "education": r"(education|academic background)[:\n]",
        "work_experience": r"(work experience|employment history|professional experience)[:\n]",
        "projects": r"(projects|personal projects|academic projects)[:\n]",
        "skills": r"(skills|technical skills|programming languages)[:\n]",
        "certifications": r"(certifications|courses|training)[:\n]",
        "publications": r"(publications|research papers)[:\n]",
        "competencies": r"(competencies|key competencies|expertise)[:\n]",
    }

    for section, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_idx = match.end()
            next_match = min(
                [m.start() for m in [re.search(p, text[start_idx:], re.IGNORECASE) for p in section_patterns.values()] if m], 
                default=len(text)
            )
            sections[section] = text[start_idx:start_idx + next_match]

    return sections

def extract_skills(text):
    extracted_skills = set()
    doc = nlp(text)

    for token in doc:
        word = token.text.lower()  # Convert token to lowercase
        if word in common_skills:
            extracted_skills.add(word)

    return list(extracted_skills)

def process_resume():
    
    file_path = input("Enter the full path of the resume file (PDF/DOCX): ")
    
    if file_path.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        resume_text = extract_text_from_docx(file_path)
    else:
        print("❌ Unsupported file format. Please provide a PDF or DOCX.")
        return None

    resume_text = resume_text.lower()

    structured_resume = extract_sections(resume_text)
    

    extracted_skills = extract_skills(resume_text)


    print("Extracted Text: \n",structured_resume)
    print("Extracted Skills :\n", extracted_skills)
    
    text_vectorized = tfidf.transform([resume_text]).toarray()

    predicted_role = rf.predict(text_vectorized)[0]

    print("\n🔍 Predicted Job Role:", predicted_role)


# In[10]:


process_resume()


# In[11]:


process_resume()


# In[12]:


process_resume()


# In[13]:


process_resume()


# In[14]:


process_resume()


# In[15]:


process_resume()


# In[ ]:




