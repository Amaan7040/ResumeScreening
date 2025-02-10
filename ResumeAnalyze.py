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


# In[8]:


def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        return None
    
    return text.strip() if text.strip() else None
def predict_job_role():
    file_path = input("Enter the full path of the resume file (PDF/DOCX): ").strip()
    
    text = extract_text_from_file(file_path)
    print(text)
    if text is None:
        print("❌ Unsupported file format or empty file. Please use PDF or DOCX.")
        return

    text_vectorized = tfidf.transform([text]).toarray()

    predicted_role = rf.predict(text_vectorized)[0]

    print("\n🔍 Predicted Job Role:", predicted_role)


# In[9]:


predict_job_role()


# In[10]:


predict_job_role()


# In[11]:


predict_job_role()


# In[ ]:




