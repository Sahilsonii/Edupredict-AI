# llm.py - LangChain & Gemini Integration

import os
import sys
import streamlit as st

# Windows compatibility fix: Mock pwd module before importing langchain_community
if sys.platform == 'win32':
    import types
    pwd = types.ModuleType('pwd')
    pwd.getpwnam = lambda x: None
    pwd.getpwuid = lambda x: None
    sys.modules['pwd'] = pwd

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

@st.cache_resource
def build_retriever(csv_path: str, api_key: str):
    """
    Load CSV, create embeddings and FAISS vectorstore, and return a retriever.
    Cached so repeated queries are fast.
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # load docs from csv
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    
    # instantiate embeddings (Hugging Face)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Build FAISS vectorstore from docs + embeddings
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

def get_answer_from_llm(user_question, context, retriever, api_key):
    """
    Get answer from Gemini LLM using logic from app.py
    """
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        return_source_documents=False,
    )

    enhanced_question = f"""
You are a Specialized Academic Data Analyst. 
Your role is to strictly analyze educational and academic datasets (Schools, Universities, Student Performance, Enrollment, etc.).

CONTEXT:
{context}

User Question: {user_question}

INSTRUCTIONS:
1. **DOMAIN CHECK**:
   - If the data or the question is NOT related to the academic/education domain, politely refuse to answer.
   - Say: "I specialize in academic data analysis. Please upload a valid educational dataset."

2. **ACADEMIC ANALYSIS**:
   - Answer the question using ONLY the provided data context.
   - Use academic terminology (e.g., "enrollment trends", "student performance", "faculty ratios").
   - Be precise with numbers and trends.

3. **TONE**:
   - Professional, insightful, and focused on educational outcomes.
"""
    
    answer = chain.run(enhanced_question)
    return answer
