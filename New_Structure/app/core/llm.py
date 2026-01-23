# llm.py - LangChain & Gemini Integration

import os
import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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
    
    # instantiate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Build FAISS vectorstore from docs + embeddings
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

def get_answer_from_llm(user_question, context, retriever, api_key):
    """
    Get answer from Gemini LLM using logic from app.py
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
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
You are a universal data analyst that can understand and analyze ANY type of CSV dataset. 
You adapt your analysis approach based on the data structure and user question.

UNIVERSAL DATASET CONTEXT:
{context}

User Question: {user_question}

ANALYSIS INSTRUCTIONS:
- This CSV could contain any type of data (sales, students, products, financial, etc.)
- Analyze based on the actual column names and data patterns shown
- For counting questions: count the actual relevant rows
- For statistical questions: calculate from the provided data
- For pattern questions: identify trends in the actual data structure
- Be precise and show your reasoning process
- Adapt your language to match the domain of the data
"""
    
    answer = chain.run(enhanced_question)
    return answer
