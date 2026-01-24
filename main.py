# main.py - Application Entry Point

import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="EduPredict AI - Refactored", page_icon="📊", layout="wide")

# Import UI components
from app.ui.sidebar import render_sidebar
from app.ui.dashboard import render_dashboard

def load_api_key():
    # Try getting from env var first
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if api_key:
        return api_key
        
    # Fallback to secrets.json if .env fails or for backward compat
    if os.path.exists("secrets.json"):
        with open("secrets.json", 'r') as f:
            secrets = json.load(f)
            return secrets.get('GEMINI_API_KEY') or secrets.get('GOOGLE_API_KEY')
    return None

def main():
    st.title("EduPredict AI")
    
    api_key = load_api_key()
    if not api_key:
        st.error("❌ API key not found. Please set GEMINI_API_KEY in .env or secrets.json")
    
    # File Uploader
    uploaded_file = st.file_uploader("📁 Upload ANY CSV File", type="csv")
    
    if uploaded_file:
        # Save temp file for tools that need path
        csv_path = f"data/raw/{uploaded_file.name}"
        os.makedirs("data/raw", exist_ok=True)
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        df = pd.read_csv(csv_path)
        
        # Render Sidebar
        df_modified, csv_analysis = render_sidebar(df)
        
        # Render Main Dashboard
        render_dashboard(df_modified, csv_analysis, api_key, csv_path)
            
    else:
        st.info("👆 Upload ANY CSV file to start.")
        st.markdown("""
        ### Feature Support:
        - **Modular Architecture**: Clean separation of UI and Logic
        - **Advanced Imputation**: Handle missing data
        - **Interactive Visualizations**: Power BI style charts
        - **ML Forecasts**: Batch prediction and analysis
        - **AI Q&A**: Ask questions about your data
        """)

if __name__ == "__main__":
    main()
