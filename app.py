# app.py - Universal CSV Analyzer with Enhanced ML Predictions & LangChain with Gemini AI

import streamlit as st
import pandas as pd
import os
import json
import datetime
import re
import asyncio
from pathlib import Path
from missing_value_handler import (
    show_missing_summary, 
    iterative_impute, 
    advanced_iterative_impute, 
    visualize_missing,
    compare_imputation,
    get_imputation_stats
)

from visualization import visualizations_sidebar
from langchain_google_genai import ChatGoogleGenerativeAI

# --- LangChain / Gemini imports ---
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import predictions

# Ensure an event loop exists for Streamlit's thread
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Universal CSV Analyzer", page_icon="📊", layout="wide")

@st.cache_data
def load_api_key():
    secrets_file = Path("secrets.json")
    if secrets_file.exists():
        with open(secrets_file, 'r') as f:
            secrets = json.load(f)
            # Prefers Gemini key with fallback to Google key
            return secrets.get('GEMINI_API_KEY') or secrets.get('GOOGLE_API_KEY')
    return None

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

@st.cache_data
def process_csv_with_predictions(uploaded_file_name, df_csv):
    """
    Process CSV with clustering and batch predictions with enhanced debugging.
    """
    try:
        # Step 1: Apply intelligent clustering to all columns
        df_processed, cluster_mappings = predictions.cluster_all_columns(
            df_csv,
            numeric_bins=8,
            cat_top_k=15,
            treat_years=True,
            keep_original=True
        )
        
        # Step 2: Run batch predictions with explicit column detection
        time_cols = predictions.detect_time_columns(df_processed)
        numeric_cols = predictions.detect_numerical_columns(df_processed)
        
        # Debug information
        st.info(f"🔍 Debug: Found {len(time_cols)} time columns: {time_cols}")
        st.info(f"🔍 Debug: Found {len(numeric_cols)} numeric columns: {numeric_cols}")
        
        # Only proceed if we have both time and numeric columns
        if not time_cols:
            st.warning("⚠️ No time-related columns detected in the dataset")
            return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns detected in the dataset")  
            return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}
        
        # Remove time columns from target columns to avoid self-forecasting
        target_cols = [col for col in numeric_cols if col not in time_cols]
        
        if not target_cols:
            st.warning("⚠️ No valid target columns (numeric columns that aren't time columns)")
            return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}
        
        st.info(f"🎯 Processing {len(time_cols)} time columns and {len(target_cols)} target columns")
        
        # Run batch predictions - FIXED: batch_results is directly a dictionary of forecasts
        batch_results = predictions.batch_forecast_backend(
            df_processed,
            potential_time_cols=time_cols,
            target_cols=target_cols
        )
        
        # Step 3: Create summary - FIXED: Process batch_results directly
        successful_predictions = []
        successful_forecasts = []
        
        # Handle case where batch_results might be an error dict
        if isinstance(batch_results, dict) and batch_results.get('status') == 'error':
            st.error(f"❌ Batch processing error: {batch_results.get('error', 'Unknown error')}")
            return df_csv, cluster_mappings, batch_results, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}
        
        # Process successful forecasts
        for key, result in batch_results.items():
            if isinstance(result, dict) and result.get('status') == 'ok':
                if '__by__' in key:
                    successful_forecasts.append(key)
                else:
                    successful_predictions.append(key)
        
        prediction_summary = {
            "ml_predictions": successful_predictions,
            "time_forecasts": successful_forecasts,
            "total_successful": len(successful_predictions) + len(successful_forecasts)
        }
        
        # Debug output
        st.info(f"✅ Generated {prediction_summary['total_successful']} successful predictions")
        
        return df_processed, cluster_mappings, batch_results, prediction_summary
        
    except Exception as e:
        st.error(f"❌ Prediction processing failed: {str(e)}")
        st.exception(e)  # This will show the full traceback for debugging
        return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}

def debug_predictions(df, batch_results, prediction_summary):
    """Debug function to understand why predictions fail"""
    st.write("### 🔍 Prediction Debug Information")
    
    # Check original data
    st.write("**Original DataFrame Info:**")
    st.write(f"- Shape: {df.shape}")
    st.write(f"- Columns: {list(df.columns)}")
    st.write(f"- Data types: {dict(df.dtypes)}")
    
    # Check what columns are detected
    time_cols = predictions.detect_time_columns(df)
    numeric_cols = predictions.detect_numerical_columns(df)
    
    st.write("**Column Detection:**")
    st.write(f"- Time columns detected: {time_cols}")
    st.write(f"- Numeric columns detected: {numeric_cols}")
    
    # Check batch results
    st.write("**Batch Results Summary:**")
    if isinstance(batch_results, dict):
        st.write(f"- Total forecast attempts: {len(batch_results)}")
        
        status_counts = {}
        for key, result in batch_results.items():
            if isinstance(result, dict):
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
                
                if status != 'ok':
                    st.write(f"- {key}: {status} - {result.get('reason', result.get('error', 'No details'))}")
            else:
                st.write(f"- {key}: Invalid result type - {type(result)}")
        
        st.write(f"- Status counts: {status_counts}")
    else:
        st.write(f"- Unexpected batch_results type: {type(batch_results)}")
        st.write(f"- Content: {batch_results}")

def analyze_csv_structure(df):
    analysis = {
        'numeric_cols': list(df.select_dtypes(include=['int64', 'float64']).columns),
        'text_cols': list(df.select_dtypes(include=['object']).columns),
        'date_cols': [],
        'categorical_cols': [],
        'key_patterns': {}
    }
    for col in df.columns:
        if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'period']):
            analysis['date_cols'].append(col)
    for col in analysis['text_cols']:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5 and df[col].nunique() < 100:
            analysis['categorical_cols'].append(col)
    for col in df.columns:
        col_lower = col.lower()
        sample_values = df[col].dropna().astype(str).head(10).tolist()
        analysis['key_patterns'][col] = {
            'column_name': col,
            'sample_values': sample_values,
            'unique_count': df[col].nunique(),
            'has_numbers': any(char.isdigit() for char in col_lower),
            'keywords': [word for word in col_lower.split('_') if len(word) > 2]
        }
    return analysis

def extract_question_keywords(question):
    """Extract meaningful keywords from user question"""
    question_lower = question.lower()
    
    patterns = {
        'count_words': ['how many', 'count', 'total', 'number of', 'sum'],
        'time_words': ['year', 'month', 'date', 'time', 'period', 'when'],
        'comparison_words': ['highest', 'lowest', 'most', 'least', 'compare', 'versus'],
        'grouping_words': ['by', 'group', 'category', 'type', 'kind'],
        'statistical_words': ['average', 'mean', 'median', 'min', 'max', 'range']
    }
    
    found_patterns = {}
    for pattern_type, words in patterns.items():
        found_patterns[pattern_type] = [word for word in words if word in question_lower]
    
    numbers = re.findall(r'\b\d{4}\b|\b\d+\.?\d*\b', question)
    quoted_terms = re.findall(r'"([^"]*)"', question) + re.findall(r"'([^']*)'", question)
    
    return {
        'patterns': found_patterns,
        'numbers': numbers,
        'quoted_terms': quoted_terms,
        'question_words': question_lower.split()
    }

def find_relevant_data_smart(df, question, csv_analysis):
    """Universal function to find relevant data based on ANY question and ANY CSV structure"""
    
    question_analysis = extract_question_keywords(question)
    relevant_data = df.copy()
    filters_applied = []
    relevance_score = {}
    
    # Score columns by relevance to question
    for col, col_info in csv_analysis['key_patterns'].items():
        score = 0
        col_lower = col.lower()
        
        # Check if column name matches question keywords
        for word in question_analysis['question_words']:
            if word in col_lower or word in ' '.join(col_info['keywords']):
                score += 10
        
        # Check if sample values match question terms
        for value in col_info['sample_values']:
            for word in question_analysis['question_words']:
                if word in str(value).lower():
                    score += 5
        
        # Bonus for specific quoted terms
        for term in question_analysis['quoted_terms']:
            if term.lower() in col_lower or any(term.lower() in str(val).lower() for val in col_info['sample_values']):
                score += 20
        
        relevance_score[col] = score
    
    # Filter by specific values mentioned in question
    for col in df.columns:
        col_data = df[col].astype(str)
        
        # Filter by numbers mentioned
        for num in question_analysis['numbers']:
            mask = col_data.str.contains(num, na=False, case=False)
            if mask.any():
                old_len = len(relevant_data)
                relevant_data = relevant_data[mask]
                if len(relevant_data) < old_len:
                    filters_applied.append(f"contains '{num}' in {col}")
                if len(relevant_data) < 50:
                    break
        
        # Filter by quoted terms
        for term in question_analysis['quoted_terms']:
            mask = col_data.str.contains(term, na=False, case=False)
            if mask.any():
                old_len = len(relevant_data)
                relevant_data = relevant_data[mask]
                if len(relevant_data) < old_len:
                    filters_applied.append(f"contains '{term}' in {col}")
    
    return relevant_data, filters_applied, relevance_score

def create_universal_context(df, question):
    """Create context for ANY CSV file based on the question asked"""
    
    csv_analysis = analyze_csv_structure(df)
    relevant_data, filters_applied, relevance_scores = find_relevant_data_smart(df, question, csv_analysis)
    
    context = []
    context.append(f"=== UNIVERSAL CSV ANALYSIS ===")
    context.append(f"Dataset: {len(df):,} rows, {len(df.columns)} columns")
    context.append(f"Columns: {', '.join(df.columns)}")
    
    # Add column analysis
    context.append(f"\n=== COLUMN ANALYSIS ===")
    for col in df.columns:
        col_type = "Numeric" if col in csv_analysis['numeric_cols'] else "Text"
        if col in csv_analysis['categorical_cols']:
            col_type += " (Categorical)"
        if col in csv_analysis['date_cols']:
            col_type += " (Date/Time)"
        
        context.append(f"'{col}' ({col_type}): {df[col].nunique()} unique values")
        
        # Add sample values
        if df[col].dtype in ['int64', 'float64']:
            context.append(f"  Range: {df[col].min()} to {df[col].max()}")
        else:
            sample_vals = df[col].value_counts().head(5)
            context.append(f"  Top values: {dict(sample_vals)}")
    
    # Add filtering information
    if filters_applied:
        context.append(f"\n=== SMART FILTERING APPLIED ===")
        context.append(f"Filters: {', '.join(filters_applied)}")
        context.append(f"Filtered from {len(df):,} to {len(relevant_data):,} rows")
    
    # Add the actual data
    if len(relevant_data) <= 500:
        context.append(f"\n=== COMPLETE RELEVANT DATA ({len(relevant_data)} rows) ===")
        context.append(relevant_data.to_string(index=False))
    elif len(relevant_data) <= 2000:
        context.append(f"\n=== RELEVANT DATA SAMPLE ({len(relevant_data)} total rows) ===")
        context.append(f"First 200 rows:")
        context.append(relevant_data.head(200).to_string(index=False))
        context.append(f"\nLast 100 rows:")
        context.append(relevant_data.tail(100).to_string(index=False))
    else:
        context.append(f"\n=== STRATEGIC SAMPLE FROM {len(relevant_data)} RELEVANT ROWS ===")
        context.append(f"First 150 rows:")
        context.append(relevant_data.head(150).to_string(index=False))
        context.append(f"\nRandom sample (100 rows):")
        sample = relevant_data.sample(n=min(100, len(relevant_data)-150), random_state=42)
        context.append(sample.to_string(index=False))
        context.append(f"\nLast 50 rows:")
        context.append(relevant_data.tail(50).to_string(index=False))
        
        # Add summary statistics for large datasets
        context.append(f"\n=== SUMMARY STATISTICS ===")
        for col in csv_analysis['categorical_cols']:
            if col in relevant_data.columns:
                counts = relevant_data[col].value_counts()
                context.append(f"{col} counts: {dict(counts.head(10))}")
    
    return "\n".join(context)

def show_prediction_interface(df, cluster_mappings, batch_results, prediction_summary):
    st.subheader("🔮 Smart ML Predictions (Pre-computed)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ML Predictions", len(prediction_summary['ml_predictions']))
    with col2:
        st.metric("Time Forecasts", len(prediction_summary['time_forecasts']))
    with col3:
        st.metric("Total Successful", prediction_summary['total_successful'])

    # Add debugging checkbox
    if st.checkbox("🔍 Show Debug Information", value=False):
        debug_predictions(df, batch_results, prediction_summary)

    if prediction_summary['total_successful'] == 0:
        st.warning("⚠️ No predictions could be generated from this dataset.")
        st.info("💡 This might happen with very small datasets or datasets without sufficient patterns.")
        
        # Show what we tried to process
        st.write("**What we detected:**")
        time_cols = predictions.detect_time_columns(df)
        numeric_cols = predictions.detect_numerical_columns(df)
        st.write(f"- Time columns: {time_cols}")
        st.write(f"- Numeric columns: {numeric_cols}")
        
        if batch_results:
            st.write("**Forecast attempts:**")
            for key, result in list(batch_results.items())[:5]:  # Show first 5
                if isinstance(result, dict):
                    status = result.get('status', 'unknown')
                    reason = result.get('reason', result.get('error', 'No details'))
                    st.write(f"- `{key}`: {status} - {reason}")
        
        return

    with st.expander("🔄 Applied Preprocessing & Clustering"):
        if cluster_mappings:
            for col, mapping in list(cluster_mappings.items())[:6]:
                st.write(f"**{col}**: {mapping['description']}")
                if mapping['type'] == 'year' and mapping['unique_clusters']:
                    years = mapping['unique_clusters']
                    st.write(f"   → Years detected: {min(years)} to {max(years)} ({len(years)} unique)")
        else:
            st.write("No clustering applied to this dataset.")

    if prediction_summary['time_forecasts']:
        st.write("### 📈 Available Time-series Forecasts")
        successful_forecasts = [key for key in prediction_summary['time_forecasts'] if batch_results.get(key, {}).get('status') == 'ok']
        if successful_forecasts:
            forecast_options = {}
            for forecast_key in successful_forecasts:
                if '__by__' in forecast_key:
                    parts = forecast_key.split('__by__')
                    target_col = parts[0]
                    date_col = parts[1]
                    display_name = f"{target_col} forecasted by {date_col}"
                    forecast_options[display_name] = forecast_key
            if forecast_options:
                selected_display = st.selectbox("Select forecast to view:", list(forecast_options.keys()))
                selected_key = forecast_options[selected_display]
                if st.button("📈 Show Time-series Forecast Results"):
                    try:
                        result = batch_results.get(selected_key, {"status": "not_found"})
                        if result.get('status') == 'ok':
                            st.success(f"✅ Time-series Forecast: {selected_display}")
                            forecast_data = result
                            method = result.get('method', 'Unknown')
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Method", method)
                            with col2:
                                if 'mse' in forecast_data:
                                    st.metric("MSE", f"{forecast_data['mse']:.4f}")
                                else:
                                    st.metric("MSE", "N/A")
                            with col3:
                                if 'next_period_prediction' in forecast_data:
                                    st.metric("Next Period Forecast", f"{forecast_data['next_period_prediction']:.2f}")
                                else:
                                    st.metric("Next Forecast", "N/A")
                            
                            # Show historical data if available
                            if 'historical_data' in forecast_data and 'periods' in forecast_data['historical_data']:
                                historical_df = pd.DataFrame({
                                    'Period': forecast_data['historical_data']['periods'],
                                    'Values': forecast_data['historical_data']['values']
                                })
                                st.write("**Historical Data:**")
                                st.dataframe(historical_df)
                                csv = historical_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Historical Data",
                                    csv,
                                    f"forecast_{selected_key}.csv",
                                    "text/csv"
                                )
                            elif 'y_values' in forecast_data and 'periods' in forecast_data:
                                historical_df = pd.DataFrame({
                                    'Period': forecast_data['periods'],
                                    'Values': forecast_data['y_values']
                                })
                                st.write("**Historical Data:**")
                                st.dataframe(historical_df)
                                csv = historical_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Historical Data",
                                    csv,
                                    f"forecast_{selected_key}.csv",
                                    "text/csv"
                                )
                            
                            if 'next_period_prediction' in forecast_data:
                                st.info(f"🔮 **Next Period Forecast**: {forecast_data['next_period_prediction']:.2f}")
                        else:
                            st.error(f"❌ Forecast failed or not found: {result.get('status', 'unknown')}")
                    except Exception as e:
                        st.error(f"❌ Error displaying forecast: {str(e)}")
        else:
            st.warning("⚠️ No successful forecasts found.")
    else:
        st.info("📈 No time-series forecasts available for this dataset.")

def main():
    st.title("EduPredict AI")

    api_key = load_api_key()
    if not api_key:
        st.error("❌ API key not found. Create `secrets.json` with your Gemini or Google API key:")
        st.code('{"GEMINI_API_KEY": "your_gemini_api_key_here"}')
        st.stop()

    uploaded_file = st.file_uploader("📁 Upload ANY CSV File", type="csv")

    if uploaded_file:
        csv_path = "uploaded_data.csv"
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv(csv_path)

        # Sidebar options
        st.sidebar.header("🛠️ Data Structure Options")
        transpose_data = st.sidebar.checkbox("Transpose Data (swap rows and columns)", value=False)
        if transpose_data:
            index_column = st.sidebar.selectbox("Select column to use as new header (if transposing):", df.columns, index=0)
            df = df.set_index(index_column).transpose()
            st.success(f"Data transposed! '{index_column}' switched to columns.")

        st.sidebar.header("🛠️ Handle Missing Values with Iterative Imputer")
        missing_summary = show_missing_summary(df)

        imputation_method = st.sidebar.selectbox(
            "Select Imputation Method",
            ["Basic Iterative Imputer", "Advanced Iterative Imputer (with categorical encoding)"]
        )

        st.sidebar.subheader("⚙️ Imputation Parameters")
        max_iter = st.sidebar.slider("Max Iterations", min_value=1, max_value=20, value=10)
        random_state = st.sidebar.number_input("Random State", min_value=0, max_value=1000, value=42)

        n_nearest_features = None
        if imputation_method == "Basic Iterative Imputer":
            n_nearest_features = st.sidebar.selectbox(
                "Number of features to use",
                [None, 5, 10, 15, "All"],
                index=0
            )
            if n_nearest_features == "All":
                n_nearest_features = None
            elif n_nearest_features is not None:
                n_nearest_features = int(n_nearest_features)

        if missing_summary.empty:
            st.sidebar.info("✅ No missing values found in the dataset!")

        if st.sidebar.button("🚀 Run Iterative Imputation"):
            if missing_summary.empty:
                st.sidebar.warning("⚠️ No missing values to impute!")
            else:
                with st.spinner("Running iterative imputation..."):
                    df_original = df.copy()
                    if imputation_method == "Basic Iterative Imputer":
                        df = iterative_impute(df, max_iter=max_iter, random_state=random_state, n_nearest_features=n_nearest_features)
                    else:
                        df = advanced_iterative_impute(df, max_iter=max_iter, random_state=random_state)
                    compare_imputation(df_original, df)
                    get_imputation_stats(df_original, df)
                st.sidebar.success("✅ Iterative imputation completed!")

        if st.sidebar.checkbox("Show missing values heatmap"):
            visualize_missing(df)

        # Auto-detect CSV characteristics
        csv_analysis = analyze_csv_structure(df)

        # Main content
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum()//1024} KB")

        # Show detected structure
        with st.expander("🔍 Auto-Detected CSV Structure"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Numeric Columns:**", csv_analysis['numeric_cols'])
                st.write("**Date Columns:**", csv_analysis['date_cols'])
            with col2:
                st.write("**Text Columns:**", csv_analysis['text_cols']) 
                st.write("**Categorical Columns:**", csv_analysis['categorical_cols'])

        st.subheader("📖 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("🖼️ Data Visualization")
        if st.checkbox("🔳 Show Visualizations", value=True):
            visualizations_sidebar(df)

        # Enhanced ML predictions
        with st.spinner("🔄 Processing CSV: clustering columns and running batch predictions..."):
            df_processed, cluster_mappings, batch_results, prediction_summary = process_csv_with_predictions(uploaded_file.name, df)

        if st.checkbox("🔮 Enable Smart ML Predictions", value=False):
            show_prediction_interface(df, cluster_mappings, batch_results, prediction_summary)

        st.subheader("💬 Ask Questions About Your Data")
        
        # Dynamic example questions
        example_questions = [""]
        if csv_analysis['numeric_cols']:
            example_questions.extend([
                f"What is the average {csv_analysis['numeric_cols'][0]}?",
                f"What is the total sum of {csv_analysis['numeric_cols']}?"
            ])
        if csv_analysis['categorical_cols']:
            example_questions.extend([
                f"How many different {csv_analysis['categorical_cols'][0]} are there?",
                f"Show me the distribution of {csv_analysis['categorical_cols']}"
            ])
        if csv_analysis['date_cols']:
            example_questions.append(f"What is the date range in {csv_analysis['date_cols'][0]}?")
        example_questions.extend([
            "Summarize this dataset",
            "What are the main patterns?",
            "Find any interesting insights"
        ])

        example = st.selectbox("💡 Smart Example Questions (Auto-Generated)", example_questions)
        user_question = st.text_input("Your question:", value=example, placeholder="Ask anything about your data - works with any CSV structure!")

        if user_question:
            with st.spinner("🤖 Analyzing your data with universal AI intelligence..."):
                try:
                    # Use universal context creation
                    context = create_universal_context(df, user_question)
                    
                    retriever = build_retriever(csv_path, api_key)
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
                    st.markdown("### 📝 Universal Analysis Result:")
                    st.success(answer)

                    # Download report
                    report_content = f"""Universal CSV Analysis Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset: {len(df):,} rows × {len(df.columns)} columns
ML Predictions Available: {prediction_summary['total_successful']}

Question: {user_question}
Answer: {answer}

Detected Structure:
- Numeric columns: {csv_analysis['numeric_cols']}
- Text columns: {csv_analysis['text_cols']}
- Categorical columns: {csv_analysis['categorical_cols']}
- Date columns: {csv_analysis['date_cols']}

ML Predictions: {prediction_summary['ml_predictions']}
Time Forecasts: {prediction_summary['time_forecasts']}
"""
                    
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=report_content,
                        file_name=f"universal_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Try rephrasing your question")

    else:
        st.info("👆 Upload ANY CSV file to start universal analysis with ML predictions and AI Q&A!")
        st.markdown("""
        ### 🌟 Enhanced Universal Features:
        - **🔄 Auto-Detection**: Automatically understands any CSV structure
        - **🧠 Smart Clustering**: Year rounding, numeric binning, categorical grouping  
        - **🤖 Batch ML Predictions**: Pre-computed ML models for all columns
        - **📈 Time-series Forecasting**: Automatic trend analysis and forecasting
        - **🎯 Smart Filtering**: Finds relevant data based on your question
        - **📏 Size Adaptation**: Handles small to massive datasets
        - **📊 Universal Visualization**: Interactive charts for any data structure
        - **💬 LangChain AI Q&A**: Natural language querying with context understanding
        - **🛠️ Missing Value Handling**: Advanced iterative imputation methods
        - **🔄 Data Transformation**: Transpose and restructure data as needed

        ### 📋 Supported Data Types:
        - Sales data, Financial records, Student data, Product catalogs
        - Survey responses, Scientific measurements, Log files  
        - Inventory data, Customer records, Time series data
        - **And literally ANY other CSV structure!**

        ### 🔮 ML Capabilities:
        - **Intelligent Preprocessing**: Automatic year clustering, binning
        - **Batch Predictions**: All possible ML models computed automatically
        - **Smart Model Selection**: Auto-chooses regression vs classification
        - **Time-series Forecasting**: ARIMA-based future value prediction
        - **Feature Importance**: Identifies most predictive columns
        """)

if __name__ == "__main__":
    main()