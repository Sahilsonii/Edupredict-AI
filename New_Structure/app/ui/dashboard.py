# dashboard.py - Main Dashboard Logic

import streamlit as st
import datetime
import pandas as pd
from .visualizations import visualizations_sidebar
from ..core.predictor import cluster_all_columns, detect_time_columns, detect_numerical_columns, batch_forecast_backend
from ..core.analyzer import create_universal_context
from ..core.llm import build_retriever, get_answer_from_llm

def process_csv_with_predictions(uploaded_file_name, df_csv):
    """
    Process CSV with clustering and batch predictions.
    """
    try:
        # Step 1: Apply intelligent clustering
        df_processed, cluster_mappings = cluster_all_columns(
            df_csv, numeric_bins=8, cat_top_k=15, treat_years=True, keep_original=True
        )
        
        # Step 2: Run batch predictions
        time_cols = detect_time_columns(df_processed)
        numeric_cols = detect_numerical_columns(df_processed)
        
        st.info(f"🔍 Debug: Found {len(time_cols)} time columns, {len(numeric_cols)} numeric columns")
        
        if not time_cols or not numeric_cols:
            return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}
        
        target_cols = [col for col in numeric_cols if col not in time_cols]
        if not target_cols:
            return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}
        
        batch_results = batch_forecast_backend(
            df_processed, potential_time_cols=time_cols, target_cols=target_cols
        )
        
        # Step 3: Create summary
        successful_predictions = []
        successful_forecasts = []
        
        if isinstance(batch_results, dict) and batch_results.get('status') != 'error':
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
        
        return df_processed, cluster_mappings, batch_results, prediction_summary
        
    except Exception as e:
        st.error(f"❌ Prediction processing failed: {str(e)}")
        return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}

def show_prediction_interface(df, cluster_mappings, batch_results, prediction_summary):
    st.subheader("🔮 Smart ML Predictions (Pre-computed)")

    col1, col2, col3 = st.columns(3)
    col1.metric("ML Predictions", len(prediction_summary['ml_predictions']))
    col2.metric("Time Forecasts", len(prediction_summary['time_forecasts']))
    col3.metric("Total Successful", prediction_summary['total_successful'])

    if prediction_summary['time_forecasts']:
        st.write("### 📈 Available Time-series Forecasts")
        successful_forecasts = [key for key in prediction_summary['time_forecasts'] if batch_results.get(key, {}).get('status') == 'ok']
        if successful_forecasts:
            forecast_options = {}
            for forecast_key in successful_forecasts:
                if '__by__' in forecast_key:
                    parts = forecast_key.split('__by__')
                    display_name = f"{parts[0]} forecasted by {parts[1]}"
                    forecast_options[display_name] = forecast_key
            
            if forecast_options:
                selected_display = st.selectbox("Select forecast to view:", list(forecast_options.keys()))
                selected_key = forecast_options[selected_display]
                
                if st.button("📈 Show Time-series Forecast Results"):
                    result = batch_results.get(selected_key)
                    st.json(result) # Simplified display for now, can be enhanced

def render_dashboard(df, csv_analysis, api_key, csv_path):
    """
    Renders the main dashboard area
    """
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum()//1024} KB")

    # Structure info
    with st.expander("🔍 Auto-Detected CSV Structure"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numeric Columns:**", csv_analysis['numeric_cols'])
            st.write("**Date Columns:**", csv_analysis['date_cols'])
        with col2:
            st.write("**Text Columns:**", csv_analysis['text_cols']) 
            st.write("**Categorical Columns:**", csv_analysis['categorical_cols'])

    # Data Preview
    st.subheader("📖 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Visualization
    st.subheader("🖼️ Data Visualization")
    if st.checkbox("🔳 Show Visualizations", value=True):
        visualizations_sidebar(df)

    # ML Predictions
    with st.spinner("🔄 Processing CSV: clustering columns and running batch predictions..."):
        # Note: We need a temporary filename or passing valid name if from uploaded_file
        df_processed, cluster_mappings, batch_results, prediction_summary = process_csv_with_predictions("data", df)

    if st.checkbox("🔮 Enable Smart ML Predictions", value=False):
        show_prediction_interface(df, cluster_mappings, batch_results, prediction_summary)

    # Chat / Q&A
    st.subheader("💬 Ask Questions About Your Data")
    
    example_questions = ["Summarize this dataset", "What are the main patterns?"]
    if csv_analysis['numeric_cols']:
        example_questions.append(f"What is the average {csv_analysis['numeric_cols'][0]}?")
    
    example = st.selectbox("💡 Smart Example Questions", example_questions)
    user_question = st.text_input("Your question:", value=example)

    if user_question:
        if not api_key:
            st.error("❌ API Key missing")
        else:
            with st.spinner("🤖 Analyzing..."):
                try:
                    context = create_universal_context(df, user_question)
                    retriever = build_retriever(csv_path, api_key) # Requires file path on disk
                    answer = get_answer_from_llm(user_question, context, retriever, api_key)
                    
                    st.markdown("### 📝 Analysis Result:")
                    st.success(answer)
                    
                    # Download report
                    report = f"Question: {user_question}\nAnswer: {answer}"
                    st.download_button("📥 Download Report", report, "analysis.txt")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
