# main.py - Application Entry Point with Tabs

import streamlit as st
import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="EduPredict AI", page_icon="📊", layout="wide")

from app.ui.sidebar import render_sidebar
from app.ui.dashboard import process_csv_intelligent, show_prediction_interface
from app.ui.visualizations import visualizations_sidebar
from app.core.data_handler import show_missing_summary, iterative_impute, advanced_iterative_impute
from app.core.analyzer import analyze_csv_structure, create_universal_context
from app.core.llm import build_retriever, get_answer_from_llm
from app.core.predictor import get_segmented_series, arima_forecast_students, aggregate_by_period_for_target
import plotly.graph_objects as go

def load_api_key():
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if api_key:
        return api_key
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
    
    uploaded_file = st.file_uploader("📁 Upload ANY CSV File", type="csv")
    
    if uploaded_file:
        csv_path = f"data/raw/{uploaded_file.name}"
        os.makedirs("data/raw", exist_ok=True)
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        df = pd.read_csv(csv_path)
        
        # Show file info
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum()//1024} KB")
        
        # Check for missing values
        missing_summary = show_missing_summary(df)
        if missing_summary.empty:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.warning(f"⚠️ Missing values detected in {len(missing_summary)} columns!")
            st.info("💡 **Suggestion:** Go to **🔧 Data Processing** tab to handle missing values.")
        
        # Smart transpose suggestion
        def suggest_transpose(df):
            """Detect if data might benefit from transposing"""
            # Check if first column looks like years/dates and other columns are metrics
            first_col = df.iloc[:, 0]
            first_col_str = first_col.astype(str)
            
            # Check for year-like patterns in first column
            year_pattern = first_col_str.str.match(r'^\d{4}').sum() > len(df) * 0.5
            date_pattern = first_col_str.str.contains(r'\d{4}[-/]\d{1,2}', na=False).sum() > len(df) * 0.5
            
            # Check if most other columns are numeric
            numeric_cols = df.select_dtypes(include=['number']).columns
            mostly_numeric = len(numeric_cols) > len(df.columns) * 0.7
            
            # Check for unusual structure: many columns, few rows
            wide_structure = len(df.columns) > 10 and len(df) < 20
            
            # Check if column names look like years/dates
            col_names_are_years = sum(str(col).isdigit() and len(str(col)) == 4 for col in df.columns) > len(df.columns) * 0.5
            
            if col_names_are_years or (wide_structure and mostly_numeric):
                return True, "Column names appear to be years/periods. Consider transposing for time-series analysis."
            elif year_pattern or date_pattern:
                return False, None  # Already in correct format
            elif wide_structure:
                return True, "Dataset has many columns but few rows. Transposing might improve analysis."
            
            return False, None
        
        should_transpose, transpose_reason = suggest_transpose(df)
        if should_transpose:
            st.info(f"💡 **Suggestion:** {transpose_reason} Go to **🔧 Data Processing** tab to transpose.")
        
        # Create Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Data Upload", "🔧 Data Processing", "📊 Visualization", "🤖 ML Predictions", "💬 AI Q&A"])
        
        # TAB 1: Data Upload (already shown above)
        with tab1:
            st.info("✅ File uploaded successfully! Use the tabs above to process and analyze your data.")
            
            # Data Preview - only visible in Tab 1
            st.subheader("📖 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
        
        # TAB 2: Data Processing
        with tab2:
            st.subheader("🔧 Data Processing Options")
            
            # Transpose option
            transpose_data = st.checkbox("Transpose Data (swap rows and columns)", value=False)
            
            if transpose_data:
                index_column = st.selectbox("Select column to use as new header:", df.columns, index=0)
                df = df.set_index(index_column).transpose()
                
                if not df.columns.is_unique:
                    st.warning("⚠️ Duplicate headers found. Auto-renaming...")
                    new_cols = []
                    seen = {}
                    for c in df.columns:
                        c_str = str(c)
                        if c_str in seen:
                            seen[c_str] += 1
                            new_cols.append(f"{c_str}_{seen[c_str]}")
                        else:
                            seen[c_str] = 0
                            new_cols.append(c_str)
                    df.columns = new_cols
                
                df = df.reset_index()
                
                if 'index' in df.columns:
                    sample = df['index'].astype(str).head(5).tolist()
                    if any(len(s) == 4 and s.isdigit() for s in sample):
                        df = df.rename(columns={'index': 'Year'})
                        st.caption("ℹ️ Renamed 'index' to 'Year'")
                
                st.success(f"✅ Data transposed! '{index_column}' switched to columns.")
            
            # Missing values handling
            st.subheader("🛠️ Handle Missing Values")
            
            missing_summary = show_missing_summary(df)
            
            if not missing_summary.empty:
                st.dataframe(missing_summary, use_container_width=True)
                
                imputation_method = st.selectbox(
                    "Select Imputation Method",
                    ["Basic Iterative Imputer", "Advanced Iterative Imputer (with categorical encoding)"]
                )
                
                st.write("⚙️ Imputation Parameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_iter = st.slider("Max Iterations", 1, 20, 10)
                with col2:
                    random_state = st.number_input("Random State", 0, 1000, 42)
                with col3:
                    n_nearest_features = None
                    if imputation_method == "Basic Iterative Imputer":
                        n_feat = st.selectbox("Number of features to use", ["None", "5", "10", "15"])
                        if n_feat != "None":
                            n_nearest_features = int(n_feat)
                
                if st.button("🚀 Run Imputation"):
                    with st.spinner("Running imputation..."):
                        if imputation_method == "Basic Iterative Imputer":
                            df = iterative_impute(df, max_iter=max_iter, random_state=random_state, n_nearest_features=n_nearest_features)
                        else:
                            df = advanced_iterative_impute(df, max_iter=max_iter, random_state=random_state)
                    st.success("✅ Imputation completed!")
            else:
                st.success("✅ No missing values found!")
        
        # TAB 3: Visualization
        with tab3:
            visualizations_sidebar(df)
        
        # TAB 4: ML Predictions
        with tab4:
            with st.spinner("🔄 Processing: Standardization → Clustering → Forecasting..."):
                df_processed, cluster_mappings, batch_results, prediction_summary, mapping_result = process_csv_intelligent(df, api_key)
            
            if mapping_result.get("error") == "Non-Academic Data Detected":
                st.error(f"🛑 {mapping_result['error']}")
                st.warning(f"⚠️ {mapping_result['message']}")
                st.info("Please upload a valid academic dataset.")
            else:
                if "full_mapping" in mapping_result:
                    with st.expander("🧠 AI Schema Standardization Details"):
                        st.json(mapping_result["full_mapping"])
                
                show_prediction_interface(df, cluster_mappings, batch_results, prediction_summary)
                
                # Drill-down forecasting
                st.divider()
                st.subheader("🔬 Segmented Forecasting (Drill-Down)")
                st.markdown("Forecast for specific categories (e.g., specific Degree Type, Department).")
                
                dimensions = mapping_result.get("dimensions", [])
                metrics = mapping_result.get("metrics", [])
                time_col = mapping_result.get("time_col")
                
                if (dimensions and metrics and time_col) or (metrics and time_col):
                    use_dimension_mode = len(dimensions) > 0
                    
                    if use_dimension_mode:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            sel_dim = st.selectbox("Select Dimension", dimensions)
                        with c2:
                            unique_vals = sorted(df[sel_dim].unique().tolist())
                            sel_val = st.selectbox("Select Segment", unique_vals)
                        with c3:
                            sel_metric = st.selectbox("Select Metric", metrics)
                    else:
                        st.info("ℹ️ No categorical dimensions found.")
                        sel_dim = None
                        sel_val = "All Data"
                        sel_metric = st.selectbox("Select Target Column", metrics)
                    
                    if st.button("🚀 Forecast Segment"):
                        with st.spinner(f"Forecasting {sel_metric}..."):
                            if use_dimension_mode:
                                series = get_segmented_series(df, sel_dim, sel_val, time_col, sel_metric)
                            else:
                                series = aggregate_by_period_for_target(df, time_col, sel_metric)
                            
                            if series.empty:
                                st.warning("No data found.")
                            else:
                                result = arima_forecast_students(series)
                                
                                if result['status'] == 'ok':
                                    m1, m2 = st.columns(2)
                                    m1.metric("Next Period Forecast", f"{result['next_period_prediction']:,.0f}")
                                    m2.metric("Model", result['method'])
                                    
                                    st.write(f"### Forecast Trend: {sel_metric} ({sel_val})")
                                    
                                    hist_years = result['historical_data']['periods']
                                    hist_values = result['historical_data']['values']
                                    last_year = int(max(hist_years))
                                    pred_year = last_year + 1
                                    pred_value = result['next_period_prediction']
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=hist_years, y=hist_values,
                                        mode='lines+markers', name='Historical',
                                        line=dict(color='#1f77b4', width=3)
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=[hist_years[-1], pred_year], y=[hist_values[-1], pred_value],
                                        mode='lines+markers', name='Forecast Trend',
                                        line=dict(color='#ff7f0e', width=3, dash='dash'),
                                        marker=dict(symbol='star', size=12)
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"Forecast: {sel_metric} ({sel_val})",
                                        xaxis_title="Year", yaxis_title=sel_metric,
                                        hovermode="x unified"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error(f"Forecasting failed: {result.get('error')}")
                else:
                    st.info("Insufficient dimensions or metrics for drill-down.")
        
        # TAB 5: AI Q&A
        with tab5:
            st.subheader("💬 Ask Questions About Your Data")
            
            csv_analysis = analyze_csv_structure(df)
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
                            retriever = build_retriever(csv_path, api_key)
                            answer = get_answer_from_llm(user_question, context, retriever, api_key)
                            
                            st.markdown("### 📝 Analysis Result:")
                            st.success(answer)
                            
                            report = f"Question: {user_question}\nAnswer: {answer}"
                            st.download_button("📥 Download Report", report, "analysis.txt")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
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
