# dashboard.py - Main Dashboard Logic

import streamlit as st
import datetime
import pandas as pd
from .visualizations import visualizations_sidebar
from ..core.predictor import cluster_all_columns, detect_time_columns, detect_numerical_columns, batch_forecast_backend
from ..core.analyzer import create_universal_context
from ..core.llm import build_retriever, get_answer_from_llm
from ..core.schema_mapper import SchemaMapper

def process_csv_intelligent(df_csv, api_key):
    """
    Process CSV using LLM Schema Standardization -> Predictions
    """
    try:
        # 1. Schema Standardization (The "Smart" Layer)
        if api_key:
            mapper = SchemaMapper(api_key)
            mapping_result = mapper.standardize(df_csv)
            st.success("✅ AI Schema Standardization Complete")
            
            # Use standardized columns
            time_cols = [mapping_result['time_col']] if mapping_result['time_col'] else []
            target_cols = mapping_result['metrics']
        else:
            # Fallback to heuristics if no key
            mapping_result = {"error": "No API Key"}
            time_cols = detect_time_columns(df_csv)
            target_cols = [c for c in detect_numerical_columns(df_csv) if c not in time_cols]

        # 2. Intelligent Clustering
        df_processed, cluster_mappings = cluster_all_columns(
            df_csv, numeric_bins=8, cat_top_k=15, treat_years=True, keep_original=True
        )
        
        # 3. Batch Predictions (using standardized columns)
        st.info(f"🔍 Forecasting for: {len(target_cols)} metrics over {time_cols}")
        
        if not time_cols or not target_cols:
             return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}, mapping_result

        batch_results = batch_forecast_backend(
            df_processed, potential_time_cols=time_cols, target_cols=target_cols
        )
        
        # 4. Create summary
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
        
        return df_processed, cluster_mappings, batch_results, prediction_summary, mapping_result
        
    except Exception as e:
        st.error(f"❌ Prediction processing failed: {str(e)}")
        return df_csv, {}, {}, {"ml_predictions": [], "time_forecasts": [], "total_successful": 0}, {}

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
        from .visualizations import visualizations_sidebar
        visualizations_sidebar(df)

    # ML Predictions
    with st.spinner("🔄 Processing CSV: Standardization -> Clustering -> Forecasting..."):
        # Note: We need a temporary filename or passing valid name if from uploaded_file
        df_processed, cluster_mappings, batch_results, prediction_summary, mapping_result = process_csv_intelligent(df, api_key)

    # Check for Domain Validation Error
    if mapping_result.get("error") == "Non-Academic Data Detected":
        st.error(f"🛑 {mapping_result['error']}")
        st.warning(f"⚠️ {mapping_result['message']}")
        st.info("Please upload a valid academic dataset (e.g., student enrollment, university rankings, grades).")
        return # Stop further rendering

    if st.checkbox("🔮 Show Smart ML Predictions", value=True):
        if "full_mapping" in mapping_result:
             with st.expander("🧠 AI Schema Standardization Details"):
                 st.json(mapping_result["full_mapping"])
        show_prediction_interface(df, cluster_mappings, batch_results, prediction_summary)
        
        # --- NEW: Drill-Down Forecasting ---
        st.divider()
        st.subheader("🔬 Segmented Forecasting (Drill-Down)")
        st.markdown("Forecast for specific categories (e.g., specific Degree Type, Department).")
        
        dimensions = mapping_result.get("dimensions", [])
        metrics = mapping_result.get("metrics", [])
        time_col = mapping_result.get("time_col")

        if (dimensions and metrics and time_col) or (metrics and time_col):
            
            # Determine mode: Filtered (Dimension-based) or Direct (Metric-based)
            use_dimension_mode = len(dimensions) > 0
            
            if use_dimension_mode:
                c1, c2, c3 = st.columns(3)
                with c1:
                    sel_dim = st.selectbox("1. Select Dimension", dimensions)
                with c2:
                    # Get unique values for this dimension
                    unique_vals = sorted(df[sel_dim].unique().tolist())
                    sel_val = st.selectbox("2. Select Segment", unique_vals)
                with c3:
                    sel_metric = st.selectbox("3. Select Metric", metrics)
            else:
                # Direct mode (for Wide datasets / Transposed)
                st.info("ℹ️ No categorical dimensions found. Select a column to forecast directly.")
                sel_dim = None
                sel_val = "All Data"
                sel_metric = st.selectbox("Select Target Column to Forecast", metrics)

            if st.button("🚀 Forecast Segment"):
                with st.spinner(f"Forecasting {sel_metric}..."):
                    from ..core.predictor import get_segmented_series, arima_forecast_students, aggregate_by_period_for_target
                    
                    # 1. Get Series
                    if use_dimension_mode:
                        series = get_segmented_series(df, sel_dim, sel_val, time_col, sel_metric)
                    else:
                        # Direct aggregation (for wide datasets)
                        series = aggregate_by_period_for_target(df, time_col, sel_metric)
                    
                    if series.empty:
                        st.warning("No data found for this combination.")
                    else:
                        # 2. Forecast
                        result = arima_forecast_students(series)
                        
                        if result['status'] == 'ok':
                            # Display Result
                            m1, m2 = st.columns(2)
                            m1.metric("Next Period Forecast", f"{result['next_period_prediction']:,.0f}")
                            m2.metric("Model", result['method'])
                            
                            st.write(f"### Forecast Trend: {sel_metric} ({sel_val})")
                            
                            # Prepare Data for Plotly
                            hist_years = result['historical_data']['periods']
                            hist_values = result['historical_data']['values']
                            
                            last_year = int(max(hist_years))
                            pred_year = last_year + 1
                            pred_value = result['next_period_prediction']
                            
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            
                            # 1. Historical Data (Solid Line)
                            fig.add_trace(go.Scatter(
                                x=hist_years, 
                                y=hist_values,
                                mode='lines+markers',
                                name='Historical',
                                line=dict(color='#1f77b4', width=3)
                            ))
                            
                            # 2. Forecast Trend Line (Dashed, connecting last point -> prediction)
                            # We must include the last historical point to make the line continuous
                            fig.add_trace(go.Scatter(
                                x=[hist_years[-1], pred_year],
                                y=[hist_values[-1], pred_value],
                                mode='lines+markers',
                                name='Forecast Trend',
                                line=dict(color='#ff7f0e', width=3, dash='dash'),
                                marker=dict(symbol='star', size=12)
                            ))
                            
                            # 3. Confidence Interval (Shaded Area)
                            if 'confidence_interval' in result:
                                fig.add_trace(go.Scatter(
                                    x=[pred_year, pred_year],
                                    y=[result['confidence_interval']['lower'], result['confidence_interval']['upper']],
                                    mode='markers',
                                    name='Confidence Interval',
                                    error_y=dict(
                                        type='data',
                                        symmetric=False,
                                        array=[result['confidence_interval']['upper'] - pred_value],
                                        arrayminus=[pred_value - result['confidence_interval']['lower']],
                                        visible=True
                                    ),
                                    marker=dict(color='rgba(0,0,0,0)') # Invisible marker, just showing error bar
                                ))
                            
                            fig.update_layout(
                                title=f"Forecast: {sel_metric} ({sel_val})",
                                xaxis_title="Year",
                                yaxis_title=sel_metric,
                                hovermode="x unified",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            st.error(f"Forecasting failed: {result.get('error') or result.get('detail')}")
                            
        else:
            st.info("Insufficient dimensions or metrics identified for drill-down.")


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
