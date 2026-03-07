# planning_ui.py - Planning Studio User Interface

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app.core.planning import PlanningStudio

def render_planning_studio(df: pd.DataFrame, api_key: str, mapping_result: dict):
    """Render the Planning Studio tab"""
    
    st.subheader("🎯 Planning Studio - AI-Powered Decision Support")
    st.markdown("Set targets, simulate scenarios, detect anomalies, and get AI recommendations.")
    
    # Initialize Planning Studio
    planning = PlanningStudio(api_key)
    
    # Get metrics and time column from schema mapping
    metrics = mapping_result.get("metrics", [])
    time_col = mapping_result.get("time_col")
    dimensions = mapping_result.get("dimensions", [])
    
    if not metrics or not time_col:
        st.warning("⚠️ No metrics or time column detected. Please ensure your data has numeric metrics and a time column.")
        return
    
    # Metric selector
    selected_metric = st.selectbox("📊 Select Metric to Analyze", metrics)
    
    # Create tabs for different planning features
    plan_tab1, plan_tab2, plan_tab3, plan_tab4 = st.tabs([
        "🎯 Target vs Actual",
        "🔮 What-If Scenarios", 
        "⚠️ Anomaly Detection",
        "💡 AI Recommendations"
    ])
    
    # TAB 1: Target vs Actual
    with plan_tab1:
        st.subheader("Target vs Actual Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            target_mode = st.radio("Target Mode", ["Growth-based", "Manual"], horizontal=True)
        
        with col2:
            if target_mode == "Growth-based":
                growth_rate = st.slider("Annual Growth Rate (%)", -20, 50, 5) / 100
                manual_target = None
            else:
                manual_target = st.number_input(f"Target {selected_metric}", value=float(df[selected_metric].mean()))
                growth_rate = 0.05
        
        if st.button("📊 Calculate Target Analysis", key="calc_target"):
            with st.spinner("Analyzing..."):
                result = planning.calculate_target_vs_actual(
                    df, selected_metric, time_col,
                    target_mode="growth" if target_mode == "Growth-based" else "manual",
                    growth_rate=growth_rate,
                    manual_target=manual_target
                )
                
                if "error" not in result:
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    achievement_rates = list(result['achievement_rate'].values())
                    col1.metric("Avg Achievement Rate", f"{np.mean(achievement_rates):.1f}%")
                    col2.metric("Best Period", f"{max(achievement_rates):.1f}%")
                    col3.metric("Worst Period", f"{min(achievement_rates):.1f}%")
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    periods = list(result['historical'].keys())
                    historical_vals = list(result['historical'].values())
                    target_vals = list(result['targets'].values())
                    
                    fig.add_trace(go.Scatter(
                        x=periods, y=historical_vals,
                        mode='lines+markers', name='Actual',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=periods, y=target_vals,
                        mode='lines+markers', name='Target',
                        line=dict(color='#ff7f0e', width=3, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Target vs Actual: {selected_metric}",
                        xaxis_title="Period",
                        yaxis_title=selected_metric,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gap analysis table
                    st.subheader("Gap Analysis")
                    gap_df = pd.DataFrame({
                        "Period": periods,
                        "Actual": historical_vals,
                        "Target": target_vals,
                        "Gap": list(result['gaps'].values()),
                        "Achievement %": [f"{v:.1f}%" for v in achievement_rates]
                    })
                    st.dataframe(gap_df, use_container_width=True)
                else:
                    st.error(result['error'])
    
    # TAB 2: What-If Scenarios
    with plan_tab2:
        st.subheader("What-If Scenario Simulator")
        
        # Natural language input
        st.markdown("**💬 Describe your scenario in plain English:**")
        scenario_text = st.text_input(
            "Scenario description",
            placeholder=f"What if {selected_metric} increases by 15%?",
            key="scenario_text"
        )
        
        if scenario_text and st.button("🔮 Parse Scenario", key="parse_scenario"):
            with st.spinner("Parsing with AI..."):
                parsed = planning.parse_scenario_from_text(scenario_text, metrics)
                
                if "error" not in parsed:
                    st.success("✅ Scenario parsed successfully!")
                    st.json(parsed)
                    
                    # Run simulation
                    sim_result = planning.simulate_scenario(df, selected_metric, time_col, parsed)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Baseline Total", f"{sim_result['baseline'].values().__iter__().__next__():,.0f}")
                    col2.metric("Simulated Impact", f"{sim_result['impact']:+,.0f}", 
                               delta=f"{sim_result['impact_pct']:+.1f}%")
                    
                    # Visualization
                    fig = go.Figure()
                    periods = list(sim_result['baseline'].keys())
                    
                    fig.add_trace(go.Bar(
                        x=periods, y=list(sim_result['baseline'].values()),
                        name='Baseline', marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=periods, y=list(sim_result['simulated'].values()),
                        name='Simulated', marker_color='orange'
                    ))
                    
                    fig.update_layout(
                        title=f"Scenario Impact: {selected_metric}",
                        xaxis_title="Period",
                        yaxis_title=selected_metric,
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(parsed['error'])
        
        # Manual scenario sliders
        st.markdown("**🎚️ Or use manual adjustments:**")
        adjustment = st.slider(f"Adjust {selected_metric} by (%)", -50, 100, 0)
        
        if adjustment != 0 and st.button("🚀 Run Manual Scenario", key="manual_scenario"):
            multiplier = 1 + (adjustment / 100)
            sim_result = planning.simulate_scenario(
                df, selected_metric, time_col, 
                {selected_metric: multiplier}
            )
            
            col1, col2 = st.columns(2)
            col1.metric("Impact", f"{sim_result['impact']:+,.0f}")
            col2.metric("Change", f"{sim_result['impact_pct']:+.1f}%")
    
    # TAB 3: Anomaly Detection
    with plan_tab3:
        st.subheader("Anomaly Detection & Root Cause Analysis")
        
        threshold = st.slider("Sensitivity (Z-score threshold)", 1.0, 3.0, 2.0, 0.1)
        
        if st.button("🔍 Detect Anomalies", key="detect_anomalies"):
            with st.spinner("Analyzing..."):
                anomalies = planning.detect_anomalies(df, selected_metric, time_col, threshold)
                
                if anomalies:
                    st.warning(f"⚠️ Found {len(anomalies)} anomalies")
                    
                    # Display anomalies table
                    anomaly_df = pd.DataFrame(anomalies)
                    st.dataframe(anomaly_df, use_container_width=True)
                    
                    # AI Explanation for first anomaly
                    if api_key:
                        st.subheader("🤖 AI Root Cause Analysis")
                        selected_anomaly = st.selectbox(
                            "Select anomaly to explain",
                            range(len(anomalies)),
                            format_func=lambda i: f"{anomalies[i]['period']} ({anomalies[i]['severity']})"
                        )
                        
                        if st.button("💡 Explain This Anomaly", key="explain_anomaly"):
                            with st.spinner("AI is analyzing..."):
                                explanation = planning.explain_anomaly(
                                    df, selected_metric, 
                                    anomalies[selected_anomaly],
                                    dimensions
                                )
                                st.markdown(explanation)
                else:
                    st.success("✅ No significant anomalies detected")
    
    # TAB 4: AI Recommendations
    with plan_tab4:
        st.subheader("AI-Powered Recommendations")
        
        if not api_key:
            st.warning("⚠️ API key required for AI recommendations")
            return
        
        if st.button("💡 Generate Recommendations", key="gen_recommendations"):
            with st.spinner("AI is analyzing your data..."):
                # Get target analysis
                target_result = planning.calculate_target_vs_actual(
                    df, selected_metric, time_col
                )
                
                # Get anomalies
                anomalies = planning.detect_anomalies(df, selected_metric, time_col)
                
                # Generate executive summary
                st.subheader("📋 Executive Summary")
                summary = planning.generate_executive_summary(
                    target_result, anomalies, selected_metric
                )
                st.markdown(summary)
                
                st.divider()
                
                # Generate recommendations
                st.subheader("🎯 Action Recommendations")
                recommendations = planning.get_recommendations(
                    df, selected_metric, target_result, anomalies
                )
                st.markdown(recommendations)
                
                # Download button
                report = f"""
# Planning Studio Report
## Metric: {selected_metric}

## Executive Summary
{summary}

## Recommendations
{recommendations}

Generated by EduPredict AI Planning Studio
"""
                st.download_button(
                    "📥 Download Full Report",
                    report,
                    f"planning_report_{selected_metric}.md",
                    mime="text/markdown"
                )

import numpy as np
