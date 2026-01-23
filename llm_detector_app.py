"""
LLM Column Detector - Standalone App
Upload ANY CSV and get automatic column analysis using AI
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from llm_column_detector import LLMColumnDetector, auto_analyze_csv

# Page config
st.set_page_config(
    page_title="AI Column Detector",
    page_icon="🤖",
    layout="wide"
)

# Load API key
@st.cache_data
def load_api_key():
    secrets_file = Path("secrets.json")
    if secrets_file.exists():
        with open(secrets_file, 'r') as f:
            secrets = json.load(f)
            return secrets.get('GEMINI_API_KEY')
    return None

# Main app
def main():
    st.title("🤖 AI-Powered Column Detector")
    st.markdown("Upload **ANY CSV** and let AI automatically understand your data!")
    
    # Check API key
    api_key = load_api_key()
    if not api_key:
        st.error("❌ API key not found. Create `secrets.json` with your Gemini API key:")
        st.code('{"GEMINI_API_KEY": "your_key_here"}')
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")
    
    if uploaded_file:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        
        # Show preview
        st.subheader("📊 Data Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Size", f"{uploaded_file.size // 1024} KB")
        
        st.dataframe(df.head(10), use_container_width=True)
        
        # Analyze button
        if st.button("🤖 Analyze with AI", type="primary"):
            with st.spinner("🧠 AI is analyzing your columns..."):
                try:
                    # Run analysis
                    result = auto_analyze_csv(df, api_key)
                    
                    # Store in session state
                    st.session_state['analysis'] = result
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.stop()
        
        # Display results if available
        if 'analysis' in st.session_state:
            result = st.session_state['analysis']
            
            # Domain detection
            st.subheader("🎯 Dataset Domain")
            col1, col2 = st.columns(2)
            col1.metric("Detected Domain", result['domain'].title())
            col2.metric("Confidence", f"{result['confidence']}%")
            
            # Column analysis
            st.subheader("📋 Column Analysis")
            
            analysis_data = []
            for col in result['analysis']['columns']:
                analysis_data.append({
                    'Column Name': col['name'],
                    'Semantic Type': col['semantic_type'],
                    'Category': col['category'],
                    'Best Visualization': col['visualization_type'],
                    'Description': col['description']
                })
            
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df, use_container_width=True)
            
            # Suggested analyses
            st.subheader("💡 AI-Suggested Analyses")
            
            for i, suggestion in enumerate(result['analysis']['suggested_analyses'], 1):
                with st.expander(f"📊 Suggestion {i}: {suggestion['description']}"):
                    col1, col2 = st.columns(2)
                    col1.write(f"**Type:** {suggestion['type']}")
                    col2.write(f"**Visualization:** {suggestion['visualization']}")
                    st.write(f"**Columns:** {', '.join(suggestion['columns'])}")
            
            # Category breakdown
            st.subheader("📊 Column Categories")
            
            categories = {}
            for col in result['analysis']['columns']:
                cat = col['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(col['name'])
            
            cols = st.columns(len(categories))
            for idx, (category, columns) in enumerate(categories.items()):
                with cols[idx]:
                    st.metric(category.replace('_', ' ').title(), len(columns))
                    st.write(", ".join(columns))
            
            # Export results
            st.subheader("💾 Export Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as JSON
                json_str = json.dumps(result['analysis'], indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    f"analysis_{uploaded_file.name}.json",
                    "application/json"
                )
            
            with col2:
                # Export as CSV
                csv_str = analysis_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_str,
                    f"analysis_{uploaded_file.name}",
                    "text/csv"
                )
            
            # Raw JSON (collapsible)
            with st.expander("🔍 View Raw JSON Response"):
                st.json(result['analysis'])
    
    else:
        # Instructions
        st.info("👆 Upload a CSV file to get started!")
        
        st.markdown("""
        ### 🌟 What This Tool Does:
        
        1. **Automatic Column Understanding** 🧠
           - Detects what each column represents
           - Identifies data types and categories
           - No manual configuration needed!
        
        2. **Smart Categorization** 🏷️
           - Time-based columns (dates, years)
           - Metrics (counts, scores, amounts)
           - Categorical data (labels, groups)
           - Identifiers (IDs, codes)
           - Geographic data (locations)
        
        3. **Visualization Suggestions** 📊
           - Best chart type for each column
           - Recommended analyses
           - Column relationships
        
        4. **Domain Detection** 🎯
           - Automatically identifies dataset type
           - Education, Finance, Healthcare, Retail, etc.
           - Confidence score included
        
        ### 📝 Example Use Cases:
        
        - **Educational Data**: Student records, enrollment, grades
        - **Financial Data**: Sales, revenue, transactions
        - **Healthcare Data**: Patient records, treatments
        - **E-commerce Data**: Products, orders, customers
        - **ANY CSV**: Works with any structured data!
        
        ### 🚀 Try It Now:
        Upload your CSV file above and click "Analyze with AI"
        """)

if __name__ == "__main__":
    main()
