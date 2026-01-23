# sidebar.py - Application Sidebar Logic

import streamlit as st
from ..core.data_handler import show_missing_summary, iterative_impute, advanced_iterative_impute, visualize_missing, compare_imputation, get_imputation_stats
from ..core.analyzer import analyze_csv_structure

def render_sidebar(df):
    """
    Renders the sidebar and returns the modified dataframe and settings
    """
    st.sidebar.header("🛠️ Data Structure Options")
    transpose_data = st.sidebar.checkbox("Transpose Data (swap rows and columns)", value=False)
    
    if transpose_data:
        index_column = st.sidebar.selectbox("Select column to use as new header (if transposing):", df.columns, index=0)
        df = df.set_index(index_column).transpose()
        
        # Ensure unique columns to prevent analyzer errors (e.g., duplicates from the index)
        if not df.columns.is_unique:
             st.sidebar.warning("⚠️ Duplicate headers found after transpose. Auto-renaming...")
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
        
        # Reset index so that the index (e.g., Year) becomes a column again
        df = df.reset_index()
        
        # Smart Rename: If the new column is generic 'index' but contains years, rename it!
        if 'index' in df.columns:
            sample = df['index'].astype(str).head(5).tolist()
            # Simple check for 4-digit years
            if any(len(s) == 4 and s.isdigit() for s in sample):
                df = df.rename(columns={'index': 'Year'})
                st.sidebar.caption("ℹ️ Renamed 'index' to 'Year' for better detection.")
            else:
                st.sidebar.caption("ℹ️ Index reset to column for analysis.")
        
        st.sidebar.success(f"Data transposed! '{index_column}' switched to columns.")

    st.sidebar.header("🛠️ Handle Missing Values")
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
        st.sidebar.info("✅ No missing values found!")

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
                
                # Show comparison in main area (or sidebar? original was sidebar calls but display was likely mixed)
                # We'll return the imputing stats to display in dashboard if needed, or display here
                compare_imputation(df_original, df)
                get_imputation_stats(df_original, df)
            st.sidebar.success("✅ Imputation completed!")

    if st.sidebar.checkbox("Show missing values heatmap"):
        visualize_missing(df)

    # Auto-detect CSV characteristics
    csv_analysis = analyze_csv_structure(df)
    
    return df, csv_analysis
