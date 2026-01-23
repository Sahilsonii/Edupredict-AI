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
